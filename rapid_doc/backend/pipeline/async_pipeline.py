"""
진정한 비동기 파이프라인 처리 모듈
DX Engine run_async() + register_callback을 활용한 CPU-하드웨어 오버랩 문서 분석

Stage DAG (각 스테이지는 모든 페이지 작업을 한꺼번에 배치 제출):
    Stage 1: Layout      — 전(全) 페이지 run_async 병렬 제출
    Stage 2: 영역 플래닝  — layout 결과 → OCR/테이블/수식 후보 분류 (CPU)
    Stage 3: Formula     — 전(全) 수식 영역 batch_predict
    Stage 4: PDF-det     — PDF 텍스트 직접 추출 (모델 없음, CPU)
    Stage 5: OCR-det     — 전(全) OCR 후보 영역 배치 검출
    Stage 6: Table       — 전(全) 테이블 순차 처리 (OCR-det 결과 포함)
    Stage 7: OCR-rec     — 전(全) 텍스트 크롭 일괄 인식
"""

import os
import time
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict

import cv2
import numpy as np
from loguru import logger
from tqdm import tqdm

from .model_init import MineruPipelineModel, AtomModelSingleton
from .pipeline_analyze import custom_model_init
from .model_list import AtomicModel
from ...utils.config_reader import get_formula_enable, get_table_enable, get_device
from ...utils.enum_class import CategoryId
from ...utils.model_utils import crop_img, get_res_list_from_layout_res, clean_memory
from ...utils.ocr_utils import (
    merge_det_boxes, update_det_boxes, sorted_boxes,
    get_adjusted_mfdetrec_res, get_ocr_result_list,
    OcrConfidence, get_ocr_result_list_table,
)
from ...utils.span_pre_proc import (
    txt_spans_bbox_extract, extract_table_fill_image, txt_most_angle_extract_table,
)
from ...utils.boxbase import rotate_image_and_boxes
from ...utils.checkbox_det_cls import checkbox_predict

# ─── 상수 ────────────────────────────────────────────────────────────────────
_TABLE_OPEN_TAG = '<table>'
_TABLE_CLOSE_TAG = '</table>'


# ─────────────────────────────────────────────────────────────────────────────
# 페이지 컨텍스트: 한 페이지의 모든 처리 상태를 추적
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PageContext:
    """파이프라인을 흐르는 페이지 단위 상태 객체"""

    # 식별자
    pdf_idx: int
    page_idx: int

    # 입력 데이터
    np_img: Optional[np.ndarray] = None
    scale: float = 1.0
    ocr_enable: bool = False
    lang: str = "ch"
    page_dict: Optional[dict] = None

    # Stage 1: Layout 결과
    layout_res: list = field(default_factory=list)

    # Stage 2: 영역 후보 (layout 해석 결과)
    ocr_candidates: list = field(default_factory=list)   # OCR 후보 영역 (res dict)
    table_candidates: list = field(default_factory=list) # 테이블 후보 (table_img 포함)
    formula_regions: list = field(default_factory=list)  # 수식 검출 영역 (mfdetrec_res)
    formula_crops: list = field(default_factory=list)    # 크롭된 수식 이미지
    checkbox_res: list = field(default_factory=list)     # 체크박스 결과


# ─────────────────────────────────────────────────────────────────────────────
# 진정한 비동기 파이프라인
# ─────────────────────────────────────────────────────────────────────────────

class TrueAsyncPipeline:
    """
    DX Engine run_async() 기반 파이프라인.

    핵심 원칙:
      • 각 스테이지는 모든 페이지의 작업을 모아 일제히 제출한 뒤 전체 결과를 수집한다.
      • use_async=True 플래그가 설정된 경우, DX 하드웨어는 여러 요청을 동시에 처리한다.
      • CPU 측 전·후처리는 하드웨어 추론과 최대한 오버랩한다.
    """

    def __init__(
        self,
        model: MineruPipelineModel,
        formula_enable: bool = True,
        table_enable: bool = True,
        use_det_mode: str = 'auto',
        layout_config: dict = None,
        ocr_config: dict = None,
        formula_config: dict = None,
        table_config: dict = None,
        checkbox_config: dict = None,
        verbose: bool = False,
    ):
        self.model = model
        self.formula_enable = get_formula_enable(formula_enable)
        self.table_enable = get_table_enable(table_enable)
        self.use_det_mode = use_det_mode
        self.layout_config = layout_config or {}
        self.ocr_config = ocr_config or {}
        self.formula_config = formula_config or {}
        self.table_config = table_config or {}
        self.checkbox_config = checkbox_config or {}
        self.verbose = verbose

        # 세부 옵션
        self.checkbox_enable = self.checkbox_config.get("checkbox_enable", False)
        self.formula_rec_enable = self.formula_config.get("formula_rec_enable", True)
        self.formula_level = self.formula_config.get("formula_level", 0)
        self.table_force_ocr = self.table_config.get("force_ocr", False)
        self.skip_text_in_image = self.table_config.get("skip_text_in_image", True)
        self.use_img2table = self.table_config.get("use_img2table", False)

        self.atom_model_manager = AtomModelSingleton()

        # 성능 통계
        self.perf_stats: Dict[str, Dict] = {}
        self.pdf_perf_stats = defaultdict(lambda: defaultdict(lambda: {'time': 0.0, 'count': 0}))

    # ─────────────────────────────── public ──────────────────────────────────

    def run(
        self,
        images_with_extra_info: List[Tuple],
    ) -> Tuple[List[Any], Dict]:
        """
        파이프라인 전체 실행.

        Returns:
            (images_layout_res 리스트, pdf_perf_stats 딕셔너리)
        """
        total = len(images_with_extra_info)
        logger.info(f"🚀 TrueAsyncPipeline: {total} pages")
        t_total = time.perf_counter()

        contexts = self._build_page_contexts(images_with_extra_info)

        self._stage_layout(contexts)            # Stage 1
        self._stage_plan_regions(contexts)      # Stage 2
        if self.formula_enable and self.formula_rec_enable:
            self._stage_formula(contexts)       # Stage 3
        elif self.formula_enable:
            logger.info("[Stage 3/7] Formula — rec disabled (formula_rec_enable=False), kept as image")
        self._stage_pdf_det(contexts)           # Stage 4
        self._stage_ocr_det(contexts)           # Stage 5
        if self.table_enable:
            self._stage_table(contexts)         # Stage 6
        self._stage_ocr_rec(contexts)           # Stage 7

        elapsed = time.perf_counter() - t_total
        logger.info(
            f"✅ TrueAsyncPipeline: {total} pages in {elapsed:.2f}s "
            f"({total / max(elapsed, 0.001):.2f} it/s)"
        )
        self._print_perf_summary()

        results = [ctx.layout_res for ctx in contexts]
        return results, dict(self.pdf_perf_stats)

    # ─────────────────────────── 입력 파싱 ───────────────────────────────────

    def _build_page_contexts(self, items: List[Tuple]) -> List[PageContext]:
        contexts = []
        for item in items:
            if len(item) == 7:
                img, scale, ocr_enable, lang, page_dict, pdf_idx, page_idx = item
            else:
                img, scale, ocr_enable, lang, page_dict = item
                pdf_idx, page_idx = 0, len(contexts)

            np_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            contexts.append(PageContext(
                pdf_idx=pdf_idx,
                page_idx=page_idx,
                np_img=np_img,
                scale=scale,
                ocr_enable=ocr_enable,
                lang=lang,
                page_dict=page_dict,
            ))
        return contexts

    # ─────────────────────────── Stage 1: Layout ─────────────────────────────

    def _stage_layout(self, contexts: List[PageContext]) -> None:
        """
        모든 페이지 이미지를 layout_model.batch_predict에 일제히 전달.
        use_async=True 모델에서는 내부적으로 run_async를 페이지별로 제출하고
        callback을 통해 결과를 수집하므로 하드웨어가 병렬 처리한다.
        """
        n = len(contexts)
        logger.info(f"[Stage 1/7] Layout — {n} pages")
        t0 = time.perf_counter()

        np_images = [ctx.np_img for ctx in contexts]
        batch_size = self.layout_config.get("batch_num", 1)

        all_layout_res = self.model.layout_model.batch_predict(np_images, batch_size)

        for ctx, layout_res in zip(contexts, all_layout_res):
            # formula_level 필터링
            if self.formula_enable and self.formula_level == 1:
                layout_res = [item for item in layout_res if item["category_id"] != 13]
            ctx.layout_res = layout_res

        elapsed = time.perf_counter() - t0
        self._record_perf('layout', elapsed, n, contexts)
        logger.info(f"   ↳ {elapsed:.3f}s | {n / max(elapsed, 0.001):.2f} it/s")

    # ─────────────────────────── Stage 2: 영역 플래닝 ────────────────────────

    def _stage_plan_regions(self, contexts: List[PageContext]) -> None:
        """
        layout 결과를 해석해 OCR 후보·테이블 후보·수식 영역·체크박스를 분류한다 (CPU only).
        """
        for ctx in contexts:
            ocr_candidates, table_candidates, formula_regions = get_res_list_from_layout_res(
                ctx.layout_res, ctx.np_img
            )

            # 체크박스 검출
            checkbox_res = []
            if self.checkbox_enable:
                checkbox_img = cv2.cvtColor(ctx.np_img, cv2.COLOR_RGB2BGR)
                checkbox_res = checkbox_predict(checkbox_img)
                for res in checkbox_res:
                    poly = [
                        res['bbox'][0], res['bbox'][1],
                        res['bbox'][2], res['bbox'][1],
                        res['bbox'][2], res['bbox'][3],
                        res['bbox'][0], res['bbox'][3],
                    ]
                    ctx.layout_res.append({
                        'bbox': res['bbox'], 'poly': poly,
                        'category_id': CategoryId.CheckBox,
                        'checkbox': res['text'], 'score': 0.9,
                    })

            ctx.ocr_candidates = list(ocr_candidates)
            ctx.checkbox_res = checkbox_res
            ctx.formula_regions = list(formula_regions)  # mfdetrec_res (latex 채워질 예정)

            # 테이블 후보: crop 이미지 포함
            ctx.table_candidates = []
            for tr in table_candidates:
                table_img, useful_list = crop_img(tr, ctx.np_img)
                ctx.table_candidates.append({
                    'table_res': tr,
                    'table_img': table_img,
                    'useful_list': useful_list,
                    'ocr_enable': ctx.ocr_enable,
                })

            # 수식 crop 이미지
            ctx.formula_crops = []
            for fr in formula_regions:
                latex_img, _ = crop_img(fr, ctx.np_img)
                ctx.formula_crops.append(latex_img)

    # ─────────────────────────── Stage 3: Formula ────────────────────────────

    def _stage_formula(self, contexts: List[PageContext]) -> None:
        """
        모든 페이지의 수식 crop 이미지를 모아 formula_model.batch_predict로 일괄 처리.
        formula_regions 항목에 latex 필드를 in-place로 채운다.
        """
        # (ctx, formula_region_ref, crop_img) 수집
        all_items = []
        all_crops = []
        for ctx in contexts:
            for fr_dict, crop in zip(ctx.formula_regions, ctx.formula_crops):
                all_items.append(fr_dict)
                all_crops.append(crop)

        if not all_crops:
            logger.info("[Stage 3/7] Formula — 없음 (skip)")
            return

        n = len(all_crops)
        logger.info(f"[Stage 3/7] Formula — {n} regions")
        t0 = time.perf_counter()

        batch_size = self.formula_config.get("batch_num", 1)
        latex_results = self.model.formula_model.batch_predict(all_crops, batch_size=batch_size)

        success = 0
        for fr_dict, latex in zip(all_items, latex_results):
            if latex:
                fr_dict['latex'] = latex
                success += 1

        elapsed = time.perf_counter() - t0
        self._record_perf('formula', elapsed, n, contexts)
        logger.info(f"   ↳ {elapsed:.3f}s | {n / max(elapsed, 0.001):.2f} it/s | success={success}/{n}")

    # ─────────────────────────── Stage 4: PDF-det ────────────────────────────

    def _stage_pdf_det(self, contexts: List[PageContext]) -> None:
        """
        텍스트 기반 PDF에서 텍스트 위치를 직접 추출한다 (모델 추론 없음, CPU only).
        스캔 PDF(ocr_enable=True) 또는 ocr 강제 모드면 건너뛴다.
        """
        if self.use_det_mode == 'ocr':
            return

        t0 = time.perf_counter()
        count = 0

        for ctx in contexts:
            if ctx.ocr_enable:
                continue
            for res in ctx.ocr_candidates:
                new_image, useful_list = crop_img(
                    res, ctx.np_img, crop_paste_x=50, crop_paste_y=50
                )
                adjusted = get_adjusted_mfdetrec_res(
                    ctx.formula_regions + ctx.checkbox_res, useful_list
                )
                bgr_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
                ocr_res = txt_spans_bbox_extract(
                    ctx.page_dict, res, mfd_res=adjusted,
                    scale=ctx.scale, useful_list=useful_list,
                )
                if ocr_res:
                    result_list = get_ocr_result_list(
                        ocr_res, useful_list, ctx.ocr_enable, bgr_image, ctx.lang
                    )
                    ctx.layout_res.extend(result_list)
                    res['_pdf_det_done'] = True
                    count += 1

        elapsed = time.perf_counter() - t0
        if count > 0:
            self._record_perf('pdf_det', elapsed, count, contexts)
            if self.verbose:
                logger.info(f"[Stage 4/7] PDF-det — {count} regions in {elapsed:.3f}s")

    # ─────────────────────────── Stage 5: OCR-det ────────────────────────────

    def _should_skip_ocr_det(self, ctx: PageContext, res: dict) -> bool:
        """OCR-det를 건너뛰어야 하는 영역이면 True를 반환한다."""
        if ctx.ocr_enable:
            return False
        if self.use_det_mode == 'txt':
            return True
        return (
            self.use_det_mode != 'ocr'
            and not res.get('need_ocr_det')
            and bool(res.get('_pdf_det_done'))
        )

    def _collect_ocr_det_items(self, contexts: List[PageContext]) -> list:
        """OCR-det 처리 대상 이미지 목록을 수집해 반환한다."""
        all_items = []
        for ctx in contexts:
            for res in ctx.ocr_candidates:
                if self._should_skip_ocr_det(ctx, res):
                    continue
                res.pop('need_ocr_det', None)
                new_image, useful_list = crop_img(
                    res, ctx.np_img, crop_paste_x=50, crop_paste_y=50
                )
                adjusted = get_adjusted_mfdetrec_res(
                    ctx.formula_regions + ctx.checkbox_res, useful_list
                )
                bgr_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
                all_items.append((ctx, res, adjusted, bgr_image, useful_list))
        return all_items

    def _stage_ocr_det(self, contexts: List[PageContext]) -> None:
        """
        OCR이 필요한 모든 영역을 모아 배치로 검출한다.

        배치 경로 (RapidOcrModel): det_batch_predict를 사용해 해상도별 그룹핑 후 일괄 처리.
        개별 경로 (DxOcrModel):    ocr_model.ocr(img, rec=False) 개별 처리
                                   (DX Engine은 내부 run_async로 하드웨어 병렬화).
        """
        ocr_model = self.atom_model_manager.get_atom_model(
            atom_model_name=AtomicModel.OCR,
            det_db_box_thresh=0.3,
            ocr_config=self.ocr_config,
        )

        all_items = self._collect_ocr_det_items(contexts)

        if not all_items:
            if self.verbose:
                logger.info("[Stage 5/7] OCR-det — 없음 (skip)")
            return

        n = len(all_items)
        logger.info(f"[Stage 5/7] OCR-det — {n} regions")
        t0 = time.perf_counter()
        count = 0

        # 배치 경로 (RapidOcrModel.det_batch_predict 지원 여부 확인)
        if hasattr(ocr_model, 'det_batch_predict'):
            count = self._ocr_det_batch(ocr_model, all_items)
        else:
            count = self._ocr_det_single(ocr_model, all_items)

        elapsed = time.perf_counter() - t0
        if count > 0:
            self._record_perf('ocr_det', elapsed, count, contexts)
        logger.info(f"   ↳ {elapsed:.3f}s | {count} boxes | {count / max(elapsed, 0.001):.2f} it/s")

    def _ocr_det_batch(self, ocr_model, all_items) -> int:
        """해상도별 그룹핑 후 det_batch_predict로 일괄 처리 (RapidOcrModel용)."""
        STRIDE = 64
        resolution_groups = defaultdict(list)
        for item in all_items:
            bgr_image = item[3]   # (ctx, res, adjusted, bgr_image, useful_list)
            h, w = bgr_image.shape[:2]
            nh = ((h + STRIDE) // STRIDE) * STRIDE
            nw = ((w + STRIDE) // STRIDE) * STRIDE
            resolution_groups[(nh, nw)].append(item)

        count = 0
        for _group_key, group_items in tqdm(resolution_groups.items(), desc="OCR-det batch"):
            count += self._process_ocr_det_group(ocr_model, group_items, STRIDE)
        return count

    def _process_ocr_det_group(
        self, ocr_model, group_items: list, stride: int
    ) -> int:
        """단일 해상도 그룹의 OCR-det 배치를 실행하고 처리된 박스 수를 반환한다."""
        max_h = max(it[3].shape[0] for it in group_items)
        max_w = max(it[3].shape[1] for it in group_items)
        th = ((max_h + stride - 1) // stride) * stride
        tw = ((max_w + stride - 1) // stride) * stride

        batch_imgs = []
        for _ctx, _res, _adj, bgr_image, _ul in group_items:
            h, w = bgr_image.shape[:2]
            padded = np.ones((th, tw, 3), dtype=np.uint8) * 255
            padded[:h, :w] = bgr_image
            batch_imgs.append(padded)

        batch_results = ocr_model.det_batch_predict(batch_imgs, len(batch_imgs))
        count = 0
        for item, (dt_boxes, _) in zip(group_items, batch_results):
            ctx, _res, adjusted, bgr_image, useful_list = item
            count += self._apply_det_boxes(ctx, dt_boxes, adjusted, bgr_image, useful_list)
        return count

    def _apply_det_boxes(
        self,
        ctx: PageContext,
        dt_boxes,
        adjusted,
        bgr_image: np.ndarray,
        useful_list: list,
    ) -> int:
        """검출된 박스를 후처리해 layout_res에 추가하고 처리된 박스 수를 반환한다."""
        if dt_boxes is None or len(dt_boxes) == 0:
            return 0
        dt_sorted = sorted_boxes(dt_boxes)
        dt_merged = merge_det_boxes(dt_sorted) if dt_sorted else []
        dt_final = (
            update_det_boxes(dt_merged, adjusted)
            if (dt_merged and adjusted) else dt_merged
        )
        ocr_res = [b.tolist() if hasattr(b, 'tolist') else b for b in dt_final]
        if not ocr_res:
            return 0
        result_list = get_ocr_result_list(
            ocr_res, useful_list, ctx.ocr_enable, bgr_image, None
        )
        ctx.layout_res.extend(result_list)
        return 1

    def _ocr_det_single(self, ocr_model, all_items) -> int:
        """개별 이미지 처리 (DxOcrModel 등 det_batch_predict 미지원 모델용)."""
        count = 0
        for ctx, _res, adjusted, bgr_image, useful_list in tqdm(all_items, desc="OCR-det"):
            ocr_res = ocr_model.ocr(bgr_image, mfd_res=adjusted, rec=False)[0]
            if ocr_res:
                result_list = get_ocr_result_list(
                    ocr_res, useful_list, ctx.ocr_enable, bgr_image, None
                )
                ctx.layout_res.extend(result_list)
                count += 1
        return count

    # ─────────────────────────── Stage 6: Table ──────────────────────────────

    def _prepare_table_ocr_result(
        self,
        ocr_model_for_table,
        ctx: PageContext,
        ti: dict,
    ) -> list:
        """테이블 영역의 OCR 결과를 [boxes, texts, scores] 형태로 반환한다."""
        adjusted = get_adjusted_mfdetrec_res(
            ctx.formula_regions + ctx.checkbox_res,
            ti['useful_list'],
            return_text=True,
        )
        new_table_img = cv2.cvtColor(np.asarray(ti['table_img']), cv2.COLOR_RGB2BGR)
        raw_ocr = ocr_model_for_table.ocr(new_table_img, mfd_res=adjusted, rec=False)[0] or []

        most_angle = txt_most_angle_extract_table(ctx.page_dict, ti, scale=ctx.scale)
        if most_angle in [90, 270] and raw_ocr:
            ti['table_img'], raw_ocr = rotate_image_and_boxes(
                np.asarray(ti['table_img']), raw_ocr, most_angle
            )

        if not raw_ocr:
            return [[], [], []]

        ocr_spans = get_ocr_result_list_table(raw_ocr, ti['useful_list'], ctx.scale)
        if not ocr_spans:
            return [[], [], []]
        return [list(x) for x in zip(*[
            [s['ori_bbox'], s['content'], s['score']] for s in ocr_spans
        ])]

    def _stage_table(self, contexts: List[PageContext]) -> None:
        """
        모든 테이블 후보를 모아 순차 처리한다.
        table_model.predict 호출 전에 OCR-det를 통해 텍스트 박스를 준비하여
        Table 모델 내부의 중복 OCR 실행을 방지한다.
        """
        all_items = [
            (ctx, ti)
            for ctx in contexts
            for ti in ctx.table_candidates
        ]

        if not all_items:
            if self.verbose:
                logger.info("[Stage 6/7] Table — 없음 (skip)")
            return

        n = len(all_items)
        logger.info(f"[Stage 6/7] Table — {n} tables")
        t0 = time.perf_counter()

        table_model = self.atom_model_manager.get_atom_model(
            atom_model_name='table',
            ocr_config=self.ocr_config,
            table_config=self.table_config,
        )
        ocr_model_for_table = self.atom_model_manager.get_atom_model(
            atom_model_name=AtomicModel.OCR,
            ocr_show_log=False,
            det_db_box_thresh=0.3,
            ocr_config=self.ocr_config,
            enable_merge_det_boxes=False,
        )

        with tqdm(total=n, desc="Table Predict") as pbar:
            for ctx, ti in all_items:
                adjusted = get_adjusted_mfdetrec_res(
                    ctx.formula_regions + ctx.checkbox_res,
                    ti['useful_list'],
                    return_text=True,
                )
                ocr_result = self._prepare_table_ocr_result(ocr_model_for_table, ctx, ti)
                fill_image_res = extract_table_fill_image(ctx.page_dict, ti, scale=ctx.scale)

                html_code, _, _, _ = table_model.predict(
                    ti['table_img'], ocr_result, fill_image_res,
                    adjusted, self.skip_text_in_image, self.use_img2table,
                )
                self._apply_table_html(ti, html_code)
                pbar.update(1)

        elapsed = time.perf_counter() - t0
        self._record_perf('table', elapsed, n, contexts)
        logger.info(f"   ↳ {elapsed:.3f}s | {n / max(elapsed, 0.001):.2f} it/s")

    # ─────────────────────────── Stage 7: OCR-rec ────────────────────────────

    def _stage_ocr_rec(self, contexts: List[PageContext]) -> None:
        """
        category_id=15(OcrText) 항목의 텍스트 크롭을 모아 일괄 인식한다.
        DxOcrModel(use_async=True)의 경우 텍스트 인식기가 항목별로 run_async를
        제출하고 wait_request로 수집한다.
        """
        need_ocr_list = []
        img_crop_list = []

        for ctx in contexts:
            for item in ctx.layout_res:
                if item.get('category_id') == 15 and 'np_img' in item:
                    item['_pdf_idx'] = ctx.pdf_idx
                    need_ocr_list.append(item)
                    img_crop_list.append(item.pop('np_img'))
                    item.pop('lang', None)

        if not img_crop_list:
            return

        n = len(img_crop_list)
        logger.info(f"[Stage 7/7] OCR-rec — {n} crops")
        t0 = time.perf_counter()

        ocr_model = self.atom_model_manager.get_atom_model(
            atom_model_name=AtomicModel.OCR,
            det_db_box_thresh=0.3,
            ocr_config=self.ocr_config,
        )
        ocr_res_list = ocr_model.ocr(img_crop_list, det=False, tqdm_enable=True)[0]

        assert len(ocr_res_list) == len(need_ocr_list), (
            f"ocr_res_list 길이 불일치: {len(ocr_res_list)} vs {len(need_ocr_list)}"
        )

        for item, (text, score) in zip(need_ocr_list, ocr_res_list):
            item.pop('_pdf_idx', None)
            item['text'] = text
            item['score'] = float(f"{score:.3f}")
            if score < OcrConfidence.min_confidence:
                item['category_id'] = 16

        elapsed = time.perf_counter() - t0
        self._record_perf('ocr_rec', elapsed, n, contexts)
        logger.info(f"   ↳ {elapsed:.3f}s | {n / max(elapsed, 0.001):.2f} it/s")

    # ─────────────────────────── 통계 헬퍼 ──────────────────────────────────

    @staticmethod
    def _apply_table_html(ti: dict, html_code: Optional[str]) -> None:
        """Table 모델 결과 HTML을 검증 후 table_res에 저장한다."""
        if html_code and _TABLE_OPEN_TAG in html_code and _TABLE_CLOSE_TAG in html_code:
            s = html_code.find(_TABLE_OPEN_TAG)
            e = html_code.rfind(_TABLE_CLOSE_TAG) + len(_TABLE_CLOSE_TAG)
            ti['table_res']['html'] = html_code[s:e]
        else:
            logger.warning("Table recognition: HTML 테이블 태그 미발견")

    def _record_perf(
        self,
        key: str,
        elapsed: float,
        count: int,
        contexts: List[PageContext],
    ) -> None:
        self.perf_stats[key] = {'time': elapsed, 'count': count}
        # PDF별 통계 누적
        per_item = elapsed / max(count, 1)
        pdf_counts: Dict[int, int] = defaultdict(int)
        for ctx in contexts:
            pdf_counts[ctx.pdf_idx] += 1
        for pdf_idx, n in pdf_counts.items():
            self.pdf_perf_stats[pdf_idx][key]['time'] += per_item * n
            self.pdf_perf_stats[pdf_idx][key]['count'] += n

    def _print_perf_summary(self) -> None:
        if not self.perf_stats:
            return

        total = sum(v['time'] for v in self.perf_stats.values())
        labels = {
            'layout':  '📊 Layout  ',
            'formula': '📐 Formula ',
            'pdf_det': '📄 PDF-det ',
            'ocr_det': '🔍 OCR-det ',
            'table':   '📋 Table   ',
            'ocr_rec': '✍️  OCR-rec ',
        }

        logger.info("=" * 80)
        logger.info("📈 TrueAsyncPipeline Performance Summary")
        logger.info("=" * 80)

        for key in ['layout', 'formula', 'pdf_det', 'ocr_det', 'table', 'ocr_rec']:
            if key not in self.perf_stats:
                continue
            s = self.perf_stats[key]
            t, c = s['time'], s['count']
            pct = t / total * 100 if total > 0 else 0
            logger.info(
                f"{labels.get(key, key)} | {t:7.2f}s ({pct:5.1f}%) | "
                f"{c:4d}it | {t / max(c, 1):.3f} s/it | {c / max(t, 0.001):6.2f} it/s"
            )

        logger.info(f"{'─' * 80}")
        logger.info(f"🔥 Total: {total:.2f}s")
        logger.info("=" * 80)


# ─────────────────────────────────────────────────────────────────────────────
# 공개 API (pipeline_analyze.py 에서 호출)
# ─────────────────────────────────────────────────────────────────────────────

def async_batch_image_analyze(
    images_with_extra_info: List[Tuple],
    formula_enable: bool = True,
    table_enable: bool = True,
    layout_config: dict = None,
    ocr_config: dict = None,
    formula_config: dict = None,
    table_config: dict = None,
    checkbox_config: dict = None,
    input_interval: float = 0.0,   # pipeline_analyze.py 하위 호환 — 미사용
    verbose: bool = False,
) -> Tuple[List[Any], Dict]:
    """
    TrueAsyncPipeline을 사용한 배치 이미지 분석 (공개 API).

    pipeline_analyze.py → use_async_pipeline=True 경로에서 호출된다.
    """
    # use_async=True 로 복사체 생성 (원본 dict 수정 방지)
    ocr_config = {**(ocr_config or {}), 'use_async': True}
    layout_config = {**(layout_config or {}), 'use_async': True}
    table_config = {**(table_config or {}), 'use_async': True}

    use_det_mode = ocr_config.get('use_det_mode', 'auto')

    model = custom_model_init(
        lang=None,
        formula_enable=formula_enable,
        table_enable=table_enable,
        layout_config=layout_config,
        ocr_config=ocr_config,
        formula_config=formula_config,
        table_config=table_config,
    )

    pipeline = TrueAsyncPipeline(
        model=model,
        formula_enable=formula_enable,
        table_enable=table_enable,
        use_det_mode=use_det_mode,
        layout_config=layout_config,
        ocr_config=ocr_config,
        formula_config=formula_config,
        table_config=table_config,
        checkbox_config=checkbox_config,
        verbose=verbose,
    )

    results, pdf_perf_stats = pipeline.run(images_with_extra_info)
    clean_memory(get_device())
    return results, pdf_perf_stats


# backward-compatible alias (pipeline_analyze.py의 기존 호출부가 input_interval을 넘기는 경우 대비)
def _async_batch_image_analyze_compat(
    images_with_extra_info,
    formula_enable=True,
    table_enable=True,
    layout_config=None,
    ocr_config=None,
    formula_config=None,
    table_config=None,
    checkbox_config=None,
    input_interval=0.0,   # 하위 호환을 위해 수신하지만 미사용
    verbose=False,
):
    """input_interval 파라미터를 포함한 하위 호환 래퍼."""
    return async_batch_image_analyze(
        images_with_extra_info,
        formula_enable=formula_enable,
        table_enable=table_enable,
        layout_config=layout_config,
        ocr_config=ocr_config,
        formula_config=formula_config,
        table_config=table_config,
        checkbox_config=checkbox_config,
        verbose=verbose,
    )
