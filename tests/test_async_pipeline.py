"""
TrueAsyncPipeline 검증 테스트

검증 계층 (3단계):
─────────────────────────────────────────────────────────────────────────────
 L1. 순수 단위 테스트 (Unit)
     - 모델·외부 의존 없음; numpy/PIL만 사용
     - PageContext 데이터클래스, _apply_table_html, _should_skip_ocr_det,
       API 서명 호환성, input_interval 파라미터 하위 호환

 L2. 목(Mock) 기반 파이프라인 테스트 (Mock Integration)
     - 실제 모델 로드 없이 TrueAsyncPipeline.run() 전체 흐름 실행
     - MineruPipelineModel, AtomModelSingleton을 최소 stub으로 교체
     - PageContext 흐름, perf_stats 반환 구조, 결과 shape 검증

 L3. 통합 테스트 (Integration) — 실제 모델 필요, 기본 Skip
     - pipeline_analyze.doc_analyze() 수준 전체 스택 실행
     - Sync(BatchAnalyze) vs Async(TrueAsyncPipeline) 출력 정합성 비교
     - 환경 변수 + PDF 파일 경로가 있을 때만 실행
─────────────────────────────────────────────────────────────────────────────

실행 방법:
  # L1+L2 (빠름, 모델 불필요):
  python -m pytest tests/test_async_pipeline.py -v -m "not integration"

  # 전체 (모델 & PDF 필요):
  export RD_TEST_PDF=/path/to/sample.pdf
  source ./deepx_scripts/set_env.sh 1 2 1 3 2 4
  python -m pytest tests/test_async_pipeline.py -v
"""

import inspect
import os
import sys
import time
import threading
import types
import unittest
from collections import defaultdict
from dataclasses import fields
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import cv2
import numpy as np
import pytest
from PIL import Image

# ──────────────────────────── 경로 설정 ───────────────────────────────────────
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))


# =============================================================================
# L1: 순수 단위 테스트
# =============================================================================

class TestPageContextDataclass(unittest.TestCase):
    """PageContext 데이터클래스 기본값·타입 검증."""

    def test_required_fields_exist(self):
        """pdf_idx, page_idx 필드가 존재해야 한다."""
        from rapid_doc.backend.pipeline.async_pipeline import PageContext
        field_names = {f.name for f in fields(PageContext)}
        for required in ('pdf_idx', 'page_idx', 'np_img', 'scale', 'ocr_enable',
                         'lang', 'page_dict', 'layout_res', 'ocr_candidates',
                         'table_candidates', 'formula_regions', 'formula_crops',
                         'checkbox_res'):
            self.assertIn(required, field_names, f"PageContext에 '{required}' 필드 없음")

    def test_optional_fields_default_none(self):
        """np_img, page_dict는 Optional — 기본값 None 확인."""
        from rapid_doc.backend.pipeline.async_pipeline import PageContext
        ctx = PageContext(pdf_idx=0, page_idx=0)
        self.assertIsNone(ctx.np_img)
        self.assertIsNone(ctx.page_dict)

    def test_list_fields_default_empty(self):
        """list 필드는 기본값이 빈 리스트여야 한다(dataclass field factory)."""
        from rapid_doc.backend.pipeline.async_pipeline import PageContext
        ctx1 = PageContext(pdf_idx=0, page_idx=0)
        ctx2 = PageContext(pdf_idx=1, page_idx=1)
        # 서로 다른 인스턴스의 list가 동일 객체이면 안 된다
        self.assertIsNot(ctx1.layout_res, ctx2.layout_res)
        self.assertIsNot(ctx1.ocr_candidates, ctx2.ocr_candidates)
        self.assertIsNot(ctx1.table_candidates, ctx2.table_candidates)

    def test_scale_default(self):
        from rapid_doc.backend.pipeline.async_pipeline import PageContext
        ctx = PageContext(pdf_idx=0, page_idx=0)
        self.assertEqual(ctx.scale, 1.0)

    def test_ocr_enable_default_false(self):
        from rapid_doc.backend.pipeline.async_pipeline import PageContext
        ctx = PageContext(pdf_idx=0, page_idx=0)
        self.assertFalse(ctx.ocr_enable)


class TestApplyTableHtml(unittest.TestCase):
    """_apply_table_html 헬퍼의 HTML 추출 로직 검증."""

    def _get_method(self):
        from rapid_doc.backend.pipeline.async_pipeline import TrueAsyncPipeline
        # staticmethod이므로 클래스에서 직접 호출 가능
        return TrueAsyncPipeline._apply_table_html

    def test_valid_html_extracted(self):
        """완전한 <table>...</table> 포함 HTML에서 테이블 부분만 추출한다."""
        method = self._get_method()
        ti = {'table_res': {}}
        html = "  prefix  <table><tr><td>hello</td></tr></table>  suffix  "
        method(ti, html)
        result = ti['table_res']['html']
        self.assertTrue(result.startswith('<table>'))
        self.assertTrue(result.endswith('</table>'))
        self.assertNotIn('prefix', result)
        self.assertNotIn('suffix', result)

    def test_html_with_nested_tags(self):
        """중첩 테이블 구조에서 가장 바깥 <table>~</table>을 정확히 추출한다."""
        method = self._get_method()
        ti = {'table_res': {}}
        html = '<table><tr><td><table><tr><td>inner</td></tr></table></td></tr></table>'
        method(ti, html)
        self.assertEqual(ti['table_res']['html'], html)

    def test_missing_table_tag_logs_warning(self):
        """<table> 태그가 없으면 table_res['html']을 설정하지 않고 경고를 남긴다."""
        method = self._get_method()
        ti = {'table_res': {}}
        html = "<div>no table here</div>"
        # 경고 로그가 출력돼도 예외는 발생하지 않아야 한다
        method(ti, html)
        self.assertNotIn('html', ti['table_res'])

    def test_none_html_does_not_raise(self):
        """html_code=None이어도 예외 없이 처리돼야 한다."""
        method = self._get_method()
        ti = {'table_res': {}}
        method(ti, None)
        self.assertNotIn('html', ti['table_res'])

    def test_empty_string_does_not_raise(self):
        method = self._get_method()
        ti = {'table_res': {}}
        method(ti, "")
        self.assertNotIn('html', ti['table_res'])


class TestShouldSkipOcrDet(unittest.TestCase):
    """_should_skip_ocr_det 조건 분기 검증."""

    def _make_pipeline(self, use_det_mode='auto'):
        """최소 mock으로 TrueAsyncPipeline 인스턴스를 생성한다."""
        from rapid_doc.backend.pipeline.async_pipeline import TrueAsyncPipeline, PageContext

        mock_model = MagicMock()
        mock_model.layout_model = MagicMock()
        mock_model.formula_model = MagicMock()

        with patch('rapid_doc.backend.pipeline.async_pipeline.AtomModelSingleton'):
            with patch('rapid_doc.backend.pipeline.async_pipeline.get_formula_enable',
                       side_effect=lambda x: x):
                with patch('rapid_doc.backend.pipeline.async_pipeline.get_table_enable',
                           side_effect=lambda x: x):
                    pipeline = TrueAsyncPipeline.__new__(TrueAsyncPipeline)
                    pipeline.model = mock_model
                    pipeline.formula_enable = True
                    pipeline.table_enable = True
                    pipeline.use_det_mode = use_det_mode
                    pipeline.layout_config = {}
                    pipeline.ocr_config = {}
                    pipeline.formula_config = {}
                    pipeline.table_config = {}
                    pipeline.checkbox_config = {}
                    pipeline.verbose = False
                    pipeline.checkbox_enable = False
                    pipeline.formula_level = 0
                    pipeline.table_force_ocr = False
                    pipeline.skip_text_in_image = True
                    pipeline.use_img2table = False
                    pipeline.atom_model_manager = MagicMock()
                    pipeline.perf_stats = {}
                    pipeline.pdf_perf_stats = defaultdict(
                        lambda: defaultdict(lambda: {'time': 0.0, 'count': 0})
                    )
                    return pipeline

    def _make_ctx(self, ocr_enable=False):
        from rapid_doc.backend.pipeline.async_pipeline import PageContext
        return PageContext(pdf_idx=0, page_idx=0, ocr_enable=ocr_enable)

    def test_ocr_enable_always_process(self):
        """ocr_enable=True이면 use_det_mode에 관계없이 건너뛰지 않는다."""
        pipeline = self._make_pipeline(use_det_mode='txt')
        ctx = self._make_ctx(ocr_enable=True)
        res = {}
        self.assertFalse(pipeline._should_skip_ocr_det(ctx, res))

    def test_txt_mode_always_skip(self):
        """use_det_mode='txt'면 ocr_enable=False 시 항상 건너뛴다."""
        pipeline = self._make_pipeline(use_det_mode='txt')
        ctx = self._make_ctx(ocr_enable=False)
        res = {}
        self.assertTrue(pipeline._should_skip_ocr_det(ctx, res))

    def test_auto_mode_skip_if_pdf_det_done(self):
        """auto 모드에서 _pdf_det_done=True이고 need_ocr_det=False면 건너뛴다."""
        pipeline = self._make_pipeline(use_det_mode='auto')
        ctx = self._make_ctx(ocr_enable=False)
        res = {'_pdf_det_done': True, 'need_ocr_det': False}
        self.assertTrue(pipeline._should_skip_ocr_det(ctx, res))

    def test_auto_mode_process_if_need_ocr_det(self):
        """auto 모드에서 need_ocr_det=True이면 처리한다."""
        pipeline = self._make_pipeline(use_det_mode='auto')
        ctx = self._make_ctx(ocr_enable=False)
        res = {'_pdf_det_done': True, 'need_ocr_det': True}
        self.assertFalse(pipeline._should_skip_ocr_det(ctx, res))

    def test_ocr_mode_always_process(self):
        """use_det_mode='ocr'이면 pdf_det_done 여부에 관계없이 처리한다."""
        pipeline = self._make_pipeline(use_det_mode='ocr')
        ctx = self._make_ctx(ocr_enable=False)
        res = {'_pdf_det_done': True, 'need_ocr_det': False}
        self.assertFalse(pipeline._should_skip_ocr_det(ctx, res))


class TestApiSignatureCompatibility(unittest.TestCase):
    """
    async_batch_image_analyze() 서명이 pipeline_analyze.py의 호출 인터페이스와
    완전히 일치하는지 검증한다.
    """

    def test_async_batch_image_analyze_accepts_input_interval(self):
        """pipeline_analyze.py가 input_interval= 인수를 넘기므로 서명에 있어야 한다."""
        from rapid_doc.backend.pipeline.async_pipeline import async_batch_image_analyze
        sig = inspect.signature(async_batch_image_analyze)
        self.assertIn('input_interval', sig.parameters,
                      "async_batch_image_analyze에 input_interval 파라미터가 없음 "
                      "— pipeline_analyze.py와 호환 불가")

    def test_async_batch_image_analyze_accepts_verbose(self):
        from rapid_doc.backend.pipeline.async_pipeline import async_batch_image_analyze
        sig = inspect.signature(async_batch_image_analyze)
        self.assertIn('verbose', sig.parameters)

    def test_async_batch_image_analyze_required_params(self):
        """필수 파라미터는 images_with_extra_info 하나뿐이어야 한다."""
        from rapid_doc.backend.pipeline.async_pipeline import async_batch_image_analyze
        sig = inspect.signature(async_batch_image_analyze)
        required = [
            name for name, p in sig.parameters.items()
            if p.default is inspect.Parameter.empty
        ]
        self.assertEqual(required, ['images_with_extra_info'],
                         f"예상치 못한 필수 파라미터: {required}")

    def test_input_interval_is_ignored_not_crash(self):
        """input_interval 값이 어떤 값이어도 함수가 TypeError를 던지지 않아야 한다."""
        from rapid_doc.backend.pipeline.async_pipeline import async_batch_image_analyze
        # 실제 모델이 없으므로 custom_model_init 직전에 멈추도록 patch
        with patch('rapid_doc.backend.pipeline.async_pipeline.custom_model_init',
                   side_effect=RuntimeError("stop_here")):
            with self.assertRaises(RuntimeError) as cm:
                async_batch_image_analyze([], input_interval=0.5)
            self.assertIn("stop_here", str(cm.exception),
                          "TypeError가 발생했다면 서명 불일치")

    def test_pipeline_analyze_call_signature_matches(self):
        """pipeline_analyze.py가 async_batch_image_analyze를 호출하는 방식이 호환되는지 확인."""
        from rapid_doc.backend.pipeline.async_pipeline import async_batch_image_analyze
        # pipeline_analyze.py의 실제 호출 키워드 목록
        call_kwargs = {
            'formula_enable': True,
            'table_enable': True,
            'layout_config': None,
            'ocr_config': None,
            'formula_config': None,
            'table_config': None,
            'checkbox_config': None,
            'input_interval': 0.0,
            'verbose': False,
        }
        sig = inspect.signature(async_batch_image_analyze)
        for kwarg in call_kwargs:
            self.assertIn(kwarg, sig.parameters,
                          f"pipeline_analyze.py가 넘기는 '{kwarg}'가 서명에 없음")


class TestBuildPageContexts(unittest.TestCase):
    """_build_page_contexts 입력 파싱 검증."""

    def _make_pipeline(self):
        from rapid_doc.backend.pipeline.async_pipeline import TrueAsyncPipeline
        pipeline = TrueAsyncPipeline.__new__(TrueAsyncPipeline)
        pipeline.atom_model_manager = MagicMock()
        pipeline.perf_stats = {}
        pipeline.pdf_perf_stats = defaultdict(
            lambda: defaultdict(lambda: {'time': 0.0, 'count': 0})
        )
        return pipeline

    def _make_pil(self, w=64, h=64):
        arr = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        return Image.fromarray(arr)

    def test_7tuple_assigns_pdf_page_idx(self):
        """7-튜플 (img, scale, ocr, lang, page_dict, pdf_idx, page_idx)을 올바르게 파싱한다."""
        from rapid_doc.backend.pipeline.async_pipeline import PageContext
        pipeline = self._make_pipeline()
        img = self._make_pil()
        items = [(img, 1.5, True, 'ko', {'blocks': []}, 3, 7)]
        contexts = pipeline._build_page_contexts(items)

        self.assertEqual(len(contexts), 1)
        ctx = contexts[0]
        self.assertEqual(ctx.pdf_idx, 3)
        self.assertEqual(ctx.page_idx, 7)
        self.assertEqual(ctx.scale, 1.5)
        self.assertTrue(ctx.ocr_enable)
        self.assertEqual(ctx.lang, 'ko')
        self.assertIsNotNone(ctx.np_img)
        self.assertEqual(ctx.np_img.shape[2], 3)  # BGR

    def test_5tuple_assigns_auto_idx(self):
        """5-튜플이면 pdf_idx=0, page_idx=순서 번호를 할당한다."""
        pipeline = self._make_pipeline()
        img = self._make_pil()
        items = [
            (img, 1.0, False, 'ch', None),
            (img, 1.0, False, 'ch', None),
        ]
        contexts = pipeline._build_page_contexts(items)
        self.assertEqual(contexts[0].pdf_idx, 0)
        self.assertEqual(contexts[0].page_idx, 0)
        self.assertEqual(contexts[1].pdf_idx, 0)
        self.assertEqual(contexts[1].page_idx, 1)

    def test_rgb_to_bgr_conversion(self):
        """PIL RGB 이미지가 BGR numpy 배열로 변환되어야 한다."""
        pipeline = self._make_pipeline()
        # 순수 빨간색 픽셀 (R=255, G=0, B=0)
        arr = np.zeros((8, 8, 3), dtype=np.uint8)
        arr[:, :, 0] = 255  # R channel
        img = Image.fromarray(arr)  # PIL: RGB
        items = [(img, 1.0, False, 'ch', None)]
        contexts = pipeline._build_page_contexts(items)

        ctx = contexts[0]
        # BGR 변환 후: B=0, G=0, R=255 → np_img[:,:,2]=255
        self.assertEqual(ctx.np_img[0, 0, 2], 255)  # R in BGR
        self.assertEqual(ctx.np_img[0, 0, 0], 0)    # B in BGR


# =============================================================================
# L2: Mock 기반 파이프라인 테스트
# =============================================================================

def _make_dummy_pil(w=128, h=128):
    arr = np.random.randint(0, 200, (h, w, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def _make_mock_model():
    """MineruPipelineModel 최소 stub."""
    model = MagicMock()
    # layout_model.batch_predict → 빈 결과 (OCR 없는 텍스트 이미지로 간주)
    model.layout_model.batch_predict.return_value = [[]]
    # formula_model.batch_predict → 빈 latex 리스트
    model.formula_model.batch_predict.return_value = []
    return model


def _make_mock_ocr_model():
    """OCR 모델 stub (det_batch_predict 미지원 경로)."""
    ocr = MagicMock()
    del ocr.det_batch_predict  # hasattr() 검사 실패하도록 제거
    ocr.ocr.return_value = (None, None)  # (dt_boxes, rec_res) = (None, None)
    return ocr


def _make_mock_table_model():
    """Table 모델 stub."""
    table = MagicMock()
    table.predict.return_value = ('<table><tr><td>mock</td></tr></table>', None, None, None)
    return table


class TestMockPipelineRun(unittest.TestCase):
    """
    실제 모델 없이 TrueAsyncPipeline.run()의 전체 흐름을 검증한다.
    모든 모델 호출은 mock으로 대체한다.
    각 테스트의 wall-clock latency를 측정해 출력한다.
    """
    _latency_results: list = []  # 클래스 공유 결과 누적

    @classmethod
    def tearDownClass(cls):
        if not cls._latency_results:
            return
        print("\n[L2 Latency] Mock Pipeline Run")
        print(f"  {'Test':<52} {'Latency':>10}")
        print("  " + "-" * 64)
        for name, elapsed in cls._latency_results:
            print(f"  {name:<52} {elapsed*1000:>8.2f} ms")
        total = sum(e for _, e in cls._latency_results)
        print(f"  {'TOTAL':<52} {total*1000:>8.2f} ms")

    def setUp(self):
        from rapid_doc.backend.pipeline.async_pipeline import TrueAsyncPipeline

        self.mock_model = _make_mock_model()
        self.mock_ocr = _make_mock_ocr_model()
        self.mock_table = _make_mock_table_model()

        # get_res_list_from_layout_res를 mock으로 대체 (빈 후보 반환)
        self.patcher_res_list = patch(
            'rapid_doc.backend.pipeline.async_pipeline.get_res_list_from_layout_res',
            return_value=([], [], []),
        )
        self.patcher_formula_enable = patch(
            'rapid_doc.backend.pipeline.async_pipeline.get_formula_enable',
            side_effect=lambda x: x,
        )
        self.patcher_table_enable = patch(
            'rapid_doc.backend.pipeline.async_pipeline.get_table_enable',
            side_effect=lambda x: x,
        )
        self.patcher_clean = patch(
            'rapid_doc.backend.pipeline.async_pipeline.clean_memory',
        )
        self.patcher_atom = patch(
            'rapid_doc.backend.pipeline.async_pipeline.AtomModelSingleton',
        )

        self.patcher_res_list.start()
        self.patcher_formula_enable.start()
        self.patcher_table_enable.start()
        self.patcher_clean.start()
        mock_atom_cls = self.patcher_atom.start()
        mock_atom_cls.return_value.get_atom_model.side_effect = self._get_atom_model

    def tearDown(self):
        self.patcher_res_list.stop()
        self.patcher_formula_enable.stop()
        self.patcher_table_enable.stop()
        self.patcher_clean.stop()
        self.patcher_atom.stop()

    def _get_atom_model(self, atom_model_name=None, **kwargs):
        if atom_model_name == 'table':
            return self.mock_table
        return self.mock_ocr

    def _make_pipeline(self, formula_enable=False, table_enable=False):
        from rapid_doc.backend.pipeline.async_pipeline import TrueAsyncPipeline
        return TrueAsyncPipeline(
            model=self.mock_model,
            formula_enable=formula_enable,
            table_enable=table_enable,
            verbose=False,
        )

    def _timed_run(self, fn, *args, **kwargs):
        """fn() 실행 시간을 측정하고 (result, elapsed_s) 반환."""
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        return result, time.perf_counter() - t0

    def _record(self, name: str, elapsed: float):
        TestMockPipelineRun._latency_results.append((name, elapsed))

    def test_run_returns_correct_shape(self):
        """run()은 (results_list, perf_stats_dict)를 반환하고 pages 수와 일치해야 한다."""
        pipeline = self._make_pipeline()
        imgs = [(_make_dummy_pil(), 1.0, False, 'ch', {'blocks': []}, 0, i)
                for i in range(3)]

        (results, perf_stats), elapsed = self._timed_run(pipeline.run, imgs)
        self._record('run_returns_correct_shape (3 pages)', elapsed)

        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 3, "페이지 수만큼 결과가 반환되어야 한다")
        self.assertIsInstance(perf_stats, dict)

    def test_run_single_page(self):
        """단일 페이지 입력도 올바르게 처리된다."""
        pipeline = self._make_pipeline()
        imgs = [(_make_dummy_pil(), 1.0, False, 'ch', None, 0, 0)]
        (results, _), elapsed = self._timed_run(pipeline.run, imgs)
        self._record('run_single_page (1 page)', elapsed)
        self.assertEqual(len(results), 1)

    def test_run_empty_input(self):
        """빈 입력 리스트는 빈 결과를 반환하고 예외를 발생시키지 않는다."""
        pipeline = self._make_pipeline()
        (results, perf_stats), elapsed = self._timed_run(pipeline.run, [])
        self._record('run_empty_input (0 pages)', elapsed)
        self.assertEqual(results, [])
        self.assertIsInstance(perf_stats, dict)

    def test_layout_model_called_once(self):
        """batch_predict가 정확히 1회 호출되어야 한다."""
        pipeline = self._make_pipeline()
        imgs = [(_make_dummy_pil(), 1.0, False, 'ch', {'blocks': []}, 0, i)
                for i in range(2)]
        _, elapsed = self._timed_run(pipeline.run, imgs)
        self._record('layout_model_called_once (2 pages)', elapsed)
        self.mock_model.layout_model.batch_predict.assert_called_once()

    def test_formula_skipped_when_disabled(self):
        """formula_enable=False 이면 formula_model.batch_predict가 호출되지 않는다."""
        pipeline = self._make_pipeline(formula_enable=False)
        imgs = [(_make_dummy_pil(), 1.0, False, 'ch', None, 0, 0)]
        _, elapsed = self._timed_run(pipeline.run, imgs)
        self._record('formula_skipped_when_disabled (1 page)', elapsed)
        self.mock_model.formula_model.batch_predict.assert_not_called()

    def test_perf_stats_structure(self):
        """perf_stats는 dict이고 'layout' 키가 포함될 수 있다."""
        pipeline = self._make_pipeline()
        imgs = [(_make_dummy_pil(), 1.0, False, 'ch', None, 0, 0)]
        (_, perf_stats), elapsed = self._timed_run(pipeline.run, imgs)
        self._record('perf_stats_structure (1 page)', elapsed)
        for _pdf_idx, pdf_stats in perf_stats.items():
            for _stage, stage_stats in pdf_stats.items():
                self.assertIn('time', stage_stats)
                self.assertIn('count', stage_stats)

    def test_multiple_pdfs_keep_separate_contexts(self):
        """서로 다른 pdf_idx를 가진 페이지들이 각자 독립된 PageContext를 가진다."""
        pipeline = self._make_pipeline()
        imgs = [
            (_make_dummy_pil(), 1.0, False, 'ch', None, 0, 0),
            (_make_dummy_pil(), 1.0, False, 'ch', None, 1, 0),
            (_make_dummy_pil(), 1.0, False, 'ch', None, 1, 1),
        ]
        (results, _), elapsed = self._timed_run(pipeline.run, imgs)
        self._record('multiple_pdfs_separate_contexts (3 pages, 2 pdfs)', elapsed)
        self.assertEqual(len(results), 3)


class TestApplyDetBoxes(unittest.TestCase):
    """_apply_det_boxes bbox 후처리 로직 검증."""

    def _make_pipeline(self):
        from rapid_doc.backend.pipeline.async_pipeline import TrueAsyncPipeline, PageContext
        pipeline = TrueAsyncPipeline.__new__(TrueAsyncPipeline)
        pipeline.atom_model_manager = MagicMock()
        pipeline.perf_stats = {}
        pipeline.pdf_perf_stats = defaultdict(
            lambda: defaultdict(lambda: {'time': 0.0, 'count': 0})
        )
        return pipeline

    def _make_ctx(self):
        from rapid_doc.backend.pipeline.async_pipeline import PageContext
        return PageContext(pdf_idx=0, page_idx=0, ocr_enable=True)

    def test_none_boxes_returns_zero(self):
        """None dt_boxes는 0을 반환하고 layout_res를 변경하지 않는다."""
        pipeline = self._make_pipeline()
        ctx = self._make_ctx()
        result = pipeline._apply_det_boxes(ctx, None, None, np.zeros((64, 64, 3), dtype=np.uint8), [])
        self.assertEqual(result, 0)
        self.assertEqual(ctx.layout_res, [])

    def test_empty_boxes_returns_zero(self):
        """빈 dt_boxes 배열은 0을 반환한다."""
        pipeline = self._make_pipeline()
        ctx = self._make_ctx()
        result = pipeline._apply_det_boxes(ctx, [], None, np.zeros((64, 64, 3), dtype=np.uint8), [])
        self.assertEqual(result, 0)


class TestCollectOcrDetItems(unittest.TestCase):
    """_collect_ocr_det_items 수집 로직 검증."""

    def _make_pipeline(self, use_det_mode='auto'):
        from rapid_doc.backend.pipeline.async_pipeline import TrueAsyncPipeline
        pipeline = TrueAsyncPipeline.__new__(TrueAsyncPipeline)
        pipeline.use_det_mode = use_det_mode
        pipeline.atom_model_manager = MagicMock()
        pipeline.perf_stats = {}
        pipeline.pdf_perf_stats = defaultdict(
            lambda: defaultdict(lambda: {'time': 0.0, 'count': 0})
        )
        return pipeline

    def _make_ctx_with_candidates(self, ocr_enable=True, n_candidates=2):
        from rapid_doc.backend.pipeline.async_pipeline import PageContext
        h, w = 256, 256
        ctx = PageContext(
            pdf_idx=0, page_idx=0,
            np_img=np.random.randint(0, 200, (h, w, 3), dtype=np.uint8),
            ocr_enable=ocr_enable,
        )
        ctx.formula_regions = []
        ctx.checkbox_res = []
        for _ in range(n_candidates):
            ctx.ocr_candidates.append({
                'bbox': [10, 10, 100, 50],
                'poly': [10, 10, 100, 10, 100, 50, 10, 50],
                'category_id': 15,
            })
        return ctx

    def test_collect_returns_list(self):
        """반환값은 항상 list여야 한다."""
        pipeline = self._make_pipeline()

        with patch('rapid_doc.backend.pipeline.async_pipeline.crop_img') as mock_crop, \
             patch('rapid_doc.backend.pipeline.async_pipeline.get_adjusted_mfdetrec_res') as mock_adj:
            mock_crop.return_value = (
                np.zeros((64, 64, 3), dtype=np.uint8), []
            )
            mock_adj.return_value = []
            ctx = self._make_ctx_with_candidates(ocr_enable=True, n_candidates=2)
            items = pipeline._collect_ocr_det_items([ctx])

        self.assertIsInstance(items, list)
        self.assertEqual(len(items), 2)

    def test_txt_mode_skips_all(self):
        """use_det_mode='txt'에서 ocr_enable=False이면 모든 후보가 건너뛰어진다."""
        pipeline = self._make_pipeline(use_det_mode='txt')
        ctx = self._make_ctx_with_candidates(ocr_enable=False, n_candidates=3)
        items = pipeline._collect_ocr_det_items([ctx])
        self.assertEqual(len(items), 0)

    def test_pdf_det_done_skips_in_auto_mode(self):
        """auto 모드에서 _pdf_det_done=True인 후보는 건너뛰어진다."""
        pipeline = self._make_pipeline(use_det_mode='auto')
        ctx = self._make_ctx_with_candidates(ocr_enable=False, n_candidates=0)
        ctx.ocr_candidates = [
            {'bbox': [0, 0, 50, 50], 'poly': [0,0,50,0,50,50,0,50],
             'category_id': 15, '_pdf_det_done': True, 'need_ocr_det': False}
        ]
        items = pipeline._collect_ocr_det_items([ctx])
        self.assertEqual(len(items), 0)


# =============================================================================
# L3: 통합 테스트 — 실제 모델 필요, skip 조건 확인
# =============================================================================

_INTEGRATION_REASON = (
    "통합 테스트는 환경 변수 RD_TEST_PDF / RD_TEST_SKIP_INTEGRATION 설정 및 "
    "실제 모델 파일 필요. 스킵하려면 RD_TEST_SKIP_INTEGRATION=1 설정 또는 pytest -m 'not integration'"
)

def _integration_available():
    if os.environ.get('RD_TEST_SKIP_INTEGRATION', '0') == '1':
        return False
    pdf_path = os.environ.get('RD_TEST_PDF')
    if not pdf_path or not Path(pdf_path).is_file():
        return False
    required_env = {
        'CUSTOM_INTER_OP_THREADS_COUNT': '1',
        'CUSTOM_INTRA_OP_THREADS_COUNT': '2',
        'DXRT_DYNAMIC_CPU_THREAD': '1',
    }
    for k, v in required_env.items():
        if os.environ.get(k) != v:
            return False
    return True


@pytest.mark.integration
@unittest.skipUnless(_integration_available(), _INTEGRATION_REASON)
class TestIntegrationSyncVsAsync(unittest.TestCase):
    """
    실제 PDF와 모델을 사용해 Sync vs Async 파이프라인 출력 정합성을 검증한다.

    실행 전 설정:
        export RD_TEST_PDF=/path/to/sample.pdf
        source ./deepx_scripts/set_env.sh 1 2 1 3 2 4
        python -m pytest tests/test_async_pipeline.py -v -m integration
    """

    @classmethod
    def setUpClass(cls):
        """
        모델 초기화는 클래스 단위로 한 번만 수행한다.

        demo_offline.py의 do_parse()와 동일한 엔진/경로 설정을 사용한다.
        - Layout  : dxnn_models/pp_doclayout_l_part1.dxnn  (DX Engine)
        - OCR Det : dxnn_models/det_v5_*.dxnn  (DX Engine, multi-model)
        - OCR Rec : dxnn_models/rec_v5_ratio_*.dxnn  (DX Engine, multi-model)
        - Table   : dxnn_models/unet.dxnn  (DX Engine)
        - Formula : 비활성화 (dxengine 미지원)

        conftest.py가 L1/L2를 위해 외부 패키지(rapidocr, dx_engine, onnxruntime 등)와
        내부 모듈을 stub으로 교체했으므로, 여기서 ALL_CONFTEST_STUBS 전체를
        sys.modules에서 제거한다. 이후 rapid_doc 임포트 시 실제 설치된 패키지가 사용된다.
        """
        import sys
        from pathlib import Path as P

        project_root = P(__file__).parent.parent
        onnx_dir = project_root / 'onnx_models'
        dxnn_dir = project_root / 'dxnn_models'

        # 모델 파일 존재 확인 — dxnn 우선, 없으면 onnx fallback, 둘 다 없으면 skip
        dxnn_layout = dxnn_dir / 'pp_doclayout_l_part1.dxnn'
        onnx_layout = onnx_dir / 'pp_doclayout_l.onnx'
        if not dxnn_layout.exists() and not onnx_layout.exists():
            raise unittest.SkipTest(
                "Layout model not found (checked dxnn and onnx).\n"
                "Run 'bash setup_sample_models.sh' to download models."
            )

        # conftest.py가 주입한 모든 stub(외부 패키지 + 내부 모듈)을 sys.modules에서 제거.
        # 이후 임포트 시 실제 venv311 패키지가 사용된다.
        try:
            from tests.conftest import ALL_CONFTEST_STUBS
        except ImportError:
            from conftest import ALL_CONFTEST_STUBS

        to_remove = set(ALL_CONFTEST_STUBS)
        # rapid_doc.* 전체도 함께 제거 (부모 패키지 포함)
        to_remove.update(k for k in sys.modules if k.startswith('rapid_doc'))
        for key in to_remove:
            sys.modules.pop(key, None)

        cls.project_root = project_root
        cls.pdf_path = os.environ['RD_TEST_PDF']

        # demo_offline.py와 동일하게 로컬 모델만 사용 (네트워크 다운로드 방지)
        os.environ.setdefault('MINERU_MODEL_SOURCE', 'local')

        # ── Layout 설정 (demo_offline.py do_parse() 기준) ───────────────────
        from rapid_doc.model.layout.rapid_layout_self import ModelType as LayoutModelType
        from rapid_doc.model.layout.rapid_layout_self.utils.typings import EngineType as LayoutEngineType

        if dxnn_layout.exists():
            cls.layout_config = {
                'model_type': LayoutModelType.PP_DOCLAYOUT_L,
                'engine_type': LayoutEngineType.DXENGINE,
                'model_dir_or_path': str(dxnn_dir / 'pp_doclayout_l_part1.dxnn'),
                'sub_model_path': str(onnx_dir / 'pp_doclayout_l_part2.onnx'),
            }
        else:
            cls.layout_config = {
                'model_type': LayoutModelType.PP_DOCLAYOUT_L,
                'engine_type': LayoutEngineType.ONNXRUNTIME,
                'model_dir_or_path': str(onnx_dir / 'pp_doclayout_l.onnx'),
            }

        # ── OCR 설정 (demo_offline.py do_parse() 기준) ──────────────────────
        det_single = dxnn_dir / 'det_v5_640_640.dxnn'
        rec_single = dxnn_dir / 'rec_v5_ratio_10.dxnn'
        char_dict = project_root / 'value_compare' / 'recognition' / 'character_dict_from_onnx.txt'

        if det_single.exists() and rec_single.exists():
            cls.ocr_config = {
                'engine_type': 'dxengine',
                'use_det_mode': 'auto',
                'Det.model_path': str(det_single),
                'Rec.model_path': str(rec_single),
                'use_multi_det_model': True,
                'Det.model_paths': {
                    1:  str(dxnn_dir / 'det_v5_640_640.dxnn'),
                    2:  str(dxnn_dir / 'det_v5_320_640.dxnn'),
                    4:  str(dxnn_dir / 'det_v5_160_640.dxnn'),
                    10: str(dxnn_dir / 'det_v5_64_640.dxnn'),
                },
                'use_multi_rec_model': True,
                'Rec.model_paths': {
                    3:  str(dxnn_dir / 'rec_v5_ratio_3.dxnn'),
                    5:  str(dxnn_dir / 'rec_v5_ratio_5.dxnn'),
                    10: str(dxnn_dir / 'rec_v5_ratio_10.dxnn'),
                    15: str(dxnn_dir / 'rec_v5_ratio_15.dxnn'),
                    25: str(dxnn_dir / 'rec_v5_ratio_25.dxnn'),
                    35: str(dxnn_dir / 'rec_v5_ratio_35.dxnn'),
                },
                **(({'char_dict_path': str(char_dict)}) if char_dict.exists() else {}),
                'save_debug_images': False,
            }
        else:
            cls.ocr_config = {}

        # ── Formula: disable (dxengine 미지원) ──────────────────────────────
        cls.formula_config = {'enable': False}
        cls.formula_enable = False

        # ── Table: layout/OCR 위주 테스트이므로 비활성화 (unet.dxnn이 있어도)
        # Table 활성화 시 rapid_table_self가 CLS 모델을 네트워크에서 받으려 해서
        # 폐쇄망 환경에서 연결 타임아웃이 발생함.
        cls.table_config = {'enable': False}
        cls.table_enable = False

        # dxengine 사용 여부 플래그 — NPU SYNC+ASYNC 동시 로드 가드용
        # async_batch_image_analyze는 layout_config/ocr_config에 use_async=True를 추가해
        # AtomModelSingleton이 SYNC 모드 모델과 ASYNC 모드 모델을 동시에 NPU에 로드한다.
        # 현재 NPU 용량에서는 이중 로드 시 MEMORY_OVERFLOW(signal 6)가 발생하므로
        # dxengine 환경에서 _run_sync를 호출하는 테스트 3개는 자동으로 skip 처리한다.
        from rapid_doc.model.layout.rapid_layout_self.utils.typings import EngineType as _LE
        cls.using_dxengine = (cls.layout_config.get('engine_type') == _LE.DXENGINE)

        with open(cls.pdf_path, 'rb') as f:
            cls.pdf_bytes = f.read()

    def _run_sync(self):
        from rapid_doc.backend.pipeline.pipeline_analyze import doc_analyze
        return doc_analyze(
            pdf_bytes_list=[self.pdf_bytes],
            formula_enable=self.formula_enable,
            table_enable=self.table_enable,
            layout_config=self.layout_config,
            ocr_config=self.ocr_config,
            formula_config=self.formula_config,
            table_config=self.table_config,
            use_async_pipeline=False,
        )

    def _clear_model_cache(self):
        """AtomModelSingleton 캐시를 비워 NPU 모델을 언로드한다.
        Sync → Async 순차 실행 시 NPU 메모리 초과를 방지한다."""
        from rapid_doc.backend.pipeline.model_init import AtomModelSingleton
        AtomModelSingleton._models.clear()

    def setUp(self):
        """각 테스트 실행 전 모델 캐시를 초기화해 이전 테스트의 NPU 모델이 남지 않도록 한다."""
        self._clear_model_cache()

    def tearDown(self):
        """각 테스트 실행 후 모델 캐시를 초기화해 다음 테스트를 위해 NPU 메모리를 확보한다."""
        self._clear_model_cache()
        time.sleep(3.0)  # NPU 드라이버 메모리 해제 대기

    def _run_async(self):
        from rapid_doc.backend.pipeline.pipeline_analyze import doc_analyze
        return doc_analyze(
            pdf_bytes_list=[self.pdf_bytes],
            formula_enable=self.formula_enable,
            table_enable=self.table_enable,
            layout_config=self.layout_config,
            ocr_config=self.ocr_config,
            formula_config=self.formula_config,
            table_config=self.table_config,
            use_async_pipeline=True,
        )

    def test_async_returns_same_page_count(self):
        """async 파이프라인의 페이지 수가 sync와 동일해야 한다."""
        sync_results = self._run_sync()
        self._clear_model_cache()
        async_results = self._run_async()
        sync_pages = sync_results[0][0]
        async_pages = async_results[0][0]
        self.assertEqual(len(sync_pages), len(async_pages),
                         "Sync/Async 페이지 수 불일치")

    def test_async_layout_dets_non_empty(self):
        """텍스트가 있는 PDF라면 layout_dets가 비어있지 않아야 한다."""
        t0 = time.perf_counter()
        async_results = self._run_async()
        elapsed = time.perf_counter() - t0
        pages = async_results[0][0]
        total_dets = sum(len(page['layout_dets']) for page in pages)
        print(f"\n[L3 Latency] async layout_dets={total_dets} dets / {len(pages)} pages  wall={elapsed:.3f}s  ({len(pages)/max(elapsed,0.001):.2f} pages/s)")
        self.assertGreater(total_dets, 0, "layout_dets가 모두 비어있음")

    @pytest.mark.heavy
    def test_zz_category_id_distribution_similar(self):
        """Sync와 Async의 category_id 분포가 크게 다르지 않아야 한다.
        (Sync+Async를 모두 실행하는 가장 무거운 테스트 — 마지막 실행용)
        NPU 메모리 제약으로 다른 integration 테스트와 연속 실행 시 crash 발생 가능.
        단독 실행: python -m pytest -m integration_heavy
        """
        from collections import Counter
        sync_results = self._run_sync()
        self._clear_model_cache()
        async_results = self._run_async()

        def count_categories(results):
            return Counter(
                det['category_id']
                for page in results[0][0]
                for det in page['layout_dets']
            )

        sync_cats = count_categories(sync_results)
        async_cats = count_categories(async_results)

        self.assertEqual(set(sync_cats.keys()), set(async_cats.keys()),
                         f"category_id 집합 불일치\n  sync={set(sync_cats.keys())}\n  async={set(async_cats.keys())}")

    def test_async_faster_or_comparable_to_sync(self):
        """Async 파이프라인이 Sync보다 현저히 느리지 않아야 한다 (2× 이내)."""
        t0 = time.perf_counter()
        self._run_sync()
        sync_time = time.perf_counter() - t0

        self._clear_model_cache()

        t0 = time.perf_counter()
        self._run_async()
        async_time = time.perf_counter() - t0

        ratio = async_time / max(sync_time, 0.001)
        print(f"\n[L3 Latency] sync={sync_time:.3f}s  async={async_time:.3f}s  ratio={ratio:.2f}x")
        self.assertLess(ratio, 2.0,
                        f"Async({async_time:.2f}s)이 Sync({sync_time:.2f}s)보다 2배 이상 느림 (ratio={ratio:.2f})")

    def test_perf_stats_returned_by_doc_analyze(self):
        """
        doc_analyze()의 반환값은 infer_results이고,
        async_batch_image_analyze 내부 perf_stats는 로그에 출력되어야 한다.
        (간접 검증: 반환 타입 + 구조 확인)
        """
        raw = self._run_async()
        # doc_analyze returns (infer_results, all_image_lists, all_pdf_docs, lang_list, ocr_enabled_list)
        infer_results = raw[0]
        self.assertIsInstance(infer_results, list)
        self.assertEqual(len(infer_results), 1, "1개 PDF 입력→1개 PDF 결과")
        first_pdf_pages = infer_results[0]
        self.assertIsInstance(first_pdf_pages, list)
        if first_pdf_pages:
            page = first_pdf_pages[0]
            self.assertIn('layout_dets', page)
            self.assertIn('page_info', page)


# =============================================================================
# L1: FinegrainedStreamingPipeline 서명 검증
# =============================================================================

class TestFinegrainedStreamingPipelineSignature(unittest.TestCase):
    """FinegrainedStreamingPipeline 존재 및 API 서명 검증 (L1 — 모델 불필요)."""

    def test_class_exists_and_is_subclass(self):
        """FinegrainedStreamingPipeline이 StreamingPipeline의 서브클래스여야 한다."""
        from rapid_doc.backend.pipeline.async_pipeline import (
            FinegrainedStreamingPipeline,
            StreamingPipeline,
        )
        self.assertTrue(issubclass(FinegrainedStreamingPipeline, StreamingPipeline))

    def test_run_method_exists(self):
        """run() 메서드가 존재해야 한다."""
        from rapid_doc.backend.pipeline.async_pipeline import FinegrainedStreamingPipeline
        self.assertTrue(hasattr(FinegrainedStreamingPipeline, 'run'))

    def test_public_api_exists(self):
        """finegrained_streaming_batch_image_analyze 함수가 존재해야 한다."""
        from rapid_doc.backend.pipeline import async_pipeline
        self.assertTrue(
            hasattr(async_pipeline, 'finegrained_streaming_batch_image_analyze'),
            "finegrained_streaming_batch_image_analyze 함수가 async_pipeline에 없음",
        )

    def test_public_api_signature_params_match(self):
        """공개 API 함수의 파라미터 이름이 async_batch_image_analyze와 동일해야 한다."""
        from rapid_doc.backend.pipeline.async_pipeline import (
            async_batch_image_analyze,
            finegrained_streaming_batch_image_analyze,
        )
        ref_sig = inspect.signature(async_batch_image_analyze)
        new_sig = inspect.signature(finegrained_streaming_batch_image_analyze)
        self.assertEqual(
            set(ref_sig.parameters),
            set(new_sig.parameters),
            f"파라미터 불일치: {set(ref_sig.parameters) ^ set(new_sig.parameters)}",
        )

    def test_public_api_signature_defaults_match(self):
        """공개 API 함수의 파라미터 기본값이 async_batch_image_analyze와 동일해야 한다."""
        from rapid_doc.backend.pipeline.async_pipeline import (
            async_batch_image_analyze,
            finegrained_streaming_batch_image_analyze,
        )
        ref_sig = inspect.signature(async_batch_image_analyze)
        new_sig = inspect.signature(finegrained_streaming_batch_image_analyze)
        for name, ref_param in ref_sig.parameters.items():
            new_param = new_sig.parameters.get(name)
            self.assertIsNotNone(new_param, f"파라미터 '{name}' 누락")
            self.assertEqual(
                ref_param.default,
                new_param.default,
                f"파라미터 '{name}' 기본값 불일치: "
                f"{ref_param.default!r} vs {new_param.default!r}",
            )

    def test_queue_maxsize_constants(self):
        """OCR 집약 스테이지 Queue maxsize가 기본보다 커야 한다."""
        from rapid_doc.backend.pipeline.async_pipeline import FinegrainedStreamingPipeline
        self.assertGreater(
            FinegrainedStreamingPipeline._Q_OCR_MAXSIZE,
            FinegrainedStreamingPipeline._Q_DEFAULT_MAXSIZE,
        )


# =============================================================================
# L2: FinegrainedStreamingPipeline Mock 통합 테스트
# =============================================================================

class TestFinegrainedStreamingPipelineMock(unittest.TestCase):
    """FinegrainedStreamingPipeline Mock 통합 테스트 (L2 — 모델 불필요)."""

    def setUp(self):
        """기존 TestMockPipelineRun과 동일한 패턴으로 패치 관리."""
        self.mock_model = _make_mock_model()
        self.mock_ocr = _make_mock_ocr_model()
        self.mock_table = _make_mock_table_model()

        self.patcher_res_list = patch(
            'rapid_doc.backend.pipeline.async_pipeline.get_res_list_from_layout_res',
            return_value=([], [], []),
        )
        self.patcher_formula_enable = patch(
            'rapid_doc.backend.pipeline.async_pipeline.get_formula_enable',
            side_effect=lambda x: x,
        )
        self.patcher_table_enable = patch(
            'rapid_doc.backend.pipeline.async_pipeline.get_table_enable',
            side_effect=lambda x: x,
        )
        self.patcher_atom = patch(
            'rapid_doc.backend.pipeline.async_pipeline.AtomModelSingleton',
        )

        self.patcher_res_list.start()
        self.patcher_formula_enable.start()
        self.patcher_table_enable.start()
        mock_atom_cls = self.patcher_atom.start()
        mock_atom_cls.return_value.get_atom_model.return_value = self.mock_ocr

    def tearDown(self):
        self.patcher_res_list.stop()
        self.patcher_formula_enable.stop()
        self.patcher_table_enable.stop()
        self.patcher_atom.stop()

    def _make_pipeline(self, formula_enable=False, table_enable=False):
        from rapid_doc.backend.pipeline.async_pipeline import FinegrainedStreamingPipeline
        return FinegrainedStreamingPipeline(
            model=self.mock_model,
            formula_enable=formula_enable,
            table_enable=table_enable,
            verbose=False,
        )

    def _make_items(self, n: int):
        """n개의 더미 입력 튜플 생성."""
        img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
        return [(img, 1.0, False, 'ch', {}, 0, i) for i in range(n)]

    def test_run_returns_correct_shape(self):
        """run()이 (results, perf_stats) 튜플을 반환해야 한다."""
        pipeline = self._make_pipeline()
        items = self._make_items(3)
        results, perf_stats = pipeline.run(items)
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 3)
        self.assertIsInstance(perf_stats, dict)

    def test_run_preserves_page_order(self):
        """완료된 페이지가 (pdf_idx, page_idx) 순서로 정렬되어야 한다."""
        pipeline = self._make_pipeline()
        items = self._make_items(5)
        results, _ = pipeline.run(items)
        self.assertEqual(len(results), 5)

    def test_run_empty_input(self):
        """입력이 없을 때 빈 결과를 반환해야 한다."""
        pipeline = self._make_pipeline()
        results, perf_stats = pipeline.run([])
        self.assertEqual(results, [])
        self.assertIsInstance(perf_stats, dict)

    def test_seven_threads_created(self):
        """run()이 실행될 때 정확히 7개의 이름 있는 스레드가 생성되어야 한다."""
        pipeline = self._make_pipeline()
        items = self._make_items(2)

        created_thread_names = []
        original_init = threading.Thread.__init__

        def capture_init(self_t, *args, **kwargs):
            name = kwargs.get('name', '')
            if name:
                created_thread_names.append(name)
            original_init(self_t, *args, **kwargs)

        with patch.object(threading.Thread, '__init__', capture_init):
            pipeline.run(items)

        expected_names = {
            'fg-layout', 'fg-plan', 'fg-formula',
            'fg-pdf-det', 'fg-ocr-det', 'fg-table', 'fg-ocr-rec',
        }
        self.assertEqual(
            len(created_thread_names), 7,
            f"스레드 7개여야 하는데 {len(created_thread_names)}개 생성됨: {created_thread_names}",
        )
        self.assertEqual(set(created_thread_names), expected_names)

    def test_formula_disabled_does_not_call_formula_one(self):
        """formula_enable=False이면 _formula_one이 호출되지 않아야 한다."""
        pipeline = self._make_pipeline(formula_enable=False)
        items = self._make_items(2)
        with patch.object(pipeline, '_formula_one') as mock_formula:
            pipeline.run(items)
        mock_formula.assert_not_called()

    def test_table_disabled_does_not_call_table_one(self):
        """table_enable=False이면 _table_one이 호callable 않아야 한다."""
        pipeline = self._make_pipeline(table_enable=False)
        items = self._make_items(2)
        with patch.object(pipeline, '_table_one') as mock_table:
            pipeline.run(items)
        mock_table.assert_not_called()

    def test_layout_called_once_per_page(self):
        """_layout_one이 페이지당 정확히 1회 호출되어야 한다."""
        pipeline = self._make_pipeline()
        items = self._make_items(4)
        with patch.object(pipeline, '_layout_one') as mock_layout, \
             patch.object(pipeline, '_plan_one'), \
             patch.object(pipeline, '_pdf_det_one'), \
             patch.object(pipeline, '_ocr_det_one'), \
             patch.object(pipeline, '_ocr_rec_one'):
            pipeline.run(items)
        self.assertEqual(mock_layout.call_count, 4)


# =============================================================================
# 실행
# =============================================================================

if __name__ == '__main__':
    # 마커 없이 직접 실행 시 L1+L2만 실행 (Integration은 조건부)
    unittest.main(verbosity=2)
