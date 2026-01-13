#!/usr/bin/env python3
"""
OCR-rec FPS 벤치마크 스크립트 (DX Engine)
Layout(ONNX) -> Det(ONNX) -> Rec(DX Engine) 워크플로우로 실제 텍스트 인식 성능 측정

워크플로우:
1. PDF 로딩
2. Layout 모델(ONNX)로 텍스트 영역 검출
3. OCR-det 모델(ONNX)로 텍스트 박스 검출
4. OCR-rec 모델(DX Engine)로 텍스트 인식 FPS 측정
"""
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from loguru import logger
from PIL import Image

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

from rapid_doc.model.ocr.dx_ocr import DxTextRecognizer
from rapid_doc.utils.pdf_image_tools import load_images_from_pdf, ImageType

# RapidOCR for Det (ONNX)
from rapidocr import RapidOCR as RapidOCR_ONNX
from rapidocr import EngineType as OCREngineType

# RapidLayout for Layout (ONNX)
from rapid_doc.model.layout.rapid_layout_self import RapidLayout
from rapid_doc.model.layout.rapid_layout_self.utils.typings import (
    RapidLayoutInput,
    ModelType as LayoutModelType,
    EngineType as LayoutEngineType
)


def load_test_pdf_images(pdf_name="demo3.pdf"):
    """
    테스트용 PDF 로딩
    
    Returns:
        List[np.ndarray]: BGR 포맷의 이미지 리스트
    """
    logger.info(f"PDF 로딩: {pdf_name}")
    
    # PDF 경로
    demo_dir = project_root / "demo" / "pdfs"
    pdf_path = demo_dir / pdf_name
    
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF 파일을 찾을 수 없습니다: {pdf_path}")
    
    # PDF 바이트 읽기
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()
    
    # PIL Image로 로딩
    images_list, pdf_doc_list = load_images_from_pdf(pdf_bytes, image_type=ImageType.PIL)
    
    # PDF 문서 닫기
    for pdf_doc in pdf_doc_list:
        pdf_doc.close()
    
    # PIL Image -> BGR numpy array 변환
    bgr_images = []
    for img_dict in images_list:
        pil_img = img_dict['img_pil']
        # PIL Image (RGB) -> numpy array (RGB)
        rgb_array = np.array(pil_img)
        # RGB -> BGR
        bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
        bgr_images.append(bgr_array)
    
    logger.info(f"로딩 완료: {len(bgr_images)}장")
    return bgr_images


def crop_text_regions_from_layout(layout_model, images):
    """
    Layout 모델로 텍스트 영역 추출
    
    Args:
        layout_model: Layout 모델 인스턴스
        images: 입력 이미지 리스트 (numpy array, BGR)
        
    Returns:
        크롭된 텍스트 영역 이미지 리스트
    """
    logger.info(f"Layout 모델로 텍스트 영역 검출 중: {len(images)}장")
    
    # BGR -> PIL Image 변환
    pil_images = []
    for img in images:
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img)
        pil_images.append(pil_img)
    
    # Layout 추론 (배치 처리)
    logger.info("  Layout 모델 추론 중...")
    layout_results = layout_model(pil_images, batch_size=1)
    
    logger.info(f"  Layout 검출 완료: {len(layout_results)}개 페이지")
    
    # 각 페이지에서 텍스트 영역 크롭
    cropped_images = []
    for page_idx, layout_result in enumerate(layout_results):
        logger.info(f"  페이지 {page_idx+1}: Layout 결과 처리 중...")
        
        # RapidLayoutOutput: boxes, class_names, scores
        if layout_result.boxes is None or layout_result.class_names is None:
            logger.warning(f"    페이지 {page_idx+1}: Layout 결과가 비어있습니다")
            continue
        
        boxes = layout_result.boxes
        class_names = layout_result.class_names
        scores = layout_result.scores or [1.0] * len(boxes)
        
        # 각 박스 처리
        for box_idx, (box, class_name, score) in enumerate(zip(boxes, class_names, scores)):
            # OCR이 필요한 카테고리만 필터링 (demo_offline.py 참조)
            # text, image, figure_title, table, header, footer
            if class_name not in ['text', 'image', 'figure_title', 'table', 'header', 'footer']:
                continue
            
            # 박스 좌표
            x1, y1, x2, y2 = box[:4]
            
            # 영역 크롭
            img = images[page_idx]  # 원본 BGR 이미지
            h, w = img.shape[:2]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # 범위 체크
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x2 > x1 and y2 > y1:
                cropped = img[y1:y2, x1:x2]
                cropped_images.append(cropped)
                logger.debug(f"    텍스트 영역 크롭: ({x1},{y1})-({x2},{y2}), class={class_name}, score={score:.3f}, size={cropped.shape}")
        
        logger.info(f"    → 누적 {len(cropped_images)}개 텍스트 영역")
    
    logger.info(f"총 {len(cropped_images)}개 텍스트 영역 크롭 완료")
    return cropped_images


def extract_text_boxes_from_det(ocr_det, images):
    """
    OCR-det 모델로 텍스트 박스 추출
    
    Args:
        ocr_det: RapidOCR 인스턴스 (Det만 사용)
        images: 입력 이미지 리스트 (numpy array, BGR)
        
    Returns:
        크롭된 텍스트 박스 이미지 리스트
    """
    logger.info(f"OCR-det 모델로 텍스트 박스 검출 중: {len(images)}장")
    
    text_box_images = []
    
    for img_idx, img in enumerate(images):
        # OCR-det 추론 (det_only=True)
        # RapidOCR의 내부 메서드 직접 호출
        det_output = ocr_det.text_det(img)
        dt_boxes = det_output.boxes  # boxes 속성 사용
        
        if dt_boxes is None or len(dt_boxes) == 0:
            logger.debug(f"  이미지 {img_idx+1}: 텍스트 박스 없음")
            continue
        
        # 각 텍스트 박스 크롭
        h, w = img.shape[:2]
        for box_idx, box in enumerate(dt_boxes):
            # box: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            box = box.astype(np.int32)
            x_min = max(0, np.min(box[:, 0]))
            y_min = max(0, np.min(box[:, 1]))
            x_max = min(w, np.max(box[:, 0]))
            y_max = min(h, np.max(box[:, 1]))
            
            if x_max > x_min and y_max > y_min:
                cropped = img[y_min:y_max, x_min:x_max]
                text_box_images.append(cropped)
        
        logger.info(f"  진행: {img_idx+1}/{len(images)} 이미지 처리, 누적 {len(text_box_images)}개 텍스트 박스")
    
    logger.info(f"총 {len(text_box_images)}개 텍스트 박스 추출 완료")
    return text_box_images


def measure_fps_single(rec_model, images, warmup_iters=10):
    """
    단일 이미지 추론 FPS 측정
    
    Args:
        rec_model: OCR-rec 모델 (DX Engine)
        images: 입력 이미지 리스트 (numpy array, BGR)
        warmup_iters: 워밍업 반복 횟수
        
    Returns:
        s/it (초/이미지), it/s (이미지/초)
    """
    logger.info("=" * 80)
    logger.info("🔥 OCR-rec 추론 FPS 측정")
    logger.info("=" * 80)
    
    # 워밍업
    logger.info(f"워밍업 중... ({warmup_iters}회)")
    for _ in range(min(warmup_iters, len(images))):
        _ = rec_model([images[0]])
    
    # FPS 측정
    logger.info(f"FPS 측정 시작: {len(images)}장")
    start_time = time.time()
    
    # 배치 크기 1로 순차 처리
    results = rec_model(images)
    
    total_time = time.time() - start_time
    total_images = len(images)
    
    s_per_it = total_time / total_images
    it_per_s = total_images / total_time
    
    logger.info("─" * 80)
    logger.info("📊 측정 결과:")
    logger.info(f"   총 처리 시간: {total_time:.3f}초")
    logger.info(f"   총 이미지 수: {total_images}it")
    logger.info(f"   평균 속도: {s_per_it:.4f} s/it | {it_per_s:.2f} it/s")
    logger.info("=" * 80)
    
    return s_per_it, it_per_s


def main():
    logger.info("=" * 80)
    logger.info("🚀 DX Engine OCR-rec FPS 벤치마크 (Layout+Det: ONNX, Rec: DX Engine)")
    logger.info("=" * 80)
    
    # =========================================================================
    # 모델 경로 설정
    # =========================================================================
    onnx_models_dir = project_root / "onnx_models"
    dxnn_models_dir = project_root / "dxnn_models"
    
    layout_model_path = onnx_models_dir / "pp_doclayout_l.onnx"
    det_model_path = onnx_models_dir / "ch_PP-OCRv5_server_det.onnx"
    rec_model_path_dxnn = dxnn_models_dir / "ch_PP-OCRv5_rec_server_infer.dxnn"
    
    # 모델 파일 존재 확인
    if not rec_model_path_dxnn.exists():
        logger.error(f"DX Engine 모델을 찾을 수 없습니다: {rec_model_path_dxnn}")
        return
    
    if not det_model_path.exists():
        logger.error(f"ONNX 모델을 찾을 수 없습니다: {det_model_path}")
        return
    
    if not layout_model_path.exists():
        logger.error(f"Layout 모델을 찾을 수 없습니다: {layout_model_path}")
        return
    
    logger.info("✅ 모델 파일 확인 완료")
    logger.info(f"   Layout(ONNX): {layout_model_path.name}")
    logger.info(f"   Det(ONNX):    {det_model_path.name}")
    logger.info(f"   Rec(DX):      {rec_model_path_dxnn.name}")
    
    # =========================================================================
    # 모델 로딩
    # =========================================================================
    # OCR-rec 모델 로딩 (DX Engine)
    logger.info("")
    logger.info("📦 OCR-rec 모델 로딩 (DX Engine)...")
    rec_model = DxTextRecognizer(model_path=str(rec_model_path_dxnn))
    logger.info("✅ OCR-rec 모델 로딩 성공!")
    
    # Layout 모델 로딩 (ONNX)
    logger.info("")
    logger.info("📦 Layout 모델 로딩 (ONNX)...")
    try:
        layout_cfg = RapidLayoutInput(
            engine_type=LayoutEngineType.ONNXRUNTIME,
            model_type=LayoutModelType.PP_DOCLAYOUT_L,
            model_dir_or_path=str(layout_model_path),
        )
        layout_model = RapidLayout(cfg=layout_cfg)
        logger.info("✅ Layout 모델 로딩 성공!")
    except Exception as e:
        logger.error(f"Layout 모델 로딩 실패: {e}")
        return
    
    # OCR-det 모델 로딩 (ONNX)
    logger.info("")
    logger.info("📦 OCR-det 모델 로딩 (ONNX)...")
    try:
        # RapidOCR params 방식 사용 (demo_offline.py와 rapid_ocr.py 참조)
        ocr_params = {
            "Det.engine_type": OCREngineType.ONNXRUNTIME,
            "Det.model_path": str(det_model_path),
            "Rec.engine_type": OCREngineType.ONNXRUNTIME,
            "Rec.model_path": str(onnx_models_dir / "ch_PP-OCRv5_rec_server_infer.onnx"),  # 일단 로딩은 해야 함
            "Global.use_cls": False,
        }
        ocr_det = RapidOCR_ONNX(params=ocr_params)
        logger.info("✅ OCR-det 모델 로딩 성공!")
    except Exception as e:
        logger.error(f"OCR-det 모델 로딩 실패: {e}")
        return
    
    # =========================================================================
    # 실제 PDF 테스트
    # =========================================================================
    logger.info("")
    logger.info("=" * 80)
    logger.info("📄 실제 PDF로 테스트")
    logger.info("=" * 80)
    
    # PDF 로딩
    pdf_images = load_test_pdf_images("demo3.pdf")
    
    # Step 1: Layout으로 텍스트 영역 크롭
    text_region_images = crop_text_regions_from_layout(layout_model, pdf_images)
    
    if len(text_region_images) == 0:
        logger.error("텍스트 영역을 찾을 수 없습니다!")
        return
    
    # Step 2: OCR-det으로 텍스트 박스 추출
    logger.info("")
    text_box_images = extract_text_boxes_from_det(ocr_det, text_region_images)
    
    if len(text_box_images) == 0:
        logger.error("텍스트 박스를 찾을 수 없습니다!")
        return
    
    # Step 3: OCR-rec FPS 측정 (DX Engine)
    logger.info("")
    s_per_it, it_per_s = measure_fps_single(rec_model, text_box_images)
    
    # =========================================================================
    # 완료
    # =========================================================================
    logger.info("")
    logger.info("=" * 80)
    logger.info("✅ 벤치마크 완료!")
    logger.info("=" * 80)
    logger.info("")
    logger.info("📝 참고 사항:")
    logger.info("   - Layout(ONNX) → OCR-det(ONNX) → OCR-rec(DX) 워크플로우로 측정합니다.")
    logger.info("   - 실제 텍스트 박스로 OCR-rec의 추론 속도만 측정합니다.")
    logger.info("   - 정확도는 검증하지 않으며, 성능은 텍스트 길이에 따라 달라질 수 있습니다.")


if __name__ == "__main__":
    main()
