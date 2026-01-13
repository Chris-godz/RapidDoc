"""
OCR Detection FPS 측정 스크립트

DX Engine 기반 OCR-det 모델의 추론 속도(FPS)를 측정합니다.
Layout 모델의 실제 출력을 사용하여 현실적인 워크플로우를 시뮬레이션합니다.

사용법:
    python tests/ocr_det_fps_test.py
"""

import os
import sys
import time
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from loguru import logger

# 프로젝트 루트를 PYTHONPATH에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from rapid_doc.model.ocr.dx_ocr import DxTextDetector
from rapid_doc.model.layout.rapid_layout_self import RapidLayout
from rapid_doc.model.layout.rapid_layout_self.utils.typings import (
    RapidLayoutInput,
    ModelType as LayoutModelType,
    EngineType as LayoutEngineType
)
from rapid_doc.utils.pdf_image_tools import load_images_from_pdf, ImageType


def load_test_pdf_images(pdf_dir: Path, max_pages: int = 10) -> list:
    """
    실제 PDF 이미지 로딩 (demo_offline.py 참고)
    
    Args:
        pdf_dir: PDF 파일이 있는 디렉토리
        max_pages: 최대 페이지 수
        
    Returns:
        numpy array 리스트 (BGR 포맷)
    """
    pdf_files = list(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        logger.warning(f"PDF 파일을 찾을 수 없습니다: {pdf_dir}")
        return []
    
    # 첫 번째 PDF만 사용
    pdf_path = pdf_files[0]
    logger.info(f"PDF 로딩: {pdf_path.name}")
    
    with open(pdf_path, 'rb') as f:
        pdf_bytes = f.read()
    
    # PDF를 이미지로 변환 (demo_offline.py의 load_images_from_pdf 사용)
    images_list, pdf_doc_list = load_images_from_pdf(pdf_bytes, image_type=ImageType.PIL)
    
    # PDF 문서 닫기
    for pdf_doc in pdf_doc_list:
        pdf_doc.close()
    
    # max_pages로 제한
    images_list = images_list[:max_pages]
    
    # PIL Image를 numpy array (BGR)로 변환
    images = []
    for img_dict in images_list:
        pil_img = img_dict['img_pil']
        # RGB -> BGR 변환
        img_array = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        images.append(img_array)
    
    logger.info(f"로딩 완료: {len(images)}장")
    return images


def crop_text_regions_from_layout(layout_model, images: list) -> list:
    """
    Layout 모델로 텍스트 영역 검출 후 크롭
    
    demo_offline.py의 워크플로우를 따라 실제 layout 출력을 사용합니다.
    
    Args:
        layout_model: Layout 모델 인스턴스
        images: 입력 이미지 리스트 (numpy array, BGR)
        
    Returns:
        크롭된 텍스트 영역 이미지 리스트
    """
    logger.info(f"Layout 모델로 텍스트 영역 검출 중: {len(images)}장")
    
    cropped_images = []
    
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


def generate_test_images(num_images: int = 100, size: tuple = (1024, 768)) -> list:
    """
    테스트용 더미 이미지 생성
    
    Args:
        num_images: 생성할 이미지 수
        size: 이미지 크기 (W, H)
        
    Returns:
        numpy array 리스트 (BGR 포맷)
    """
    logger.info(f"테스트 이미지 생성 중: {num_images}장, 크기 {size}")
    
    images = []
    for i in range(num_images):
        # 랜덤 BGR 이미지 생성 (OpenCV 포맷)
        img_array = np.random.randint(0, 255, (size[1], size[0], 3), dtype=np.uint8)
        images.append(img_array)
    
    return images


def measure_fps_single(detector: DxTextDetector, images: list, warmup: int = 10):
    """
    단일 이미지 추론 FPS 측정
    
    Args:
        detector: OCR 검출기
        images: 테스트 이미지 리스트
        warmup: 워밍업 반복 횟수
    """
    logger.info("=" * 80)
    logger.info("🔥 단일 이미지 추론 FPS 측정")
    logger.info("=" * 80)
    
    # 워밍업
    logger.info(f"워밍업 중... ({warmup}회)")
    for i in range(warmup):
        _ = detector(images[0])
    
    # 실제 측정
    num_images = len(images)
    logger.info(f"FPS 측정 시작: {num_images}장")
    
    start_time = time.perf_counter()
    
    for img in images:
        _ = detector(img)
    
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    
    # 결과 출력
    fps = num_images / elapsed_time
    s_per_it = elapsed_time / num_images
    
    logger.info("─" * 80)
    logger.info(f"📊 측정 결과:")
    logger.info(f"   총 처리 시간: {elapsed_time:.3f}초")
    logger.info(f"   총 이미지 수: {num_images}it")
    logger.info(f"   평균 속도: {s_per_it:.4f} s/it | {fps:.2f} it/s")
    logger.info("=" * 80)


def main():
    """메인 함수"""
    # 폐쇄망 환경 설정
    os.environ['MINERU_MODEL_SOURCE'] = 'local'
    
    logger.info("=" * 80)
    logger.info("🚀 DX Engine OCR-det FPS 벤치마크 (Layout 기반)")
    logger.info("=" * 80)
    
    # 모델 경로 설정
    model_dir_dxnn = Path(__file__).parent.parent / "dxnn_models"
    model_dir_onnx = Path(__file__).parent.parent / "onnx_models"
    det_model_path_dxnn = model_dir_dxnn / "ch_PP-OCRv5_server_det.dxnn"
    det_model_path_onnx = model_dir_onnx / "ch_PP-OCRv5_server_det.onnx"
    
    # .dxnn 파일 우선, 없으면 .onnx 사용
    if det_model_path_dxnn.exists():
        det_model_path = det_model_path_dxnn
        logger.info("✅ DX Engine 모델 발견 (.dxnn)")
    elif det_model_path_onnx.exists():
        det_model_path = det_model_path_onnx
        logger.warning("⚠️  ONNX 모델 사용 (.onnx) - DX Engine 성능을 위해 .dxnn 변환 권장")
    else:
        logger.error(f"❌ 모델 파일을 찾을 수 없습니다:")
        logger.error(f"   - {det_model_path_dxnn}")
        logger.error(f"   - {det_model_path_onnx}")
        logger.info("💡 먼저 ONNX 모델을 준비하거나 DX Engine 형식(.dxnn)으로 변환하세요.")
        return
    
    # OCR-det 모델 초기화
    logger.info(f"📦 OCR-det 모델 로딩: {det_model_path}")
    try:
        detector = DxTextDetector(
            model_path=str(det_model_path),
            box_thresh=0.3,
            unclip_ratio=1.5,
            use_dilation=False
        )
        logger.info("✅ OCR-det 모델 로딩 성공!")
    except Exception as e:
        logger.error(f"❌ OCR-det 모델 로딩 실패: {e}")
        return
    
    # Layout 모델 초기화 (실제 Layout 결과 사용)
    logger.info("")
    logger.info("📦 Layout 모델 로딩...")
    
    try:
        # demo_offline.py의 설정 참고
        project_root = Path(__file__).parent.parent
        onnx_models_dir = project_root / "onnx_models"
        layout_model_path = onnx_models_dir / "pp_doclayout_l.onnx"
        
        if not layout_model_path.exists():
            logger.error(f"❌ Layout 모델을 찾을 수 없습니다: {layout_model_path}")
            logger.info("💡 더미 이미지로 대체합니다.")
            layout_model = None
        else:
            layout_cfg = RapidLayoutInput(
                engine_type=LayoutEngineType.ONNXRUNTIME,
                model_type=LayoutModelType.PP_DOCLAYOUT_L,
                model_dir_or_path=str(layout_model_path),
            )
            layout_model = RapidLayout(cfg=layout_cfg)
            logger.info("✅ Layout 모델 로딩 성공!")
    except Exception as e:
        logger.error(f"❌ Layout 모델 로딩 실패: {e}")
        logger.info("💡 더미 이미지로 대체합니다.")
        layout_model = None
    
    # 테스트 방법 선택
    use_real_pdf = True if layout_model is not None else False
    pdf_dir = Path(__file__).parent.parent / "demo" / "pdfs"
    
    if use_real_pdf and layout_model is not None and pdf_dir.exists():
        logger.info("")
        logger.info("=" * 80)
        logger.info("📄 실제 PDF로 테스트")
        logger.info("=" * 80)
        
        # PDF 이미지 로딩
        pdf_images = load_test_pdf_images(pdf_dir, max_pages=10)
        
        if not pdf_images:
            logger.warning("PDF 로딩 실패, 더미 이미지로 대체")
            use_real_pdf = False
        else:
            # Layout으로 텍스트 영역 추출
            text_region_images = crop_text_regions_from_layout(layout_model, pdf_images)
            
            if len(text_region_images) == 0:
                logger.warning("텍스트 영역을 찾을 수 없음, 더미 이미지로 대체")
                use_real_pdf = False
            else:
                # OCR-det FPS 측정
                logger.info("")
                measure_fps_single(detector, text_region_images, warmup=10)
    
    if not use_real_pdf:
        # 더미 이미지로 테스트
        logger.info("")
        logger.info("=" * 80)
        logger.info("🎲 더미 이미지로 테스트")
        logger.info("=" * 80)
        
        num_images = 100
        image_sizes = [
            (800, 600),   # 작은 이미지
            (1024, 768),  # 중간 이미지
        ]
        
        for size in image_sizes:
            logger.info("")
            logger.info("=" * 80)
            logger.info(f"📐 이미지 크기: {size[0]}x{size[1]}")
            logger.info("=" * 80)
            
            images = generate_test_images(num_images, size)
            measure_fps_single(detector, images, warmup=10)
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("✅ 벤치마크 완료!")
    logger.info("=" * 80)
    logger.info("")
    logger.info("📝 참고 사항:")
    logger.info("   - Layout 모델로 추출한 실제 텍스트 영역으로 OCR-det을 테스트합니다.")
    logger.info("   - demo_offline.py의 워크플로우를 따라 category_id 기반으로 필터링합니다.")
    logger.info("   - OCR-det 모델의 추론 속도만 측정하며, 정확도는 검증하지 않습니다.")
    logger.info("   - 실제 성능은 PDF 내용, 텍스트 밀도에 따라 달라질 수 있습니다.")


if __name__ == "__main__":
    main()
