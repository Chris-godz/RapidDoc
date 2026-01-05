#!/usr/bin/env python3
"""
OCR Detection 모델의 순수 ONNX Runtime 추론 성능 측정 스크립트

전체 파이프라인 없이 OCR Detection 모델의 순수 추론 속도만 측정합니다.
"""
import time
import numpy as np
from pathlib import Path
from loguru import logger


def benchmark_ocr_det_ort():
    """OCR Detection 모델의 ONNX Runtime 추론 성능 측정"""
    
    # =========================================================================
    # 모델 경로 설정
    # =========================================================================
    project_root = Path(__file__).parent.parent.absolute()
    ocr_det_model_path = project_root / "onnx_models" / "ch_PP-OCRv5_server_det.onnx"
    
    if not ocr_det_model_path.exists():
        logger.error(f"OCR Detection 모델을 찾을 수 없습니다: {ocr_det_model_path}")
        return
    
    logger.info("=" * 80)
    logger.info("OCR Detection 모델 순수 ONNX Runtime 추론 성능 측정")
    logger.info("=" * 80)
    logger.info(f"모델 경로: {ocr_det_model_path}")
    
    # =========================================================================
    # ONNX Runtime 세션 생성
    # =========================================================================
    import onnxruntime as ort
    
    # 세션 옵션 설정
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    # 프로바이더 설정 (CPU)
    providers = ['CPUExecutionProvider']
    
    logger.info("ONNX Runtime 세션 생성 중...")
    session_start = time.perf_counter()
    session = ort.InferenceSession(
        str(ocr_det_model_path),
        sess_options=sess_options,
        providers=providers
    )
    session_time = time.perf_counter() - session_start
    logger.info(f"✓ 세션 생성 완료: {session_time:.3f}초")
    
    # 입력/출력 정보 확인
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    input_dtype = session.get_inputs()[0].type
    logger.info(f"입력 이름: {input_name}")
    logger.info(f"입력 shape: {input_shape}")
    logger.info(f"입력 dtype: {input_dtype}")
    
    output_names = [output.name for output in session.get_outputs()]
    logger.info(f"출력 개수: {len(output_names)}")
    for i, name in enumerate(output_names):
        output_shape = session.get_outputs()[i].shape
        logger.info(f"  출력 {i}: {name} - shape: {output_shape}")
    
    # =========================================================================
    # 테스트 이미지 생성 (고정 크기: 640x640)
    # =========================================================================
    height, width = 640, 640
    
    logger.info("\n" + "=" * 80)
    logger.info(f"성능 측정 (이미지 크기: {height}x{width}, 배치: 1)")
    logger.info("=" * 80)
    
    # 랜덤 이미지 생성 (텍스트 영역 시뮬레이션)
    test_image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    
    # 전처리 (OCR Detection 모델 입력 형식에 맞게)
    if len(input_shape) == 4:
        if input_shape[1] == 3:  # NCHW 형식
            preprocessed = test_image.transpose(2, 0, 1)  # HWC -> CHW
            preprocessed = np.expand_dims(preprocessed, axis=0)  # CHW -> NCHW
        else:  # NHWC 형식
            preprocessed = np.expand_dims(test_image, axis=0)  # HWC -> NHWC
    else:
        preprocessed = test_image
    
    # 데이터 타입 변환 (float32)
    preprocessed = preprocessed.astype(np.float32)
    
    # 정규화 (0-255 -> 0-1)
    preprocessed = preprocessed / 255.0
    
    # 표준화 (mean, std normalization)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    if len(preprocessed.shape) == 4 and preprocessed.shape[1] == 3:  # NCHW
        mean = mean.reshape(1, 3, 1, 1)
        std = std.reshape(1, 3, 1, 1)
    
    preprocessed = (preprocessed - mean) / std
    
    # =========================================================================
    # Warm-up (첫 실행은 느릴 수 있음)
    # =========================================================================
    logger.info("Warm-up 중...")
    for _ in range(3):
        _ = session.run(output_names, {input_name: preprocessed})
    
    # =========================================================================
    # 벤치마크 (여러 번 실행하여 평균 측정)
    # =========================================================================
    num_iterations = 10
    logger.info(f"벤치마크 시작 ({num_iterations}회 반복)...")
    
    times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        _ = session.run(output_names, {input_name: preprocessed})
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    
    # 통계 계산
    times = np.array(times)
    mean_time = np.mean(times)
    
    logger.info(f"\n📊 성능 측정 결과:")
    logger.info(f"   {mean_time:.3f} s/it")
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ 벤치마크 완료")
    logger.info("=" * 80)


if __name__ == "__main__":
    try:
        benchmark_ocr_det_ort()
    except KeyboardInterrupt:
        logger.info("\n벤치마크 중단됨")
    except Exception as e:
        logger.exception(f"벤치마크 실패: {e}")
