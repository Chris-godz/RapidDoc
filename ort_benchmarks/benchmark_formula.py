#!/usr/bin/env python3
"""
Formula 모델의 순수 ONNX Runtime 추론 성능 측정 스크립트

전체 파이프라인 없이 Formula Recognition 모델의 순수 추론 속도만 측정합니다.
"""
import time
import numpy as np
from pathlib import Path
from loguru import logger


def benchmark_formula_ort():
    """Formula Recognition 모델의 ONNX Runtime 추론 성능 측정"""
    
    # =========================================================================
    # 모델 경로 설정
    # =========================================================================
    project_root = Path(__file__).parent.parent.absolute()
    formula_model_path = project_root / "onnx_models" / "pp_formulanet_plus_l.onnx"
    
    if not formula_model_path.exists():
        logger.error(f"Formula 모델을 찾을 수 없습니다: {formula_model_path}")
        return
    
    logger.info("=" * 80)
    logger.info("Formula Recognition 모델 순수 ONNX Runtime 추론 성능 측정")
    logger.info("=" * 80)
    logger.info(f"모델 경로: {formula_model_path}")
    
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
        str(formula_model_path),
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
    
    # =========================================================================
    # 테스트 이미지 생성 (Formula 모델의 실제 입력 shape에 맞춤)
    # Formula 모델: [batch_size, max_seq_len, hidden_dim] = [1, 768, 768]
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info(f"성능 측정 (입력 shape: {input_shape}, 배치: 1)")
    logger.info("=" * 80)
    
    # Formula 모델의 입력은 이미지가 아니라 인코더 출력 (sequence embedding)
    # 실제 shape에 맞춰 랜덤 텐서 생성
    if len(input_shape) == 3:
        # [batch_size, seq_len, hidden_dim]
        batch_size = 1
        seq_len = input_shape[1] if isinstance(input_shape[1], int) else 768
        hidden_dim = input_shape[2] if isinstance(input_shape[2], int) else 768
        preprocessed = np.random.randn(batch_size, seq_len, hidden_dim).astype(np.float32)
    elif len(input_shape) == 4:
        # [batch_size, channels, height, width]
        batch_size = 1
        channels = input_shape[1] if isinstance(input_shape[1], int) else 3
        height = input_shape[2] if isinstance(input_shape[2], int) else 64
        width = input_shape[3] if isinstance(input_shape[3], int) else 256
        preprocessed = np.random.randn(batch_size, channels, height, width).astype(np.float32)
    else:
        logger.error(f"지원하지 않는 입력 shape: {input_shape}")
        return
    
    logger.info(f"생성된 입력 shape: {preprocessed.shape}")
    
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
        benchmark_formula_ort()
    except KeyboardInterrupt:
        logger.info("\n벤치마크 중단됨")
    except Exception as e:
        logger.exception(f"벤치마크 실패: {e}")
