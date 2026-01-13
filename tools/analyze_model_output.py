#!/usr/bin/env python3
"""
모델 출력 형식 분석 스크립트

ONNX 모델의 출력이 softmax가 적용되었는지 확인합니다.
"""

import sys
from pathlib import Path
import numpy as np
import cv2
from loguru import logger
import scipy.special

# 프로젝트 루트 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from value_compare.recognition.onnx_recognition import ONNXRecognitionInference


def analyze_model_output():
    """모델 출력 형식 분석"""
    
    # 모델 초기화
    model_path = "onnx_models/ch_PP-OCRv5_rec_server_infer.onnx"
    image_path = "demo/output-offline/demo1/rec_inputs/page000_region0015_cat15.jpg"
    
    recognizer = ONNXRecognitionInference(model_path=model_path)
    
    # 이미지 로드 및 전처리
    img = cv2.imread(image_path)
    batch_input = recognizer.preprocess_batch([img])
    
    # 추론
    preds = recognizer.inference(batch_input)
    logger.info(f"모델 출력 형상: {preds.shape}")  # (B, T, C)
    
    # 단일 예측 분석
    pred = preds[0]  # (T, C)
    logger.info(f"단일 예측 형상: {pred.shape}")
    
    # 🔍 출력 값 범위 확인 (softmax 여부 판단)
    logger.info("\n" + "=" * 80)
    logger.info("모델 출력 통계")
    logger.info("=" * 80)
    logger.info(f"최소값: {pred.min():.6f}")
    logger.info(f"최대값: {pred.max():.6f}")
    logger.info(f"평균값: {pred.mean():.6f}")
    logger.info(f"표준편차: {pred.std():.6f}")
    
    # 첫 번째 타임스텝의 출력 확인
    first_timestep = pred[0]  # (C,)
    logger.info(f"\n첫 번째 타임스텝 (처음 10개 값): {first_timestep[:10]}")
    logger.info(f"첫 번째 타임스텝 합: {first_timestep.sum():.6f}")
    
    # 🔍 Softmax 적용 여부 판단
    if np.allclose(first_timestep.sum(), 1.0, atol=0.01):
        logger.info("✅ Softmax 이미 적용됨 (합이 ~1.0)")
    elif pred.min() >= 0 and pred.max() <= 1:
        logger.info("⚠️  값은 [0, 1] 범위지만 합이 1이 아님 - Sigmoid 또는 다른 활성화 함수?")
    else:
        logger.warning("❌ Softmax 미적용 (raw logits) - Softmax 필요!")
    
    # 🔧 Softmax 적용 후 테스트
    logger.info("\n" + "=" * 80)
    logger.info("Softmax 적용 후 디코딩 테스트")
    logger.info("=" * 80)
    
    # Softmax 적용
    preds_softmax = scipy.special.softmax(pred, axis=1)
    logger.info(f"Softmax 후 형상: {preds_softmax.shape}")
    logger.info(f"Softmax 후 첫 번째 타임스텝 합: {preds_softmax[0].sum():.6f}")
    
    # Argmax로 디코딩
    preds_idx = np.argmax(preds_softmax, axis=1)
    preds_prob = np.max(preds_softmax, axis=1)
    
    logger.info(f"\n예측 인덱스 (처음 20개): {preds_idx[:20]}")
    logger.info(f"예측 확률 (처음 20개): {preds_prob[:20]}")
    
    # CTC 디코딩
    char_list = []
    conf_list = []
    prev_idx = -1
    
    num_classes = len(recognizer.character_dict)
    
    for i, (idx, prob) in enumerate(zip(preds_idx, preds_prob)):
        # BLANK 건너뛰기 (num_classes 이상의 인덱스는 모두 BLANK)
        if idx >= num_classes:
            prev_idx = idx
            continue
        
        # 연속 중복 제거
        if idx == prev_idx:
            continue
        
        char = recognizer.character_dict[idx]
        char_list.append(char)
        conf_list.append(prob)
        
        prev_idx = idx
        
        if i < 20:  # 처음 20개만 로그
            logger.debug(f"[{i:3d}] 인덱스 {idx:5d} → '{char}'")
    
    text = ''.join(char_list)
    score = float(np.mean(conf_list)) if conf_list else 0.0
    
    logger.info(f"\n✅ Softmax 적용 후 결과: \"{text}\"")
    logger.info(f"   문자 개수: {len(char_list)}")
    logger.info(f"   평균 신뢰도: {score:.4f}")
    
    # 🔍 원래 방식 (Softmax 없이)과 비교
    logger.info("\n" + "=" * 80)
    logger.info("Softmax 없이 디코딩 (원래 방식)")
    logger.info("=" * 80)
    
    preds_idx_raw = np.argmax(pred, axis=1)
    preds_prob_raw = np.max(pred, axis=1)
    
    logger.info(f"예측 인덱스 (처음 20개): {preds_idx_raw[:20]}")
    logger.info(f"예측 확률 (처음 20개): {preds_prob_raw[:20]}")
    
    char_list_raw = []
    conf_list_raw = []
    prev_idx = -1
    
    for idx, prob in zip(preds_idx_raw, preds_prob_raw):
        if idx >= num_classes:
            prev_idx = idx
            continue
        
        if idx == prev_idx:
            continue
        
        char = recognizer.character_dict[idx]
        char_list_raw.append(char)
        conf_list_raw.append(prob)
        
        prev_idx = idx
    
    text_raw = ''.join(char_list_raw)
    score_raw = float(np.mean(conf_list_raw)) if conf_list_raw else 0.0
    
    logger.info(f"\n❌ Softmax 없이 결과: \"{text_raw}\"")
    logger.info(f"   문자 개수: {len(char_list_raw)}")
    logger.info(f"   평균 신뢰도: {score_raw:.4f}")
    
    # 비교
    logger.info("\n" + "=" * 80)
    logger.info("결과 비교")
    logger.info("=" * 80)
    logger.info(f"Softmax 적용: \"{text}\" ({len(char_list)}자, 신뢰도: {score:.4f})")
    logger.info(f"Softmax 없음: \"{text_raw}\" ({len(char_list_raw)}자, 신뢰도: {score_raw:.4f})")


if __name__ == '__main__':
    analyze_model_output()
