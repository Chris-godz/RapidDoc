#!/usr/bin/env python3
"""
CTC 디코딩 디버깅 스크립트

onnx_recognition.py의 CTC 디코딩 문제를 분석합니다.
"""

import sys
from pathlib import Path
import numpy as np
import cv2
from loguru import logger

# 프로젝트 루트 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from value_compare.recognition.onnx_recognition import ONNXRecognitionInference


def debug_ctc_decoding():
    """CTC 디코딩 디버깅"""
    
    # 모델 초기화
    model_path = "onnx_models/ch_PP-OCRv5_rec_server_infer.onnx"
    image_path = "demo/output-offline/demo1/rec_inputs/page000_region0015_cat15.jpg"
    
    recognizer = ONNXRecognitionInference(model_path=model_path)
    
    # 이미지 로드
    img = cv2.imread(image_path)
    logger.info(f"이미지 크기: {img.shape}")
    
    # 전처리
    batch_input = recognizer.preprocess_batch([img])
    logger.info(f"전처리 완료: {batch_input.shape}")
    
    # 추론
    preds = recognizer.inference(batch_input)
    logger.info(f"추론 완료: {preds.shape}")  # (B, T, C)
    
    # 단일 예측 분석
    pred = preds[0]  # (T, num_classes) = (119, 18385)
    logger.info(f"\n단일 예측 형상: {pred.shape}")
    
    # Argmax
    preds_idx = np.argmax(pred, axis=1)
    preds_prob = np.max(pred, axis=1)
    
    logger.info(f"\n예측 인덱스 (처음 20개): {preds_idx[:20]}")
    logger.info(f"예측 확률 (처음 20개): {preds_prob[:20]}")
    
    # CTC blank 확인
    num_classes = len(recognizer.character_dict)
    blank_idx = num_classes  # 마지막 인덱스가 blank
    
    logger.info(f"\n문자 사전 크기: {num_classes}")
    logger.info(f"CTC blank 인덱스: {blank_idx}")
    logger.info(f"실제 출력 클래스 수: {pred.shape[1]}")
    
    # 🔍 핵심 확인: blank 인덱스가 실제로 존재하는지?
    if pred.shape[1] != num_classes + 1:
        logger.warning(f"⚠️  클래스 수 불일치!")
        logger.warning(f"   문자 사전: {num_classes}개")
        logger.warning(f"   모델 출력: {pred.shape[1]}개 (blank 포함)")
        logger.warning(f"   예상 blank 인덱스: {blank_idx}")
        logger.warning(f"   실제 모델 출력 클래스 수가 {pred.shape[1] - num_classes}개 차이남!")
    
    # 통계 분석
    unique_indices, counts = np.unique(preds_idx, return_counts=True)
    logger.info(f"\n고유 인덱스 개수: {len(unique_indices)}")
    logger.info(f"가장 많이 예측된 인덱스 (Top 10):")
    
    # Top 10 인덱스
    top_indices = np.argsort(-counts)[:10]
    for rank, idx in enumerate(top_indices, 1):
        pred_idx = unique_indices[idx]
        pred_count = counts[idx]
        
        if pred_idx == blank_idx:
            char = "[BLANK]"
        elif pred_idx < num_classes:
            char = recognizer.character_dict[pred_idx]
        else:
            char = f"[UNKNOWN:{pred_idx}]"
        
        logger.info(f"  {rank}. 인덱스 {pred_idx}: {pred_count}회 → '{char}'")
    
    # 🔍 문제 진단: 원래 디코딩 로직
    logger.info("\n" + "=" * 80)
    logger.info("원래 CTC 디코딩 (버그 있는 버전)")
    logger.info("=" * 80)
    
    char_list = []
    conf_list = []
    
    for i, (idx, prob) in enumerate(zip(preds_idx, preds_prob)):
        # CTC blank (마지막 인덱스) 무시
        if idx == len(recognizer.character_dict):
            logger.debug(f"[{i:3d}] 인덱스 {idx:5d} → BLANK 무시")
            continue
        
        # 🐛 버그: 여기서 preds_idx[len(char_list) - 1]는 이미 디코딩된 위치를 참조!
        # 실제로는 preds_idx의 이전 타임스텝과 비교해야 함
        if char_list and idx == preds_idx[len(char_list) - 1]:
            logger.debug(f"[{i:3d}] 인덱스 {idx:5d} → 중복 제거 (잘못된 비교!)")
            continue
        
        if idx < len(recognizer.character_dict):
            char = recognizer.character_dict[idx]
            char_list.append(char)
            conf_list.append(prob)
            logger.debug(f"[{i:3d}] 인덱스 {idx:5d} → '{char}' 추가")
    
    text = ''.join(char_list)
    score = float(np.mean(conf_list)) if conf_list else 0.0
    
    logger.info(f"\n원래 결과: \"{text}\" (score: {score:.4f})")
    logger.info(f"문자 개수: {len(char_list)}")
    
    # 🔧 올바른 CTC 디코딩
    logger.info("\n" + "=" * 80)
    logger.info("올바른 CTC 디코딩")
    logger.info("=" * 80)
    
    char_list_fixed = []
    conf_list_fixed = []
    prev_idx = -1  # 이전 타임스텝의 인덱스
    
    for i, (idx, prob) in enumerate(zip(preds_idx, preds_prob)):
        # CTC blank 무시
        if idx >= len(recognizer.character_dict):
            logger.debug(f"[{i:3d}] 인덱스 {idx:5d} → BLANK 무시")
            prev_idx = idx
            continue
        
        # 🔧 수정: 이전 타임스텝과 비교 (CTC collapse)
        if idx == prev_idx:
            logger.debug(f"[{i:3d}] 인덱스 {idx:5d} → 연속 중복 제거")
            continue
        
        if idx < len(recognizer.character_dict):
            char = recognizer.character_dict[idx]
            char_list_fixed.append(char)
            conf_list_fixed.append(prob)
            logger.debug(f"[{i:3d}] 인덱스 {idx:5d} → '{char}' 추가")
        
        prev_idx = idx
    
    text_fixed = ''.join(char_list_fixed)
    score_fixed = float(np.mean(conf_list_fixed)) if conf_list_fixed else 0.0
    
    logger.info(f"\n수정된 결과: \"{text_fixed}\" (score: {score_fixed:.4f})")
    logger.info(f"문자 개수: {len(char_list_fixed)}")
    
    # 비교
    logger.info("\n" + "=" * 80)
    logger.info("결과 비교")
    logger.info("=" * 80)
    logger.info(f"원래: \"{text}\" ({len(char_list)}자)")
    logger.info(f"수정: \"{text_fixed}\" ({len(char_list_fixed)}자)")
    logger.info(f"차이: {len(char_list_fixed) - len(char_list)}자 추가됨")


if __name__ == '__main__':
    debug_ctc_decoding()
