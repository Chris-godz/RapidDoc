"""
CTC 디코딩 결과 비교 분석

직접 구현한 CTC 디코딩 vs RapidOCR의 CTCLabelDecode 비교
"""

import numpy as np
import onnxruntime as ort
from pathlib import Path
import cv2
from rapidocr.ch_ppocr_rec.utils import CTCLabelDecode


def load_model_and_character_dict(model_path: str):
    """모델 로드 및 문자 사전 추출"""
    session = ort.InferenceSession(model_path)
    metadata = session.get_modelmeta()
    
    if 'character' in metadata.custom_metadata_map:
        character_str = metadata.custom_metadata_map['character']
        character_dict = character_str.split('\n')
        return session, character_dict
    
    raise ValueError("모델에 문자 사전이 없습니다")


def preprocess_image(img_path: str, target_height: int = 48) -> np.ndarray:
    """이미지 전처리 (간단 버전)"""
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Resize
    h, w = img.shape
    ratio = target_height / h
    new_w = int(w * ratio)
    img_resized = cv2.resize(img, (new_w, target_height))
    
    # Normalize & Standardize
    img_normalized = img_resized.astype(np.float32) / 255.0
    img_standardized = (img_normalized - 0.5) / 0.5
    
    # Add channel dimension and transpose
    img_final = np.expand_dims(img_standardized, axis=0)  # (1, H, W)
    
    # Pad to required width
    max_w = 640
    if new_w < max_w:
        pad_w = max_w - new_w
        img_final = np.pad(img_final, ((0, 0), (0, 0), (0, pad_w)), mode='constant', constant_values=0)
    
    # Add batch dimension and convert to CHW format
    img_batch = np.expand_dims(img_final, axis=0)  # (1, 1, H, W)
    
    # Convert to RGB format (repeat grayscale to 3 channels)
    img_rgb = np.repeat(img_batch, 3, axis=1)  # (1, 3, H, W)
    
    return img_rgb


def decode_ctc_manual_v1(pred: np.ndarray, character_dict: list) -> tuple:
    """
    수동 구현 버전 1: 버그 있는 버전 (원래 코드)
    """
    preds_idx = np.argmax(pred, axis=1)
    preds_prob = np.max(pred, axis=1)
    
    char_list = []
    conf_list = []
    
    for idx, prob in zip(preds_idx, preds_prob):
        # CTC blank (마지막 인덱스) 무시
        if idx == len(character_dict):
            continue
        
        # 연속된 중복 문자 제거 (버그!)
        if char_list and idx == preds_idx[len(char_list) - 1]:
            continue
        
        if idx < len(character_dict):
            char_list.append(character_dict[idx])
            conf_list.append(prob)
    
    text = ''.join(char_list)
    score = float(np.mean(conf_list)) if conf_list else 0.0
    
    return text, score, preds_idx, preds_prob


def decode_ctc_manual_v2(pred: np.ndarray, character_dict: list) -> tuple:
    """
    수동 구현 버전 2: NumPy 벡터화 버전 (수정된 코드)
    """
    preds_idx = np.argmax(pred, axis=1)
    preds_prob = np.max(pred, axis=1)
    
    num_chars = len(character_dict)
    blank_idx = num_chars
    
    # 1. 중복 제거를 위한 selection mask
    selection = np.ones(len(preds_idx), dtype=bool)
    selection[1:] = preds_idx[1:] != preds_idx[:-1]
    
    # 2. Blank 제거
    selection &= preds_idx != blank_idx
    
    # 3. Out-of-range 인덱스 제거
    selection &= preds_idx < num_chars
    
    # 4. 선택된 인덱스만 문자로 변환
    selected_indices = preds_idx[selection]
    selected_probs = preds_prob[selection]
    
    char_list = [character_dict[idx] for idx in selected_indices]
    text = ''.join(char_list)
    score = float(np.mean(selected_probs)) if len(selected_probs) > 0 else 0.0
    
    return text, score, preds_idx, preds_prob


def decode_ctc_rapidocr(pred: np.ndarray, character_dict: list) -> tuple:
    """
    RapidOCR의 CTCLabelDecode 사용
    """
    # RapidOCR는 문자 사전에 'blank'와 ' '를 추가함
    ctc_decoder = CTCLabelDecode(character=character_dict)
    
    # 배치 차원 추가
    pred_batch = np.expand_dims(pred, axis=0)
    
    # 디코딩
    line_results, _ = ctc_decoder(pred_batch, return_word_box=False)
    
    if line_results:
        text, score = line_results[0]
        preds_idx = np.argmax(pred, axis=1)
        preds_prob = np.max(pred, axis=1)
        return text, score, preds_idx, preds_prob
    
    return "", 0.0, None, None


def analyze_predictions(preds_idx: np.ndarray, preds_prob: np.ndarray, 
                       character_dict: list, top_n: int = 20):
    """예측 결과 분석"""
    print(f"\n{'='*80}")
    print(f"예측 시퀀스 분석 (상위 {top_n}개)")
    print(f"{'='*80}")
    print(f"{'Step':<6} {'Index':<8} {'Prob':<10} {'Char':<10} {'Note'}")
    print(f"{'-'*80}")
    
    blank_idx = len(character_dict)
    prev_idx = -1
    
    for i in range(min(top_n, len(preds_idx))):
        idx = preds_idx[i]
        prob = preds_prob[i]
        
        if idx == blank_idx:
            char = "[BLANK]"
            note = "CTC blank"
        elif idx >= len(character_dict):
            char = "[OUT-OF-RANGE]"
            note = f"Invalid index (max={len(character_dict)-1})"
        else:
            char = repr(character_dict[idx])
            note = ""
        
        # 중복 체크
        if idx == prev_idx and idx != blank_idx:
            note += " [DUPLICATE]"
        
        print(f"{i:<6} {idx:<8} {prob:<10.6f} {char:<10} {note}")
        prev_idx = idx
    
    # 통계
    print(f"\n{'='*80}")
    print("통계")
    print(f"{'='*80}")
    print(f"총 타임스텝: {len(preds_idx)}")
    print(f"BLANK 개수: {np.sum(preds_idx == blank_idx)}")
    print(f"유효 문자 개수: {np.sum((preds_idx < len(character_dict)) & (preds_idx != blank_idx))}")
    print(f"Out-of-range 개수: {np.sum(preds_idx >= len(character_dict))}")
    print(f"평균 신뢰도: {np.mean(preds_prob):.6f}")


def main():
    print("="*80)
    print("CTC 디코딩 비교 분석")
    print("="*80)
    
    # 경로 설정
    model_path = "./onnx_models/ch_PP-OCRv5_rec_server_infer.onnx"
    img_path = "./demo/output-offline/demo1/rec_inputs/page000_region0015_cat15.jpg"
    
    # 1. 모델 및 문자 사전 로드
    print("\n[1] 모델 및 문자 사전 로드...")
    session, character_dict = load_model_and_character_dict(model_path)
    print(f"✓ 문자 사전 크기: {len(character_dict)}")
    print(f"  - 첫 5개: {[repr(c) for c in character_dict[:5]]}")
    print(f"  - 마지막 5개: {[repr(c) for c in character_dict[-5:]]}")
    
    # 2. 이미지 전처리
    print("\n[2] 이미지 전처리...")
    img = cv2.imread(img_path)
    print(f"✓ 원본 이미지 크기: {img.shape}")
    
    # 실제 전처리 (간단 버전)
    img_input = preprocess_image(img_path)
    print(f"✓ 전처리 완료: {img_input.shape}")
    
    # 3. 모델 추론
    print("\n[3] 모델 추론...")
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: img_input})
    pred = outputs[0][0]  # (T, num_classes)
    print(f"✓ 출력 형상: {pred.shape}")
    
    # 4. CTC 디코딩 비교
    print("\n" + "="*80)
    print("CTC 디코딩 결과 비교")
    print("="*80)
    
    # 버전 1: 버그 있는 버전
    print("\n[방법 1] 수동 구현 (버그 있음)")
    text_v1, score_v1, idx_v1, prob_v1 = decode_ctc_manual_v1(pred, character_dict)
    print(f"  텍스트: {repr(text_v1)}")
    print(f"  신뢰도: {score_v1:.6f}")
    print(f"  텍스트 길이: {len(text_v1)}")
    
    # 버전 2: NumPy 벡터화 버전
    print("\n[방법 2] 수동 구현 (NumPy 벡터화)")
    text_v2, score_v2, idx_v2, prob_v2 = decode_ctc_manual_v2(pred, character_dict)
    print(f"  텍스트: {repr(text_v2)}")
    print(f"  신뢰도: {score_v2:.6f}")
    print(f"  텍스트 길이: {len(text_v2)}")
    
    # 버전 3: RapidOCR
    print("\n[방법 3] RapidOCR CTCLabelDecode")
    text_v3, score_v3, idx_v3, prob_v3 = decode_ctc_rapidocr(pred, character_dict)
    print(f"  텍스트: {repr(text_v3)}")
    print(f"  신뢰도: {score_v3:.6f}")
    print(f"  텍스트 길이: {len(text_v3)}")
    
    # 5. 예측 시퀀스 분석
    analyze_predictions(idx_v1, prob_v1, character_dict, top_n=30)
    
    # 6. 차이점 분석
    print("\n" + "="*80)
    print("차이점 분석")
    print("="*80)
    
    print(f"\n[텍스트 비교]")
    print(f"  방법 1 vs 방법 3: {'동일' if text_v1 == text_v3 else '다름'}")
    print(f"  방법 2 vs 방법 3: {'동일' if text_v2 == text_v3 else '다름'}")
    
    if text_v1 != text_v3:
        print(f"\n  방법 1: {repr(text_v1[:50])}")
        print(f"  방법 3: {repr(text_v3[:50])}")
    
    print(f"\n[신뢰도 비교]")
    print(f"  방법 1: {score_v1:.6f}")
    print(f"  방법 2: {score_v2:.6f}")
    print(f"  방법 3: {score_v3:.6f}")
    
    # 7. RapidOCR의 특별한 처리 확인
    print("\n" + "="*80)
    print("RapidOCR CTCLabelDecode 특징")
    print("="*80)
    
    # RapidOCR는 문자 사전에 'blank'와 ' '를 추가
    ctc_decoder = CTCLabelDecode(character=character_dict)
    print(f"원본 문자 사전 크기: {len(character_dict)}")
    print(f"RapidOCR 문자 사전 크기: {len(ctc_decoder.character)}")
    print(f"추가된 문자: {[repr(c) for c in ctc_decoder.character if c not in character_dict]}")
    print(f"첫 3개 문자: {[repr(c) for c in ctc_decoder.character[:3]]}")
    print(f"마지막 3개 문자: {[repr(c) for c in ctc_decoder.character[-3:]]}")
    
    print("\n✅ 분석 완료!")


if __name__ == "__main__":
    main()
