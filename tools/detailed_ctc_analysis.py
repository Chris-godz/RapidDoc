"""
상세 CTC 디코딩 분석

RapidOCR가 왜 정확한지 단계별로 분석
"""

import numpy as np
import onnxruntime as ort
import cv2
from rapidocr.ch_ppocr_rec.utils import CTCLabelDecode


def main():
    print("="*100)
    print("RapidOCR CTCLabelDecode 상세 분석")
    print("="*100)
    
    # 1. 모델 및 문자 사전 로드
    model_path = "./onnx_models/ch_PP-OCRv5_rec_server_infer.onnx"
    session = ort.InferenceSession(model_path)
    metadata = session.get_modelmeta()
    character_str = metadata.custom_metadata_map['character']
    character_dict_raw = character_str.split('\n')
    
    print(f"\n[1단계] 모델 메타데이터의 문자 사전")
    print(f"  - 크기: {len(character_dict_raw)}")
    print(f"  - 첫 10개: {[repr(c) for c in character_dict_raw[:10]]}")
    
    # 2. RapidOCR CTCLabelDecode 초기화
    ctc_decoder = CTCLabelDecode(character=character_dict_raw)
    
    print(f"\n[2단계] RapidOCR가 변환한 문자 사전")
    print(f"  - 원본 크기: {len(character_dict_raw)}")
    print(f"  - 변환 후 크기: {len(ctc_decoder.character)}")
    print(f"  - 첫 10개: {[repr(c) for c in ctc_decoder.character[:10]]}")
    print(f"  - 마지막 5개: {[repr(c) for c in ctc_decoder.character[-5:]]}")
    
    # 3. 추가/변경된 부분 찾기
    print(f"\n[3단계] RapidOCR가 추가한 특수 토큰")
    
    # RapidOCR의 get_character 메서드가 하는 일:
    # 1. character_list.insert(0, 'blank')  - 첫 번째에 'blank' 추가
    # 2. character_list.insert(len(character_list), ' ')  - 마지막에 공백 추가
    
    if ctc_decoder.character[0] == 'blank':
        print(f"  ✓ 인덱스 0: 'blank' (CTC blank 토큰)")
    
    if ctc_decoder.character[-1] == ' ':
        print(f"  ✓ 인덱스 {len(ctc_decoder.character)-1}: ' ' (공백 문자)")
    
    # 4. ignored_tokens 확인
    ignored_tokens = ctc_decoder.get_ignored_tokens()
    print(f"\n[4단계] 무시되는 토큰")
    print(f"  - ignored_tokens: {ignored_tokens}")
    print(f"  - 의미: 인덱스 0 ('blank')는 디코딩 시 제거됨")
    
    # 5. 실제 이미지로 테스트
    img_path = "./demo/output-offline/demo1/rec_inputs/page000_region0015_cat15.jpg"
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    
    # 간단한 전처리
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ratio = 48 / h
    new_w = int(w * ratio)
    img_resized = cv2.resize(img_gray, (new_w, 48))
    img_normalized = img_resized.astype(np.float32) / 255.0
    img_standardized = (img_normalized - 0.5) / 0.5
    img_chw = np.expand_dims(img_standardized, axis=0)
    
    # 패딩
    max_w = 960
    if new_w < max_w:
        pad_w = max_w - new_w
        img_chw = np.pad(img_chw, ((0, 0), (0, 0), (0, pad_w)), mode='constant', constant_values=0)
    
    img_batch = np.expand_dims(img_chw, axis=0)
    img_rgb = np.repeat(img_batch, 3, axis=1)
    
    # 추론
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: img_rgb})
    pred = outputs[0]  # (1, T, num_classes)
    
    print(f"\n[5단계] 모델 출력 분석")
    print(f"  - 출력 형상: {pred.shape}")
    print(f"  - Batch size: {pred.shape[0]}")
    print(f"  - Timesteps: {pred.shape[1]}")
    print(f"  - Num classes: {pred.shape[2]}")
    
    # Argmax
    preds_idx = pred.argmax(axis=2)[0]
    preds_prob = pred.max(axis=2)[0]
    
    print(f"\n[6단계] Argmax 결과 샘플 (처음 20개 타임스텝)")
    print(f"{'Step':<6} {'Index':<8} {'Prob':<10} {'Token':<15} {'Note'}")
    print("-" * 80)
    
    for i in range(min(20, len(preds_idx))):
        idx = int(preds_idx[i])
        prob = float(preds_prob[i])
        
        if idx == 0:
            token = "'blank'"
            note = "← 이게 제거됨!"
        elif idx == len(ctc_decoder.character) - 1:
            token = "' '"
            note = "공백"
        elif idx < len(ctc_decoder.character):
            token = repr(ctc_decoder.character[idx])
            note = ""
        else:
            token = f"[{idx}]"
            note = "ERROR: Out of range!"
        
        print(f"{i:<6} {idx:<8} {prob:<10.6f} {token:<15} {note}")
    
    # 7. RapidOCR 디코딩
    line_results, _ = ctc_decoder(pred, return_word_box=False)
    text, score = line_results[0]
    
    print(f"\n[7단계] RapidOCR 디코딩 결과")
    print(f"  - 텍스트: {repr(text)}")
    print(f"  - 신뢰도: {score:.6f}")
    print(f"  - 실제 텍스트: {text}")
    
    # 8. 수동으로 따라해보기
    print(f"\n[8단계] RapidOCR 로직 수동 재현")
    
    # Selection mask (RapidOCR 방식)
    selection = np.ones(len(preds_idx), dtype=bool)
    
    # 중복 제거
    selection[1:] = preds_idx[1:] != preds_idx[:-1]
    print(f"  - 중복 제거 후: {np.sum(selection)} / {len(preds_idx)} 타임스텝 남음")
    
    # Blank 제거 (ignored_tokens)
    for ignored_token in ignored_tokens:
        selection &= preds_idx != ignored_token
    print(f"  - Blank 제거 후: {np.sum(selection)} / {len(preds_idx)} 타임스텝 남음")
    
    # 선택된 문자들
    selected_indices = preds_idx[selection]
    selected_probs = preds_prob[selection]
    
    manual_chars = [ctc_decoder.character[int(idx)] for idx in selected_indices]
    manual_text = ''.join(manual_chars)
    manual_score = float(np.mean(selected_probs))
    
    print(f"  - 수동 재현 텍스트: {repr(manual_text)}")
    print(f"  - 수동 재현 신뢰도: {manual_score:.6f}")
    print(f"  - RapidOCR와 동일? {manual_text == text}")
    
    # 9. 핵심 차이점
    print(f"\n{'='*100}")
    print("핵심 차이점 요약")
    print("="*100)
    
    print(f"\n❌ 잘못된 방법 (직접 구현):")
    print(f"  1. 모델 메타데이터의 문자 사전을 직접 사용")
    print(f"  2. BLANK를 마지막 인덱스({len(character_dict_raw)})로 가정")
    print(f"  3. 하지만 실제로는 인덱스 0이 가장 높은 확률을 가짐")
    print(f"  4. 인덱스 0을 BLANK가 아닌 '{character_dict_raw[0]}'로 해석")
    print(f"  5. 결과: 공백 문자(\u3000)가 잔뜩 섞인 이상한 텍스트")
    
    print(f"\n✅ 올바른 방법 (RapidOCR):")
    print(f"  1. 문자 사전 앞에 'blank' 추가 (인덱스 0)")
    print(f"  2. 문자 사전 뒤에 ' ' 추가 (공백 처리용)")
    print(f"  3. 인덱스 0 ('blank')을 CTC blank로 올바르게 처리")
    print(f"  4. 중복 제거 + blank 제거 로직이 정확함")
    print(f"  5. 결과: 완벽한 텍스트 '{text}'")
    
    print(f"\n💡 결론:")
    print(f"  모델 메타데이터의 문자 사전은 'blank'를 포함하지 않음!")
    print(f"  CTC 디코딩을 위해서는 'blank' 토큰을 명시적으로 추가해야 함!")
    print(f"  RapidOCR의 CTCLabelDecode.get_character() 메서드가 이를 자동으로 처리!")


if __name__ == "__main__":
    main()
