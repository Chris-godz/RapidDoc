"""
CTC 디코더 출력 개수 디버깅 스크립트
"""

import sys
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from rapid_doc.model.ocr.dx_ocr import DxTextRecognizer
from rapid_doc.utils.config_loader import ConfigLoader
import cv2

def test_ctc_decoder():
    """CTC 디코더 출력 개수 확인"""
    
    print("=" * 80)
    print("CTC 디코더 출력 개수 디버깅")
    print("=" * 80)
    
    # 1. DxTextRecognizer 초기화
    config_loader = ConfigLoader()
    config = config_loader.get_model_config()
    
    rec_config = config.get("rec", {})
    model_path = rec_config.get("model_path", "dxnn_models/ch_PP-OCRv5_rec_server_infer.dxnn")
    char_dict_path = rec_config.get("char_dict_path", "rapid_doc/resources/ppocr_keys_v1.txt")
    
    recognizer = DxTextRecognizer(
        model_path=model_path,
        rec_batch_num=6,
        char_dict_path=char_dict_path
    )
    
    print(f"\nModel loaded: {model_path}")
    print(f"Batch size: {recognizer.rec_batch_num}")
    
    # 2. 테스트 이미지 생성 (6개)
    test_images = []
    for i in range(6):
        # 더미 이미지 (48x640x3)
        img = np.ones((48, 320, 3), dtype=np.uint8) * (50 + i * 30)
        test_images.append(img)
    
    print(f"\n입력 이미지 개수: {len(test_images)}")
    
    # 3. 전처리
    batch_input = recognizer._preprocess_batch(test_images)
    print(f"전처리 후 shape: {batch_input.shape}")
    
    # 4. 추론
    preds = recognizer.session.run(batch_input)[0]
    print(f"추론 결과 shape: {preds.shape}")
    print(f"  - Batch (B): {preds.shape[0]}")
    print(f"  - Time (T): {preds.shape[1]}")
    print(f"  - Classes (C): {preds.shape[2]}")
    
    # 5. CTC 디코딩
    line_results, word_results = recognizer.ctc_decoder(preds, return_word_box=False)
    
    print(f"\n🔍 CTC 디코더 출력:")
    print(f"  - line_results 개수: {len(line_results)}")
    print(f"  - word_results 개수: {len(word_results)}")
    
    print(f"\nline_results 내용:")
    for i, (text, score) in enumerate(line_results):
        print(f"  [{i}] text='{text}', score={score:.4f}")
    
    # 6. _postprocess 호출
    texts, scores = recognizer._postprocess(preds, len(test_images))
    print(f"\n_postprocess 출력:")
    print(f"  - texts 개수: {len(texts)}")
    print(f"  - scores 개수: {len(scores)}")
    
    # 7. 결과 비교
    print(f"\n{'='*80}")
    print(f"📊 결과 비교:")
    print(f"  - 입력 배치 크기: {len(test_images)}")
    print(f"  - CTC 디코더 출력: {len(line_results)}")
    print(f"  - _postprocess 출력: {len(texts)}")
    print(f"{'='*80}")
    
    if len(texts) != len(test_images):
        print(f"\n❌ 불일치! {len(test_images)}개 입력 → {len(texts)}개 출력")
        print(f"   누락: {len(test_images) - len(texts)}개")
    else:
        print(f"\n✅ 일치! {len(test_images)}개 입력 → {len(texts)}개 출력")

if __name__ == "__main__":
    test_ctc_decoder()
