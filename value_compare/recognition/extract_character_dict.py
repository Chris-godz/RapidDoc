"""
ONNX 모델 메타데이터에서 문자 사전 추출

이 스크립트는 ONNX 모델의 메타데이터에서 문자 사전을 추출하여 텍스트 파일로 저장합니다.
추출된 파일은 DXNN 모델 등에서 사용할 수 있습니다.

사용법:
    python value_compare/recognition/extract_character_dict.py \
        --model_path onnx_models/ch_PP-OCRv5_rec_server_infer.onnx \
        --output_path value_compare/recognition/character_dict.txt
"""

import argparse
from pathlib import Path
import onnxruntime as ort
from loguru import logger


def extract_character_dict(model_path: str, output_path: str) -> bool:
    """
    ONNX 모델에서 문자 사전 추출
    
    Args:
        model_path: ONNX 모델 경로
        output_path: 출력 파일 경로
        
    Returns:
        성공 여부
    """
    try:
        logger.info(f"ONNX 모델 로딩: {model_path}")
        
        # ONNX Runtime 세션 생성
        session = ort.InferenceSession(
            str(model_path),
            providers=['CPUExecutionProvider']
        )
        
        # 메타데이터 읽기
        metadata = session.get_modelmeta()
        
        if 'character' not in metadata.custom_metadata_map:
            logger.error("모델 메타데이터에 'character' 정보가 없습니다.")
            return False
        
        # 문자 사전 추출
        character_str = metadata.custom_metadata_map['character']
        char_list = character_str.split('\n')
        
        logger.info(f"✓ 문자 사전 추출 완료: {len(char_list)} 문자")
        
        # 파일로 저장
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(character_str)
        
        logger.info(f"✓ 문자 사전 저장 완료: {output_path}")
        logger.info(f"  - 총 문자 수: {len(char_list)}")
        logger.info(f"  - 첫 5개 문자: {char_list[:5]}")
        logger.info(f"  - 마지막 5개 문자: {char_list[-5:]}")
        
        return True
        
    except Exception as e:
        logger.error(f"문자 사전 추출 실패: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='ONNX 모델에서 문자 사전 추출')
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='ONNX 모델 경로'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        default='value_compare/recognition/character_dict_from_onnx.txt',
        help='출력 파일 경로 (기본값: value_compare/recognition/character_dict_from_onnx.txt)'
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("ONNX 모델 문자 사전 추출")
    logger.info("=" * 80)
    
    model_path = Path(args.model_path)
    
    if not model_path.exists():
        logger.error(f"모델을 찾을 수 없습니다: {model_path}")
        return
    
    success = extract_character_dict(str(model_path), args.output_path)
    
    if success:
        logger.info("\n" + "=" * 80)
        logger.info("✅ 완료")
        logger.info("=" * 80)
        logger.info(f"\n추출된 문자 사전을 DXNN 스크립트에서 사용하려면:")
        logger.info(f"python value_compare/recognition/dxnn_recognition.py \\")
        logger.info(f"    --model_path ./dxnn_models/ch_PP-OCRv5_rec_server_infer.dxnn \\")
        logger.info(f"    --dict_path {args.output_path} \\")
        logger.info(f"    --image_path ./demo/output-offline/demo1/rec_inputs/page000_region0015_cat15.jpg")
    else:
        logger.error("\n❌ 실패")


if __name__ == "__main__":
    main()
