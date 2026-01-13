"""
모델 분할 전후 출력 검증 스크립트

분할된 Part1 + Part2의 연쇄 실행 결과와
원본 모델의 출력이 동일한지 확인합니다.

사용법:
    python tools/verify_partition.py \
        --original onnx_models/pp_doclayout_l.onnx \
        --part1 onnx_models/pp_doclayout_l_part1.onnx \
        --part2 onnx_models/pp_doclayout_l_part2.onnx
"""
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import onnx
import onnxruntime as ort
from loguru import logger


def generate_random_inputs(model: onnx.ModelProto, batch_size: int = 1) -> Dict[str, np.ndarray]:
    """
    모델 입력 형식에 맞는 랜덤 데이터 생성
    """
    inputs = {}
    
    for inp in model.graph.input:
        shape = []
        for dim in inp.type.tensor_type.shape.dim:
            if dim.dim_value > 0:
                shape.append(dim.dim_value)
            elif dim.dim_param:
                # Dynamic dimension
                if 'DynamicDimension' in dim.dim_param:
                    shape.append(batch_size)
                else:
                    shape.append(1)  # 기본값
            else:
                shape.append(1)
        
        dtype_map = {
            onnx.TensorProto.FLOAT: np.float32,
            onnx.TensorProto.INT32: np.int32,
            onnx.TensorProto.INT64: np.int64,
        }
        
        dtype = dtype_map.get(inp.type.tensor_type.elem_type, np.float32)
        
        # 특정 입력에 대한 합리적인 값 생성
        if inp.name == 'image':
            # 이미지: 0~1 범위의 값
            data = np.random.rand(*shape).astype(dtype)
        elif inp.name == 'im_shape':
            # 이미지 크기: [height, width]
            data = np.array([[640, 640]], dtype=dtype) if batch_size == 1 else np.tile([[640, 640]], (batch_size, 1)).astype(dtype)
        elif inp.name == 'scale_factor':
            # 스케일 팩터: [scale_h, scale_w]
            data = np.array([[1.0, 1.0]], dtype=dtype) if batch_size == 1 else np.tile([[1.0, 1.0]], (batch_size, 1)).astype(dtype)
        else:
            # 기타: 랜덤 값
            data = np.random.randn(*shape).astype(dtype)
        
        inputs[inp.name] = data
        logger.debug(f"입력 생성: {inp.name} {data.shape} {data.dtype}")
    
    return inputs


def run_original_model(
    model_path: str,
    inputs: Dict[str, np.ndarray]
) -> Dict[str, np.ndarray]:
    """
    원본 모델 실행
    """
    logger.info("원본 모델 실행 중...")
    
    session = ort.InferenceSession(model_path)
    
    # 입력 이름 확인
    input_names = [inp.name for inp in session.get_inputs()]
    logger.debug(f"모델 입력: {input_names}")
    
    # 실행
    outputs = session.run(None, inputs)
    
    # 출력을 딕셔너리로 변환
    output_dict = {}
    for i, out_meta in enumerate(session.get_outputs()):
        output_dict[out_meta.name] = outputs[i]
        logger.debug(f"출력: {out_meta.name} {outputs[i].shape} {outputs[i].dtype}")
    
    return output_dict


def run_partitioned_models(
    part1_path: str,
    part2_path: str,
    inputs: Dict[str, np.ndarray]
) -> Dict[str, np.ndarray]:
    """
    분할된 모델 연쇄 실행
    """
    logger.info("Part1 실행 중...")
    
    # Part1 실행
    part1_session = ort.InferenceSession(part1_path)
    part1_input_names = [inp.name for inp in part1_session.get_inputs()]
    logger.debug(f"Part1 입력: {part1_input_names}")
    
    # Part1 입력 준비
    part1_inputs = {name: inputs[name] for name in part1_input_names if name in inputs}
    
    part1_outputs = part1_session.run(None, part1_inputs)
    part1_output_names = [out.name for out in part1_session.get_outputs()]
    
    logger.debug(f"Part1 출력: {part1_output_names[0]} {part1_outputs[0].shape} {part1_outputs[0].dtype}")
    
    # Part2 실행
    logger.info("Part2 실행 중...")
    part2_session = ort.InferenceSession(part2_path)
    part2_input_names = [inp.name for inp in part2_session.get_inputs()]
    logger.debug(f"Part2 입력: {part2_input_names}")
    
    # Part2 입력 준비: Part1 출력 + 원본 입력
    part2_inputs = {}
    for i, name in enumerate(part1_output_names):
        part2_inputs[name] = part1_outputs[i]
    
    for name in part2_input_names:
        if name not in part2_inputs and name in inputs:
            part2_inputs[name] = inputs[name]
    
    part2_outputs = part2_session.run(None, part2_inputs)
    
    # 출력을 딕셔너리로 변환
    output_dict = {}
    for i, out_meta in enumerate(part2_session.get_outputs()):
        output_dict[out_meta.name] = part2_outputs[i]
        logger.debug(f"Part2 출력: {out_meta.name} {part2_outputs[i].shape} {part2_outputs[i].dtype}")
    
    return output_dict


def compare_outputs(
    original: Dict[str, np.ndarray],
    partitioned: Dict[str, np.ndarray],
    rtol: float = 1e-5,
    atol: float = 1e-5
) -> bool:
    """
    두 출력을 비교
    """
    logger.info("=" * 80)
    logger.info("출력 비교")
    logger.info("=" * 80)
    
    all_match = True
    
    for name in original.keys():
        if name not in partitioned:
            logger.error(f"❌ '{name}' 출력이 분할 모델에 없음")
            all_match = False
            continue
        
        orig = original[name]
        part = partitioned[name]
        
        # Shape 비교
        if orig.shape != part.shape:
            logger.error(f"❌ '{name}' Shape 불일치: {orig.shape} vs {part.shape}")
            all_match = False
            continue
        
        # Dtype 비교
        if orig.dtype != part.dtype:
            logger.warning(f"⚠️  '{name}' Dtype 불일치: {orig.dtype} vs {part.dtype}")
        
        # 값 비교
        if np.issubdtype(orig.dtype, np.floating):
            # Float: allclose 사용
            is_close = np.allclose(orig, part, rtol=rtol, atol=atol)
            
            if is_close:
                max_diff = np.max(np.abs(orig - part))
                mean_diff = np.mean(np.abs(orig - part))
                logger.info(f"✅ '{name}' 일치")
                logger.info(f"   Shape: {orig.shape}")
                logger.info(f"   최대 차이: {max_diff:.2e}")
                logger.info(f"   평균 차이: {mean_diff:.2e}")
            else:
                max_diff = np.max(np.abs(orig - part))
                mean_diff = np.mean(np.abs(orig - part))
                diff_ratio = np.sum(~np.isclose(orig, part, rtol=rtol, atol=atol)) / orig.size
                
                logger.error(f"❌ '{name}' 불일치")
                logger.error(f"   Shape: {orig.shape}")
                logger.error(f"   최대 차이: {max_diff:.2e}")
                logger.error(f"   평균 차이: {mean_diff:.2e}")
                logger.error(f"   불일치 비율: {diff_ratio*100:.2f}%")
                
                all_match = False
        else:
            # Integer: exact match
            is_equal = np.array_equal(orig, part)
            
            if is_equal:
                logger.info(f"✅ '{name}' 일치 (정수)")
                logger.info(f"   Shape: {orig.shape}")
            else:
                diff_count = np.sum(orig != part)
                diff_ratio = diff_count / orig.size
                
                logger.error(f"❌ '{name}' 불일치 (정수)")
                logger.error(f"   Shape: {orig.shape}")
                logger.error(f"   불일치 개수: {diff_count} / {orig.size}")
                logger.error(f"   불일치 비율: {diff_ratio*100:.2f}%")
                
                all_match = False
    
    logger.info("=" * 80)
    
    return all_match


def main():
    parser = argparse.ArgumentParser(
        description="모델 분할 전후 출력 검증"
    )
    parser.add_argument("--original", required=True, help="원본 ONNX 모델")
    parser.add_argument("--part1", required=True, help="Part1 ONNX 모델")
    parser.add_argument("--part2", required=True, help="Part2 ONNX 모델")
    parser.add_argument("--batch-size", type=int, default=1, help="배치 크기")
    parser.add_argument("--num-tests", type=int, default=3, help="테스트 반복 횟수")
    parser.add_argument("--rtol", type=float, default=1e-5, help="상대 허용 오차")
    parser.add_argument("--atol", type=float, default=1e-5, help="절대 허용 오차")
    parser.add_argument("--verbose", "-v", action="store_true", help="상세 로그")
    
    args = parser.parse_args()
    
    # 로그 레벨
    if not args.verbose:
        logger.remove()
        logger.add(lambda msg: print(msg, end=""), level="INFO")
    
    # 파일 존재 확인
    for path_arg, name in [
        (args.original, "원본 모델"),
        (args.part1, "Part1 모델"),
        (args.part2, "Part2 모델"),
    ]:
        if not Path(path_arg).exists():
            logger.error(f"{name} 파일 없음: {path_arg}")
            return 1
    
    logger.info("=" * 80)
    logger.info("모델 분할 검증")
    logger.info("=" * 80)
    logger.info(f"원본 모델: {args.original}")
    logger.info(f"Part1 모델: {args.part1}")
    logger.info(f"Part2 모델: {args.part2}")
    logger.info(f"배치 크기: {args.batch_size}")
    logger.info(f"테스트 횟수: {args.num_tests}")
    logger.info(f"허용 오차: rtol={args.rtol}, atol={args.atol}")
    logger.info("=" * 80)
    
    # 원본 모델 정보 로드
    original_model = onnx.load(args.original)
    
    # 테스트 실행
    all_tests_passed = True
    
    for test_idx in range(args.num_tests):
        logger.info(f"\n테스트 {test_idx + 1}/{args.num_tests}")
        logger.info("-" * 80)
        
        # 랜덤 입력 생성
        inputs = generate_random_inputs(original_model, args.batch_size)
        logger.info(f"입력 생성 완료: {list(inputs.keys())}")
        
        # 원본 모델 실행
        try:
            original_outputs = run_original_model(args.original, inputs)
        except Exception as e:
            logger.error(f"원본 모델 실행 실패: {e}")
            all_tests_passed = False
            continue
        
        # 분할 모델 실행
        try:
            partitioned_outputs = run_partitioned_models(args.part1, args.part2, inputs)
        except Exception as e:
            logger.error(f"분할 모델 실행 실패: {e}")
            all_tests_passed = False
            continue
        
        # 출력 비교
        test_passed = compare_outputs(
            original_outputs,
            partitioned_outputs,
            rtol=args.rtol,
            atol=args.atol
        )
        
        if test_passed:
            logger.info(f"✅ 테스트 {test_idx + 1} 통과")
        else:
            logger.error(f"❌ 테스트 {test_idx + 1} 실패")
            all_tests_passed = False
    
    # 최종 결과
    logger.info("\n" + "=" * 80)
    if all_tests_passed:
        logger.info("✅ 모든 테스트 통과! 분할이 정확합니다.")
        logger.info("=" * 80)
        return 0
    else:
        logger.error("❌ 일부 테스트 실패. 분할을 다시 확인하세요.")
        logger.error("=" * 80)
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
