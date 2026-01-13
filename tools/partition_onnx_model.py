"""
ONNX 모델을 두 부분으로 분할하는 스크립트

DX Engine은 multi input을 지원하지 않으므로:
- Part 1: 메인 백본 (image만 입력) → 중간 feature 출력
- Part 2: 후처리 (feature + im_shape + scale_factor) → 최종 결과

사용법:
    python tools/partition_onnx_model.py \
        --input onnx_models/pp_doclayout_l.onnx \
        --output-part1 onnx_models/pp_doclayout_l_backbone.onnx \
        --output-part2 onnx_models/pp_doclayout_l_postprocess.onnx \
        --split-tensor p2o.pd_op.concat.12.0
"""
import argparse
from pathlib import Path
from typing import List, Set, Tuple

import onnx
from onnx import helper, numpy_helper
import numpy as np
from loguru import logger


def find_split_point_inputs(model: onnx.ModelProto, split_tensor: str) -> Set[str]:
    """
    분할 지점을 생성하는 노드의 모든 입력을 역추적합니다.
    """
    # split_tensor를 생성하는 노드 찾기
    split_node = None
    for node in model.graph.node:
        if split_tensor in node.output:
            split_node = node
            break
    
    if not split_node:
        logger.error(f"분할 텐서 '{split_tensor}'를 생성하는 노드를 찾을 수 없습니다.")
        return set()
    
    logger.info(f"분할 노드: {split_node.op_type} ({split_node.name})")
    logger.info(f"  입력: {list(split_node.input)}")
    logger.info(f"  출력: {list(split_node.output)}")
    
    # 이 노드와 그 이전의 모든 노드가 의존하는 텐서 추적
    part1_tensors = set()
    to_visit = list(split_node.input)
    
    while to_visit:
        tensor = to_visit.pop(0)
        if tensor in part1_tensors:
            continue
        part1_tensors.add(tensor)
        
        # 이 텐서를 생성하는 노드 찾기
        for node in model.graph.node:
            if tensor in node.output:
                to_visit.extend(node.input)
                break
    
    return part1_tensors


def partition_model(
    model: onnx.ModelProto,
    split_tensor: str
) -> Tuple[onnx.ModelProto, onnx.ModelProto]:
    """
    ONNX 모델을 두 부분으로 분할합니다.
    
    Returns:
        (part1_model, part2_model)
        - part1: 입력 → split_tensor (백본)
        - part2: split_tensor + 기타 입력 → 최종 출력 (후처리)
    """
    logger.info(f"모델 분할 지점: {split_tensor}")
    
    # 1. Part1에 속하는 텐서 찾기 (split_tensor 이전)
    part1_tensors = find_split_point_inputs(model, split_tensor)
    part1_tensors.add(split_tensor)  # split_tensor도 포함
    logger.info(f"Part1 텐서: {len(part1_tensors)}개")
    
    # 2. Part1 노드 수집
    part1_nodes = []
    for node in model.graph.node:
        # 출력이 part1_tensors에 있으면 part1에 포함
        if any(out in part1_tensors for out in node.output):
            part1_nodes.append(node)
    
    logger.info(f"Part1 노드: {len(part1_nodes)}개")
    
    # 3. Part2 노드 수집 (나머지)
    part2_nodes = []
    part2_needed_tensors = set()  # Part2가 필요로 하는 Part1 텐서
    
    for node in model.graph.node:
        if node not in part1_nodes:
            part2_nodes.append(node)
            # Part2 노드가 사용하는 입력 중 Part1 텐서 추적
            for inp in node.input:
                if inp in part1_tensors:
                    part2_needed_tensors.add(inp)
    
    logger.info(f"Part2 노드: {len(part2_nodes)}개")
    logger.info(f"Part2가 필요로 하는 Part1 텐서: {len(part2_needed_tensors)}개")
    
    # 4. Part1 입력 찾기
    part1_inputs = []
    part1_input_names = set()
    for inp in model.graph.input:
        # 모델 입력이고 part1 노드에서 사용되면 포함
        for node in part1_nodes:
            if inp.name in node.input:
                part1_inputs.append(inp)
                part1_input_names.add(inp.name)
                break
    
    logger.info(f"Part1 입력: {[i.name for i in part1_inputs]}")
    
    # 5. Part1 출력 설정 (split_tensor + Part2가 필요로 하는 Part1 텐서)
    part1_outputs = []
    part1_output_names = set()
    
    # split_tensor를 메인 출력으로 추가
    split_value_info = None
    for vi in model.graph.value_info:
        if vi.name == split_tensor:
            split_value_info = vi
            break
    
    if split_value_info:
        part1_outputs.append(split_value_info)
        part1_output_names.add(split_tensor)
    else:
        # 타입 정보가 없으면 추론
        logger.warning(f"'{split_tensor}'의 타입 정보 없음, 추론 시도")
        part1_outputs.append(helper.make_tensor_value_info(
            split_tensor,
            onnx.TensorProto.FLOAT,
            []
        ))
        part1_output_names.add(split_tensor)
    
    # Part2가 필요로 하는 Part1 텐서를 추가 출력으로 추가
    for tensor_name in part2_needed_tensors:
        if tensor_name in part1_output_names:
            continue
        
        # 타입 정보 찾기
        tensor_vi = None
        for vi in model.graph.value_info:
            if vi.name == tensor_name:
                tensor_vi = vi
                break
        
        if tensor_vi:
            part1_outputs.append(tensor_vi)
            part1_output_names.add(tensor_name)
        else:
            # 타입 정보가 없으면 FLOAT로 추론
            part1_outputs.append(helper.make_tensor_value_info(
                tensor_name,
                onnx.TensorProto.FLOAT,
                []
            ))
            part1_output_names.add(tensor_name)
    
    logger.info(f"Part1 출력: {list(part1_output_names)[:5]}... ({len(part1_outputs)}개)")
    
    # 6. Part1 Initializer 필터링
    part1_initializer_names = set()
    for node in part1_nodes:
        for inp in node.input:
            part1_initializer_names.add(inp)
    
    part1_initializers = [
        init for init in model.graph.initializer
        if init.name in part1_initializer_names
    ]
    logger.info(f"Part1 Initializer: {len(part1_initializers)}개")
    
    # 7. Part1 Value Info 필터링
    part1_value_info = [
        vi for vi in model.graph.value_info
        if vi.name in part1_tensors
    ]
    
    # 8. Part1 모델 생성
    part1_graph = helper.make_graph(
        part1_nodes,
        model.graph.name + "_part1",
        part1_inputs,
        part1_outputs,  # 리스트로 변경
        part1_initializers,
        value_info=part1_value_info,
    )
    
    part1_model = helper.make_model(
        part1_graph,
        producer_name=model.producer_name + "_part1",
        opset_imports=model.opset_import,
    )
    
    # 9. Part2 입력 찾기
    part2_inputs = []
    part2_input_names = set()
    
    # Part1의 모든 출력을 Part2 입력으로 추가
    for part1_out in part1_outputs:
        part2_inputs.append(part1_out)
        part2_input_names.add(part1_out.name)
    
    # 원본 모델 입력 중 part2에서 사용되는 것들 추가
    for inp in model.graph.input:
        if inp.name not in part1_input_names:
            for node in part2_nodes:
                if inp.name in node.input:
                    part2_inputs.append(inp)
                    part2_input_names.add(inp.name)
                    break
    
    logger.info(f"Part2 입력: {[i.name for i in part2_inputs]}")
    
    # 10. Part2 출력 (원본 모델 출력)
    part2_outputs = model.graph.output
    logger.info(f"Part2 출력: {[o.name for o in part2_outputs]}")
    
    # 11. Part2 Initializer 필터링
    # Part2 노드가 사용하는 모든 입력(텐서, initializer 포함)
    part2_tensor_names = set()
    for node in part2_nodes:
        for inp in node.input:
            part2_tensor_names.add(inp)
    
    # Part2에서 사용되는 initializer + part1에서 사용되지 않은 initializer
    part2_initializers = []
    for init in model.graph.initializer:
        if init.name in part2_tensor_names or init.name not in part1_initializer_names:
            part2_initializers.append(init)
    
    logger.info(f"Part2 Initializer: {len(part2_initializers)}개")
    
    # 12. Part2 Value Info 필터링
    # Part2가 사용하는 모든 텐서의 value info 포함
    part2_tensor_names = set()
    for node in part2_nodes:
        for inp in node.input:
            part2_tensor_names.add(inp)
        for out in node.output:
            part2_tensor_names.add(out)
    
    part2_value_info = [
        vi for vi in model.graph.value_info
        if vi.name in part2_tensor_names
    ]
    
    logger.info(f"Part2 Value Info: {len(part2_value_info)}개")
    
    # 13. Part2 모델 생성
    part2_graph = helper.make_graph(
        part2_nodes,
        model.graph.name + "_part2",
        part2_inputs,
        part2_outputs,
        part2_initializers,
        value_info=part2_value_info,
    )
    
    part2_model = helper.make_model(
        part2_graph,
        producer_name=model.producer_name + "_part2",
        opset_imports=model.opset_import,
    )
    
    return part1_model, part2_model


def print_model_info(model: onnx.ModelProto, title: str):
    """모델 정보 출력"""
    logger.info("=" * 80)
    logger.info(title)
    logger.info("=" * 80)
    logger.info(f"입력 ({len(model.graph.input)}개):")
    for inp in model.graph.input:
        shape = [d.dim_value if d.dim_value > 0 else d.dim_param 
                for d in inp.type.tensor_type.shape.dim]
        dtype = onnx.TensorProto.DataType.Name(inp.type.tensor_type.elem_type)
        logger.info(f"  - {inp.name}: {dtype} {shape}")
    
    logger.info(f"출력 ({len(model.graph.output)}개):")
    for out in model.graph.output:
        shape = [d.dim_value if d.dim_value > 0 else d.dim_param 
                for d in out.type.tensor_type.shape.dim]
        dtype = onnx.TensorProto.DataType.Name(out.type.tensor_type.elem_type)
        logger.info(f"  - {out.name}: {dtype} {shape}")
    
    logger.info(f"노드: {len(model.graph.node)}개")
    logger.info(f"Initializer: {len(model.graph.initializer)}개")
    logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="ONNX 모델을 두 부분으로 분할"
    )
    parser.add_argument("--input", "-i", required=True, help="입력 ONNX 모델")
    parser.add_argument("--output-part1", required=True, help="Part1 출력 (백본)")
    parser.add_argument("--output-part2", required=True, help="Part2 출력 (후처리)")
    parser.add_argument("--split-tensor", required=True, help="분할 지점 텐서 이름")
    parser.add_argument("--check", action="store_true", help="모델 검증")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    part1_path = Path(args.output_part1)
    part2_path = Path(args.output_part2)
    
    for p in [part1_path, part2_path]:
        p.parent.mkdir(parents=True, exist_ok=True)
    
    # 모델 로드
    logger.info(f"모델 로딩: {input_path}")
    model = onnx.load(str(input_path))
    print_model_info(model, "원본 모델")
    
    # 분할 텐서 존재 확인
    found = False
    for node in model.graph.node:
        if args.split_tensor in node.output:
            found = True
            break
    if not found:
        for vi in model.graph.value_info:
            if vi.name == args.split_tensor:
                found = True
                break
    
    if not found:
        logger.error(f"분할 텐서 '{args.split_tensor}'를 찾을 수 없습니다.")
        logger.info("사용 가능한 중간 텐서를 찾으려면 다음을 실행하세요:")
        logger.info(f"  python -c \"import onnx; m = onnx.load('{input_path}'); "
                   "[print(vi.name) for vi in m.graph.value_info[:20]]\"")
        return 1
    
    # 모델 분할
    logger.info("\n모델 분할 중...")
    part1, part2 = partition_model(model, args.split_tensor)
    
    print_model_info(part1, "Part1 모델 (백본)")
    print_model_info(part2, "Part2 모델 (후처리)")
    
    # 검증
    if args.check:
        logger.info("\n모델 검증 중...")
        try:
            onnx.checker.check_model(part1)
            logger.info("✅ Part1 검증 성공")
        except Exception as e:
            logger.warning(f"⚠️  Part1 검증 실패: {e}")
        
        try:
            onnx.checker.check_model(part2)
            logger.info("✅ Part2 검증 성공")
        except Exception as e:
            logger.warning(f"⚠️  Part2 검증 실패: {e}")
    
    # 저장
    logger.info(f"\nPart1 저장: {part1_path}")
    onnx.save(part1, str(part1_path))
    
    logger.info(f"Part2 저장: {part2_path}")
    onnx.save(part2, str(part2_path))
    
    # 파일 크기
    input_size = input_path.stat().st_size / (1024**2)
    part1_size = part1_path.stat().st_size / (1024**2)
    part2_size = part2_path.stat().st_size / (1024**2)
    
    logger.info(f"\n파일 크기:")
    logger.info(f"  원본: {input_size:.2f} MB")
    logger.info(f"  Part1: {part1_size:.2f} MB")
    logger.info(f"  Part2: {part2_size:.2f} MB")
    logger.info(f"  합계: {part1_size + part2_size:.2f} MB")
    
    logger.info("\n✅ 완료!")
    logger.info(f"\n사용법:")
    logger.info(f"1. Part1을 DX Engine으로 변환 (single input):")
    logger.info(f"   dx_compiler --input {part1_path} --output part1.dxnn")
    logger.info(f"2. Part2는 ONNX Runtime으로 실행 (multi input 지원):")
    logger.info(f"   import onnxruntime; sess = onnxruntime.InferenceSession('{part2_path}')")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
