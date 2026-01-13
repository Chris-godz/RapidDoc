"""
ONNX 모델에서 특정 입력과 연관된 노드를 제거하는 스크립트 (간소화 버전)

DX Engine은 multi input을 지원하지 않으므로, 
im_shape, scale_factor 같은 보조 입력을 제거합니다.
제거될 출력은 의존하지 않는 가장 가까운 중간 텐서를 출력으로 설정합니다.

사용법:
    python tools/remove_onnx_inputs_simple.py \
        --input onnx_models/pp_doclayout_l.onnx \
        --output onnx_models/pp_doclayout_l_single_input.onnx \
        --remove-inputs im_shape scale_factor \
        --new-output p2o.pd_op.concat.12.0
"""
import argparse
from pathlib import Path
from typing import List, Set, Optional

import onnx
from onnx import helper
from loguru import logger


def get_dependent_tensors(model: onnx.ModelProto, input_names: List[str]) -> Set[str]:
    """
    특정 입력에 의존하는 모든 텐서(중간 출력)를 찾습니다.
    """
    dependent = set(input_names)
    
    changed = True
    while changed:
        changed = False
        for node in model.graph.node:
            if any(inp in dependent for inp in node.input):
                for out in node.output:
                    if out not in dependent:
                        dependent.add(out)
                        changed = True
    
    return dependent


def remove_inputs_simple(
    model: onnx.ModelProto, 
    input_names_to_remove: List[str],
    new_output_name: Optional[str] = None
) -> onnx.ModelProto:
    """
    ONNX 모델에서 특정 입력과 연관된 모든 노드를 제거합니다.
    """
    logger.info(f"제거할 입력: {input_names_to_remove}")
    
    # 1. 의존성 추적
    dependent = get_dependent_tensors(model, input_names_to_remove)
    logger.info(f"의존적인 텐서 {len(dependent)}개 발견")
    
    # 2. 노드 필터링
    new_nodes = []
    for node in model.graph.node:
        if any(inp in dependent for inp in node.input) or any(out in dependent for out in node.output):
            continue
        new_nodes.append(node)
    
    logger.info(f"노드: {len(model.graph.node)}개 → {len(new_nodes)}개")
    
    # 3. 입력 필터링
    new_inputs = [inp for inp in model.graph.input if inp.name not in input_names_to_remove]
    logger.info(f"입력: {len(model.graph.input)}개 → {len(new_inputs)}개")
    
    # 4. 출력 교체
    new_outputs = []
    for output in model.graph.output:
        if output.name in dependent:
            if new_output_name:
                logger.info(f"출력 교체: {output.name} → {new_output_name}")
                
                # 새 출력의 타입 정보 찾기
                value_info = None
                for vi in model.graph.value_info:
                    if vi.name == new_output_name:
                        value_info = vi
                        break
                
                # 타입 정보 생성 (없으면 FLOAT로 추정)
                if value_info:
                    shape = [d.dim_value if d.dim_value > 0 else d.dim_param 
                            for d in value_info.type.tensor_type.shape.dim]
                    dtype = value_info.type.tensor_type.elem_type
                else:
                    shape = []  # Unknown shape
                    dtype = onnx.TensorProto.FLOAT
                    logger.warning(f"'{new_output_name}'의 타입 정보 없음, FLOAT로 추정")
                
                new_outputs.append(helper.make_tensor_value_info(new_output_name, dtype, shape))
                new_output_name = None  # 한 번만 사용
            else:
                logger.warning(f"출력 '{output.name}' 제거됨 (대안 없음)")
        else:
            new_outputs.append(output)
    
    logger.info(f"출력: {len(model.graph.output)}개 → {len(new_outputs)}개")
    
    # 5. Initializer 및 Value Info 필터링
    new_initializers = [init for init in model.graph.initializer if init.name not in dependent]
    new_value_info = [vi for vi in model.graph.value_info if vi.name not in dependent]
    
    # 6. 그래프 생성
    new_graph = helper.make_graph(
        new_nodes,
        model.graph.name,
        new_inputs,
        new_outputs,
        new_initializers,
        value_info=new_value_info,
    )
    
    # 7. 모델 생성
    new_model = helper.make_model(
        new_graph,
        opset_imports=model.opset_import,
    )
    
    return new_model


def print_model_summary(model: onnx.ModelProto, title: str):
    """모델 요약 정보 출력"""
    logger.info("=" * 80)
    logger.info(title)
    logger.info("=" * 80)
    logger.info(f"입력:")
    for inp in model.graph.input:
        shape = [d.dim_value if d.dim_value > 0 else d.dim_param 
                for d in inp.type.tensor_type.shape.dim]
        logger.info(f"  - {inp.name}: {shape}")
    logger.info(f"출력:")
    for out in model.graph.output:
        shape = [d.dim_value if d.dim_value > 0 else d.dim_param 
                for d in out.type.tensor_type.shape.dim]
        logger.info(f"  - {out.name}: {shape}")
    logger.info(f"노드: {len(model.graph.node)}개")
    logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="ONNX 모델 입력 제거 (간소화 버전)")
    parser.add_argument("--input", "-i", required=True, help="입력 ONNX 파일")
    parser.add_argument("--output", "-o", required=True, help="출력 ONNX 파일")
    parser.add_argument("--remove-inputs", nargs="+", required=True, help="제거할 입력 이름")
    parser.add_argument("--new-output", help="새로운 출력 텐서 이름 (선택)")
    parser.add_argument("--check", action="store_true", help="모델 검증")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 모델 로드
    logger.info(f"모델 로딩: {input_path}")
    model = onnx.load(str(input_path))
    print_model_summary(model, "원본 모델")
    
    # 입력 제거
    logger.info("\n처리 중...")
    new_model = remove_inputs_simple(model, args.remove_inputs, args.new_output)
    print_model_summary(new_model, "수정된 모델")
    
    # 검증
    if args.check:
        try:
            onnx.checker.check_model(new_model)
            logger.info("✅ 모델 검증 성공")
        except Exception as e:
            logger.warning(f"⚠️  모델 검증 실패: {e}")
    
    # 저장
    logger.info(f"\n저장 중: {output_path}")
    onnx.save(new_model, str(output_path))
    
    input_size = input_path.stat().st_size / (1024**2)
    output_size = output_path.stat().st_size / (1024**2)
    logger.info(f"크기: {input_size:.2f} MB → {output_size:.2f} MB")
    logger.info("✅ 완료!")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
