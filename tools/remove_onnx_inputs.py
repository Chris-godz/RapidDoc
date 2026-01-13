"""
ONNX 모델에서 특정 입력과 연관된 노드를 제거하는 스크립트

DX Engine은 multi input을 지원하지 않으므로, 
im_shape, scale_factor 같은 보조 입력을 제거하고
해당 정보는 후처리에서 처리하도록 수정합니다.

사용법:
    python tools/remove_onnx_inputs.py \
        --input onnx_models/pp_doclayout_l.onnx \
        --output onnx_models/pp_doclayout_l_single_input.onnx \
        --remove-inputs im_shape scale_factor
"""
import argparse
from pathlib import Path
from typing import List, Set

import onnx
from onnx import helper, numpy_helper
import numpy as np
from loguru import logger


def get_node_dependencies(model: onnx.ModelProto, input_names: List[str]) -> Set[str]:
    """
    특정 입력에 의존하는 모든 노드와 중간 출력을 찾습니다.
    
    Args:
        model: ONNX 모델
        input_names: 제거할 입력 이름 리스트
    
    Returns:
        제거해야 할 노드의 출력 이름 집합
    """
    # 제거할 입력 이름을 시작점으로 설정
    dependent_outputs = set(input_names)
    
    # 그래프를 순회하면서 의존성 추적
    changed = True
    while changed:
        changed = False
        for node in model.graph.node:
            # 노드의 입력 중 하나라도 의존적이면 해당 노드의 출력도 의존적
            if any(inp in dependent_outputs for inp in node.input):
                for out in node.output:
                    if out not in dependent_outputs:
                        dependent_outputs.add(out)
                        changed = True
                        logger.debug(f"Node {node.op_type} (name: {node.name}) depends on removed inputs")
    
    return dependent_outputs


def find_alternative_outputs(
    model: onnx.ModelProto,
    dependent_outputs: Set[str]
) -> dict:
    """
    제거될 출력의 대안 출력을 찾습니다.
    
    Args:
        model: ONNX 모델
        dependent_outputs: 제거될 출력 집합
    
    Returns:
        {원래_출력: 대안_출력} 매핑
    """
    alternatives = {}
    
    for output in model.graph.output:
        if output.name in dependent_outputs:
            # 이 출력을 생성하는 노드 찾기
            for node in model.graph.node:
                if output.name in node.output:
                    # 이 노드의 입력 중 의존적이지 않은 것 찾기
                    for inp in node.input:
                        if inp not in dependent_outputs:
                            # 역으로 추적해서 의존적이지 않은 가장 가까운 출력 찾기
                            alternatives[output.name] = inp
                            logger.info(f"출력 대안 발견: {output.name} → {inp}")
                            break
                    break
    
    return alternatives


def remove_inputs_and_dependencies(
    model: onnx.ModelProto, 
    input_names_to_remove: List[str],
    find_outputs: bool = True
) -> onnx.ModelProto:
    """
    ONNX 모델에서 특정 입력과 연관된 모든 노드를 제거합니다.
    
    Args:
        model: 원본 ONNX 모델
        input_names_to_remove: 제거할 입력 이름 리스트
        find_outputs: 제거될 출력의 대안을 자동으로 찾을지 여부
    
    Returns:
        수정된 ONNX 모델
    """
    logger.info(f"제거할 입력: {input_names_to_remove}")
    
    # 1. 의존성 있는 노드와 출력 찾기
    dependent_outputs = get_node_dependencies(model, input_names_to_remove)
    logger.info(f"의존적인 출력 {len(dependent_outputs)}개 발견")
    
    # 1.5. 대안 출력 찾기
    output_alternatives = {}
    if find_outputs:
        output_alternatives = find_alternative_outputs(model, dependent_outputs)
    
    # 2. 새로운 그래프 생성
    new_nodes = []
    removed_nodes = []
    
    for node in model.graph.node:
        # 노드의 입력이나 출력이 의존적이면 제거
        node_depends = (
            any(inp in dependent_outputs for inp in node.input) or
            any(out in dependent_outputs for out in node.output)
        )
        
        if node_depends:
            removed_nodes.append(f"{node.op_type} ({node.name or 'unnamed'})")
            logger.debug(f"제거: {node.op_type} - inputs: {node.input}, outputs: {node.output}")
        else:
            new_nodes.append(node)
    
    logger.info(f"제거된 노드: {len(removed_nodes)}개")
    for node_desc in removed_nodes:
        logger.debug(f"  - {node_desc}")
    
    # 3. 입력 정의에서 제거
    new_inputs = [
        inp for inp in model.graph.input 
        if inp.name not in input_names_to_remove
    ]
    logger.info(f"입력: {len(model.graph.input)}개 → {len(new_inputs)}개")
    
    # 4. 출력 정의 수정
    new_outputs = []
    for output in model.graph.output:
        if output.name in dependent_outputs:
            if output.name in output_alternatives:
                # 대안 출력으로 교체
                alt_name = output_alternatives[output.name]
                logger.info(f"출력 교체: {output.name} → {alt_name}")
                
                # 대안 출력의 타입 정보 찾기
                alt_value_info = None
                for vi in model.graph.value_info:
                    if vi.name == alt_name:
                        alt_value_info = vi
                        break
                
                if alt_value_info:
                    new_outputs.append(helper.make_tensor_value_info(
                        alt_name,
                        alt_value_info.type.tensor_type.elem_type,
                        [dim.dim_value if dim.dim_value > 0 else dim.dim_param
                         for dim in alt_value_info.type.tensor_type.shape.dim]
                    ))
                else:
                    # 타입 정보를 찾을 수 없으면 원본 타입 사용
                    new_outputs.append(helper.make_tensor_value_info(
                        alt_name,
                        output.type.tensor_type.elem_type,
                        [dim.dim_value if dim.dim_value > 0 else dim.dim_param
                         for dim in output.type.tensor_type.shape.dim]
                    ))
            else:
                logger.warning(f"⚠️  모델 출력 '{output.name}'이 제거될 입력에 의존하며 대안을 찾을 수 없습니다!")
                logger.warning("   이 출력은 제거됩니다.")
        else:
            new_outputs.append(output)
    
    if len(new_outputs) < len(model.graph.output):
        logger.info(f"출력: {len(model.graph.output)}개 → {len(new_outputs)}개")
    
    # 5. Initializer 확인 (상수 텐서)
    # 제거된 출력이 initializer에 있으면 삭제
    new_initializers = [
        init for init in model.graph.initializer
        if init.name not in dependent_outputs
    ]
    if len(new_initializers) < len(model.graph.initializer):
        logger.info(f"Initializer: {len(model.graph.initializer)}개 → {len(new_initializers)}개")
    
    # 6. Value Info 정리 (중간 텐서 정보)
    new_value_info = [
        vi for vi in model.graph.value_info
        if vi.name not in dependent_outputs
    ]
    if len(new_value_info) < len(model.graph.value_info):
        logger.info(f"Value Info: {len(model.graph.value_info)}개 → {len(new_value_info)}개")
    
    # 7. 새로운 그래프 생성
    new_graph = helper.make_graph(
        new_nodes,
        model.graph.name,
        new_inputs,
        new_outputs,  # 수정된 출력 사용
        new_initializers,
        doc_string=model.graph.doc_string,
        value_info=new_value_info,
    )
    
    # 8. 새로운 모델 생성
    new_model = helper.make_model(
        new_graph,
        producer_name=model.producer_name,
        producer_version=model.producer_version,
        ir_version=model.ir_version,
        opset_imports=model.opset_import,
        doc_string=model.doc_string,
    )
    
    # 메타데이터 복사
    new_model.model_version = model.model_version
    for meta in model.metadata_props:
        new_model.metadata_props.append(meta)
    
    return new_model


def print_model_info(model: onnx.ModelProto, title: str = "모델 정보"):
    """모델 정보 출력"""
    logger.info("=" * 80)
    logger.info(title)
    logger.info("=" * 80)
    logger.info(f"IR Version: {model.ir_version}")
    logger.info(f"Producer: {model.producer_name} {model.producer_version}")
    logger.info(f"Graph Name: {model.graph.name}")
    
    logger.info(f"\n입력 ({len(model.graph.input)}개):")
    for inp in model.graph.input:
        shape = [dim.dim_value if dim.dim_value > 0 else dim.dim_param 
                 for dim in inp.type.tensor_type.shape.dim]
        dtype = onnx.TensorProto.DataType.Name(inp.type.tensor_type.elem_type)
        logger.info(f"  - {inp.name}: {dtype} {shape}")
    
    logger.info(f"\n출력 ({len(model.graph.output)}개):")
    for out in model.graph.output:
        shape = [dim.dim_value if dim.dim_value > 0 else dim.dim_param 
                 for dim in out.type.tensor_type.shape.dim]
        dtype = onnx.TensorProto.DataType.Name(out.type.tensor_type.elem_type)
        logger.info(f"  - {out.name}: {dtype} {shape}")
    
    logger.info(f"\n노드: {len(model.graph.node)}개")
    logger.info(f"Initializer: {len(model.graph.initializer)}개")
    logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="ONNX 모델에서 특정 입력과 연관된 노드 제거"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="입력 ONNX 모델 경로"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="출력 ONNX 모델 경로"
    )
    parser.add_argument(
        "--remove-inputs",
        type=str,
        nargs="+",
        required=True,
        help="제거할 입력 이름 (예: im_shape scale_factor)"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="수정 후 모델 검증"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="상세 로그 출력"
    )
    
    args = parser.parse_args()
    
    # 로그 레벨 설정
    if args.verbose:
        logger.remove()
        logger.add(lambda msg: print(msg, end=""), level="DEBUG")
    
    # 입력 파일 확인
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"입력 파일 없음: {input_path}")
        return 1
    
    # 출력 디렉토리 생성
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"입력 모델: {input_path}")
    logger.info(f"출력 모델: {output_path}")
    
    # 모델 로드
    logger.info("\n모델 로딩 중...")
    model = onnx.load(str(input_path))
    
    # 원본 모델 정보
    print_model_info(model, "원본 모델")
    
    # 입력 검증
    existing_inputs = {inp.name for inp in model.graph.input}
    for inp_name in args.remove_inputs:
        if inp_name not in existing_inputs:
            logger.warning(f"⚠️  입력 '{inp_name}'이 모델에 없습니다. 사용 가능한 입력: {existing_inputs}")
    
    # 입력 및 의존 노드 제거
    logger.info("\n입력 및 의존 노드 제거 중...")
    new_model = remove_inputs_and_dependencies(model, args.remove_inputs)
    
    # 수정된 모델 정보
    print_model_info(new_model, "수정된 모델")
    
    # 모델 검증
    if args.check:
        logger.info("\n모델 검증 중...")
        try:
            onnx.checker.check_model(new_model)
            logger.info("✅ 모델 검증 성공")
        except Exception as e:
            logger.error(f"❌ 모델 검증 실패: {e}")
            logger.warning("   모델이 유효하지 않을 수 있습니다.")
    
    # 모델 저장
    logger.info(f"\n모델 저장 중: {output_path}")
    onnx.save(new_model, str(output_path))
    
    # 파일 크기 비교
    input_size = input_path.stat().st_size / (1024 * 1024)
    output_size = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"파일 크기: {input_size:.2f} MB → {output_size:.2f} MB ({output_size - input_size:+.2f} MB)")
    
    logger.info("\n✅ 완료!")
    logger.info(f"\n다음 단계:")
    logger.info(f"1. 수정된 모델 확인:")
    logger.info(f"   python -c \"import onnx; m = onnx.load('{output_path}'); print([i.name for i in m.graph.input])\"")
    logger.info(f"2. DX Engine용 변환:")
    logger.info(f"   dx_compiler --input {output_path} --output {{output.dxnn}} --input_shape {{shape}}")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
