# Copyright (c) Opendatalab. All rights reserved.
"""
RapidDoc demo for closed networks.
Parses PDFs using only locally stored ONNX models.

Usage:
    # 1. Run once via CLI
    python demo_offline.py

    # 2. Run the API server (stays alive)
    python demo/app_offline.py

    # Example curl call (server must be running):
    curl -X POST "http://localhost:8888/file_parse" \\
      -F "files=@demo/pdfs/example.pdf" \\
      -F "output_dir=./output-api" \\
      -F "formula_enable=true" \\
      -F "table_enable=true" \\
      -F "layout_engine=dxengine" \\
      -F "ocr_engine=dxengine" \\
      -F "return_md=true"

Notes:
    - This file (demo_offline.py): one-shot CLI script
    - app_offline.py: API server; keep one server up and send multiple requests
    - For detailed API usage, see demo/README_API_OFFLINE.md
"""
import argparse
import copy
import json
import os
import time
from pathlib import Path

# =============================================================================
# 환경 변수 체크 (./deepx_scripts/set_env.sh 1 2 1 3 2 4 설정 필요)
# =============================================================================
required_env_vars = {
    'CUSTOM_INTER_OP_THREADS_COUNT': '1',
    'CUSTOM_INTRA_OP_THREADS_COUNT': '2',
    'DXRT_DYNAMIC_CPU_THREAD': '1',
    'DXRT_TASK_MAX_LOAD': '3',
    'NFH_INPUT_WORKER_THREADS': '2',
    'NFH_OUTPUT_WORKER_THREADS': '4'
}

missing_vars = []
incorrect_vars = []

for var_name, expected_value in required_env_vars.items():
    actual_value = os.environ.get(var_name)
    if actual_value is None:
        missing_vars.append(var_name)
    elif actual_value != expected_value:
        incorrect_vars.append(f"{var_name}={actual_value} (expected: {expected_value})")

if missing_vars or incorrect_vars:
    print("=" * 80)
    print("Error: required environment variables are not set correctly.")
    print("-" * 80)
    if missing_vars:
        print(f"Missing variables: {', '.join(missing_vars)}")
    if incorrect_vars:
        print(f"Variables with unexpected values: {', '.join(incorrect_vars)}")
    print("-" * 80)
    print("Please run the following command and try again:")
    print("  source ./deepx_scripts/set_env.sh 1 2 1 3 2 4")
    print("=" * 80)
    import sys
    sys.exit(1)

# =============================================================================
# 폐쇄망 설정: 모델 다운로드 차단
# =============================================================================
os.environ['MINERU_MODEL_SOURCE'] = 'local'  # 로컬 모델만 사용

# requests와 urllib를 차단하여 외부 다운로드 시도 완전 방지
import requests
import urllib.request
import urllib.error

_original_requests_get = requests.get
_original_urllib_urlopen = urllib.request.urlopen
_original_urllib_urlretrieve = urllib.request.urlretrieve

def _blocked_requests_get(*args, **kwargs):
    raise requests.exceptions.ConnectionError("Network access blocked in closed environment")

def _blocked_urllib_urlopen(*args, **kwargs):
    raise urllib.error.URLError("Network access blocked in closed environment")

def _blocked_urllib_urlretrieve(*args, **kwargs):
    raise urllib.error.URLError("Network access blocked in closed environment")

requests.get = _blocked_requests_get
urllib.request.urlopen = _blocked_urllib_urlopen
urllib.request.urlretrieve = _blocked_urllib_urlretrieve
# =============================================================================

from loguru import logger

from rapid_doc.cli.common import convert_pdf_bytes_to_bytes_by_pypdfium2, prepare_env, read_fn
from rapid_doc.data.data_reader_writer import FileBasedDataWriter
from rapid_doc.utils.draw_bbox import draw_layout_bbox, draw_span_bbox
from rapid_doc.utils.enum_class import MakeMode
from rapid_doc.backend.pipeline.pipeline_analyze import doc_analyze as pipeline_doc_analyze
from rapid_doc.backend.pipeline.pipeline_middle_json_mkcontent import union_make as pipeline_union_make
from rapid_doc.backend.pipeline.model_json_to_middle_json import result_to_middle_json as pipeline_result_to_middle_json

from rapidocr import EngineType as OCREngineType, OCRVersion, ModelType as OCRModelType
from rapid_doc.model.layout.rapid_layout_self import ModelType as LayoutModelType
from rapid_doc.model.layout.rapid_layout_self.utils.typings import EngineType as LayoutEngineType
from rapid_doc.model.formula.rapid_formula_self import ModelType as FormulaModelType
from rapid_doc.model.formula.rapid_formula_self.utils.typings import EngineType as FormulaEngineType
from rapid_doc.model.table.rapid_table_self import ModelType as TableModelType

def do_parse(
    output_dir,  # Output directory for storing parsing results
    pdf_file_names: list[str],  # List of PDF file names to be parsed
    pdf_bytes_list: list[bytes],  # List of PDF bytes to be parsed
    parse_method="auto",  # The method for parsing PDF, default is 'auto'
    formula_enable=True,  # Enable formula parsing (model_type 명시로 해결)
    table_enable=True,  # Enable table parsing (UNET 모델 사용 - paddle_cls.onnx 불필요)
    # Engine 선택 플래그
    layout_engine="dxengine",  # "onnxruntime", "dxengine", "openvino"
    ocr_engine="dxengine",  # "onnxruntime", "dxengine", "openvino", "torch", "paddle"
    formula_engine="onnxruntime",  # "onnxruntime", "dxengine", "openvino"
    table_engine="dxengine",  # "onnxruntime", "torch"
    formula_rec_enable=True,  # False: skip ONNX formula inference, keep regions as images
    f_draw_layout_bbox=True,  # Whether to draw layout bounding boxes
    f_draw_span_bbox=True,  # Whether to draw span bounding boxes
    f_dump_md=True,  # Whether to dump markdown files
    f_dump_middle_json=True,  # Whether to dump middle JSON files
    f_dump_model_output=True,  # Whether to dump model output files
    f_dump_orig_pdf=True,  # Whether to dump original PDF files
    f_dump_content_list=True,  # Whether to dump content list files
    f_make_md_mode=MakeMode.MM_MD,  # The mode for making markdown content, default is MM_MD
    start_page_id=0,  # Start page ID for parsing, default is 0
    end_page_id=None,  # End page ID for parsing, default is None (parse all pages until the end of the document)
    use_async_pipeline=True,  # Whether to use async pipeline for parallel processing
):
    # =========================================================================
    # 모델 경로 설정 (엔진별)
    # =========================================================================
    # 프로젝트 루트 디렉토리
    project_root = Path(__file__).parent.parent.absolute()
    onnx_models_dir = project_root / "onnx_models"
    dxnn_models_dir = project_root / "dxnn_models"  # DX Engine 모델 디렉토리
    
    # =========================================================================
    # Layout 모델 설정
    # =========================================================================
    layout_config = {
        "model_type": LayoutModelType.PP_DOCLAYOUT_L,
    }
    
    # Engine-specific Layout settings
    if layout_engine.lower() == "dxengine":
        layout_config["engine_type"] = LayoutEngineType.DXENGINE
        layout_config["model_dir_or_path"] = str(dxnn_models_dir / "pp_doclayout_l_part1.dxnn")
        layout_config["sub_model_path"] = str(onnx_models_dir / "pp_doclayout_l_part2.onnx")
        logger.info("Layout model: DX Engine")
    elif layout_engine.lower() == "openvino":
        layout_config["engine_type"] = LayoutEngineType.OPENVINO
        layout_config["model_dir_or_path"] = str(onnx_models_dir / "pp_doclayout_l.onnx")
        logger.info("Layout model: OpenVINO")
    else:  # onnxruntime (default)
        layout_config["engine_type"] = LayoutEngineType.ONNXRUNTIME
        layout_config["model_dir_or_path"] = str(onnx_models_dir / "pp_doclayout_l.onnx")
        logger.info("Layout model: ONNX Runtime")

    # =========================================================================
    # OCR 모델 설정
    # =========================================================================
    ocr_config = {}
    # Common OCR settings
    ocr_config.update({
        # Skip font path to prevent downloads in closed environments.
        # Fonts are only used for visualization and not required for OCR.
        # rapidocr may try to download fonts if missing, but urllib is blocked
        # above so it will error and proceed with default behavior.
        
        # 추가 설정
        # "Rec.rec_batch_num": 1,
        "use_det_mode": 'auto',  # auto: PDF 추출 우선 → OCR | txt: PDF만 | ocr: 무조건 OCR
        "engine_type": "dxengine",
        
        # 기본 단일 모델 경로 (fallback용 - multi-model 사용 시에도 필요)
        "Det.model_path": str(dxnn_models_dir / "det_v5_640_640.dxnn"),
        "Rec.model_path": str(dxnn_models_dir / "rec_v5_ratio_10.dxnn"),
        
        "char_dict_path": str(project_root / "value_compare" / "recognition" / "character_dict_from_onnx.txt"),
        
        # Multi-model detection 설정 (DX OCR에서 사용)
        "use_multi_det_model": True,  # True로 설정 시 ratio 기반 multi-model detection 사용
        "Det.model_paths": {  # ratio별 모델 경로 (use_multi_det_model=True일 때만 사용)
            1: str(dxnn_models_dir / "det_v5_640_640.dxnn"),  # H/W ratio ~1.5 (기본)
            2: str(dxnn_models_dir / "det_v5_320_640.dxnn"),  # H/W ratio ~2.5
            4: str(dxnn_models_dir / "det_v5_160_640.dxnn"),  # H/W ratio ~ 7.5
            10: str(dxnn_models_dir / "det_v5_64_640.dxnn"),  # H/W ratio > 7.5
        },
        
        # Multi-model recognition 설정 (DX OCR에서 사용)
        "use_multi_rec_model": True,  # True로 설정 시 ratio 기반 multi-model 사용
        "Rec.model_paths": {  # ratio별 모델 경로 (use_multi_rec_model=True일 때만 사용)
            3: str(dxnn_models_dir / "rec_v5_ratio_3.dxnn"),
            5: str(dxnn_models_dir / "rec_v5_ratio_5.dxnn"),
            10: str(dxnn_models_dir / "rec_v5_ratio_10.dxnn"),
            15: str(dxnn_models_dir / "rec_v5_ratio_15.dxnn"),
            25: str(dxnn_models_dir / "rec_v5_ratio_25.dxnn"),
            35: str(dxnn_models_dir / "rec_v5_ratio_35.dxnn"),
        },
        
        # Debug 이미지 저장 설정 (DX OCR에서 사용)
        "save_debug_images": False,  # True로 설정 시 detection input/output 저장
        "debug_save_dir": os.path.join(output_dir, "ocr_debug"),  # 저장 디렉토리
    })

    # =========================================================================
    # Formula 모델 설정
    # =========================================================================
    formula_config = {
        "model_type": FormulaModelType.PP_FORMULANET_PLUS_M,
    }
    if not formula_rec_enable:
        formula_config["formula_rec_enable"] = False
    
    # Engine-specific Formula settings
    if formula_engine.lower() == "dxengine":
        logger.error("=" * 80)
        logger.error("Error: Formula model is not supported by DX Engine.")
        logger.error("Supported engines: onnxruntime, openvino")
        logger.error("=" * 80)
        import sys
        sys.exit(1)
    elif formula_engine.lower() == "openvino":
        formula_config["engine_type"] = FormulaEngineType.OPENVINO
        formula_config["model_dir_or_path"] = str(onnx_models_dir / "pp_formulanet_plus_m.onnx")
        logger.info("Formula model: OpenVINO")
    else:  # onnxruntime (default)
        formula_config["engine_type"] = FormulaEngineType.ONNXRUNTIME
        formula_config["model_dir_or_path"] = str(onnx_models_dir / "pp_formulanet_plus_m.onnx")
        logger.info("Formula model: ONNX Runtime")

    # =========================================================================
    # Table model settings
    # =========================================================================
    table_config = {}
    
    # Table model_type selection
    # UNET: no paddle_cls needed, detects ruled tables only
    # UNET_SLANET_PLUS: needs paddle_cls, handles ruled/unruled tables (default)
    table_config["model_type"] = TableModelType.UNET
    logger.info("Table model type: UNET (no paddle_cls, ruled tables only)")
    
    # Engine-specific Table settings
    if table_engine.lower() == "dxengine":
        table_config["engine_type"] = "dxengine"
        table_config["model_dir_or_path"] = str(dxnn_models_dir / "unet.dxnn")
        logger.info("Table engine: DX Engine")
    elif table_engine.lower() == "torch":
        table_config["engine_type"] = "torch"
        logger.info("Table engine: PyTorch")
    else:  # onnxruntime (default)
        table_config["engine_type"] = "onnxruntime"
        table_config["model_dir_or_path"] = str(onnx_models_dir / "unet.onnx")
        logger.info("Table engine: ONNX Runtime")
    
    # UNET 모델 경로 설정

    checkbox_config = {
        # 체크박스 인식 (OpenCV 기반, 오검출 가능성 있음)
        "checkbox_enable": False,
    }

    # 이미지 추출 설정
    image_config = {
        "extract_original_image": False,  # pypdfium2로 원본 이미지 추출
        "extract_original_image_iou_thresh": 0.5,  # IOU 임계값
    }
    # =========================================================================
    
    for idx, pdf_bytes in enumerate(pdf_bytes_list):
        new_pdf_bytes = convert_pdf_bytes_to_bytes_by_pypdfium2(pdf_bytes, start_page_id, end_page_id)
        pdf_bytes_list[idx] = new_pdf_bytes

    # =========================================================================
    # Measure model inference performance
    # =========================================================================
    logger.info("=" * 80)
    logger.info("Model inference started")
    start_time = time.time()
    
    infer_results, all_image_lists, all_page_dicts, lang_list, ocr_enabled_list, *_ = pipeline_doc_analyze(
        pdf_bytes_list, 
        parse_method=parse_method, 
        formula_enable=formula_enable,
        table_enable=table_enable,
        layout_config=layout_config, 
        ocr_config=ocr_config, 
        formula_config=formula_config, 
        table_config=table_config, 
        checkbox_config=checkbox_config,
        use_async_pipeline=use_async_pipeline,
    )
    
    total_time = time.time() - start_time
    total_pages = sum([len(img_list) for img_list in all_image_lists])
    logger.info(f"Total inference time: {total_time:.2f}s")
    logger.info(f"Total pages: {total_pages}it")
    logger.info(f"Average speed: {total_time/total_pages:.3f} s/it | {total_pages/total_time:.2f} it/s")
    logger.info("=" * 80)

    for idx, model_list in enumerate(infer_results):
        model_json = copy.deepcopy(model_list)
        pdf_file_name = pdf_file_names[idx]
        local_image_dir, local_md_dir = prepare_env(output_dir, pdf_file_name, parse_method)
        image_writer, md_writer = FileBasedDataWriter(local_image_dir), FileBasedDataWriter(local_md_dir)

        images_list = all_image_lists[idx]
        pdf_dict = all_page_dicts[idx]
        _lang = lang_list[idx]
        _ocr_enable = ocr_enabled_list[idx]
        middle_json = pipeline_result_to_middle_json(
            model_list, images_list, pdf_dict, image_writer, _lang, _ocr_enable, 
            formula_enable, ocr_config=ocr_config, image_config=image_config
        )

        pdf_info = middle_json["pdf_info"]

        pdf_bytes = pdf_bytes_list[idx]
        if f_draw_layout_bbox:
            draw_layout_bbox(pdf_info, pdf_bytes, local_md_dir, f"{pdf_file_name}_layout.pdf")

        if f_draw_span_bbox:
            draw_span_bbox(pdf_info, pdf_bytes, local_md_dir, f"{pdf_file_name}_span.pdf")

        if f_dump_orig_pdf:
            md_writer.write(
                f"{pdf_file_name}_origin.pdf",
                pdf_bytes,
            )

        if f_dump_md:
            image_dir = str(os.path.basename(local_image_dir))
            md_content_str = pipeline_union_make(pdf_info, f_make_md_mode, image_dir)
            md_writer.write_string(
                f"{pdf_file_name}.md",
                md_content_str,
            )

        if f_dump_content_list:
            image_dir = str(os.path.basename(local_image_dir))
            content_list = pipeline_union_make(pdf_info, MakeMode.CONTENT_LIST, image_dir)
            md_writer.write_string(
                f"{pdf_file_name}_content_list.json",
                json.dumps(content_list, ensure_ascii=False, indent=4),
            )

        if f_dump_middle_json:
            md_writer.write_string(
                f"{pdf_file_name}_middle.json",
                json.dumps(middle_json, ensure_ascii=False, indent=4),
            )

        if f_dump_model_output:
            md_writer.write_string(
                f"{pdf_file_name}_model.json",
                json.dumps(model_json, ensure_ascii=False, indent=4),
            )

        logger.info(f"local output dir is {local_md_dir}")


def parse_doc(
        path_list: list[Path],
        output_dir,
        method="auto",
        formula_enable=True,
        table_enable=False,
        start_page_id=0,
        end_page_id=None,
        # Engine 선택
        layout_engine="dxengine",
        ocr_engine="dxengine",
        formula_engine="onnxruntime",
        table_engine="dxengine",
        use_async_pipeline=True,
        formula_rec_enable=True,
):
    """
        Parameter description:
        path_list: List of document paths to be parsed, can be PDF or image files.
        output_dir: Output directory for storing parsing results.
        method: the method for parsing pdf:
            auto: Automatically determine the method based on the file type.
            txt: Use text extraction method.
            ocr: Use OCR method for image-based PDFs.
            Without method specified, 'auto' will be used by default.
        formula_enable: Enable formula parsing, default is True
        table_enable: Enable table parsing, default is False
        start_page_id: Start page ID for parsing, default is 0
        end_page_id: End page ID for parsing, default is None (parse all pages until the end of the document)
    """
    try:
        file_name_list = []
        pdf_bytes_list = []
        for path in path_list:
            file_name = str(Path(path).stem)
            pdf_bytes = read_fn(path)
            file_name_list.append(file_name)
            pdf_bytes_list.append(pdf_bytes)
        do_parse(
            output_dir=output_dir,
            pdf_file_names=file_name_list,
            pdf_bytes_list=pdf_bytes_list,
            parse_method=method,
            formula_enable=formula_enable,
            table_enable=table_enable,
            start_page_id=start_page_id,
            end_page_id=end_page_id,
            layout_engine=layout_engine,
            ocr_engine=ocr_engine,
            formula_engine=formula_engine,
            table_engine=table_engine,
            formula_rec_enable=formula_rec_enable,
            use_async_pipeline=use_async_pipeline,
        )
    except Exception as e:
        logger.exception(e)


if __name__ == '__main__':
    # =========================================================================
    # CLI 인자 파싱
    # =========================================================================
    parser = argparse.ArgumentParser(description='RapidDoc PDF Parser - Offline Mode')
    parser.add_argument('--use-async', dest='use_async', action='store_true',
                        help='Enable async pipeline')
    parser.add_argument('--no-async', dest='use_async', action='store_false',
                        help='Disable async pipeline (default)')
    parser.set_defaults(use_async=False)  # Default: sync mode
    args = parser.parse_args()
    
    # =========================================================================
    # 모델 활성화 설정
    # =========================================================================
    FORMULA_ENABLE = True   # 수식 인식 모델 사용 여부 (True/False)
    FORMULA_REC_ENABLE = True  # False: ONNX 추론 건너뛰고 수식 영역을 이미지로 유지
    TABLE_ENABLE = True     # 표 인식 모델 사용 여부 (True/False) - UNET 모델 사용 (paddle_cls 불필요)
    
    # =========================================================================
    # 엔진 선택 설정
    # =========================================================================
    # 각 모델별로 사용할 엔진을 선택할 수 있습니다.
    # 
    # 지원 엔진:
    #   Layout  : "onnxruntime", "dxengine", "openvino"
    #   OCR     : "onnxruntime", "dxengine", "openvino", "torch", "paddle"
    #   Formula : "onnxruntime", "dxengine", "openvino"
    #   Table   : "onnxruntime", "dxengine", "torch"
    #
    # 주의사항:
    #   - dxengine 사용 시: dxnn_models/ 디렉토리에 .dxnn 파일 필요
    #   - onnxruntime 사용 시: onnx_models/ 디렉토리에 .onnx 파일 필요
    #   - openvino 사용 시: openvino 패키지 설치 필요
    #
    # 테이블 인식 관련:
    #   - 현재 ModelType.UNET 사용 (paddle_cls.onnx 불필요)
    #   - unet.onnx 모델만 있으면 됨 (유선 테이블 전용)
    #   - 무선 테이블도 인식하려면 paddle_cls.onnx + slanet_plus.onnx 필요
    # =========================================================================
    
    LAYOUT_ENGINE = "dxengine"   # Layout 모델 엔진
    OCR_ENGINE = "dxengine"      # OCR 모델 엔진
    FORMULA_ENGINE = "onnxruntime"  # Formula 모델 엔진
    TABLE_ENGINE = "dxengine"    # Table 모델 엔진
    
    # DX Engine 사용 예시 (주석 해제하여 사용)
    # LAYOUT_ENGINE = "dxengine"
    # OCR_ENGINE = "dxengine"
    # FORMULA_ENGINE = "dxengine"
    # TABLE_ENGINE = "dxengine"
    # =========================================================================
    
    __dir__ = os.path.dirname(os.path.abspath(__file__))
    pdf_files_dir = os.path.join(__dir__, "pdfs")
    # pdf_files_dir = os.path.join(__dir__, "images") # 이미지를 input으로 넣었을 때 정확도 이슈 큼 -> 진행하지 않겠음.
    output_dir = os.path.join(__dir__, "output-offline")
    pdf_suffixes = [".pdf"]
    image_suffixes = [".png", ".jpeg", ".jpg"]

    doc_path_list = []
    for doc_path in list(Path(pdf_files_dir).glob('*')):
        if doc_path.suffix in pdf_suffixes + image_suffixes:
            doc_path_list.append(doc_path)

    logger.info("=" * 80)
    logger.info(f"Running in closed-network mode: processing {len(doc_path_list)} files")
    logger.info(f"Formula recognition: {'enabled' if FORMULA_ENABLE else 'disabled'}"
                + ("" if FORMULA_REC_ENABLE else " (rec disabled — image only)"))
    logger.info(f"Table recognition: {'enabled' if TABLE_ENABLE else 'disabled'}")
    logger.info(f"Sync / Async mode: {'async' if args.use_async else 'sync'}")
    logger.info("-" * 80)
    logger.info("Engine configuration:")
    logger.info(f"  Layout  Engine: {LAYOUT_ENGINE}")
    logger.info(f"  OCR     Engine: {OCR_ENGINE}")
    logger.info(f"  Formula Engine: {FORMULA_ENGINE}")
    logger.info(f"  Table   Engine: {TABLE_ENGINE}")
    logger.info("=" * 80)
    
    parse_doc(
        doc_path_list, 
        output_dir, 
        formula_enable=FORMULA_ENABLE, 
        table_enable=TABLE_ENABLE,
        layout_engine=LAYOUT_ENGINE,
        ocr_engine=OCR_ENGINE,
        formula_engine=FORMULA_ENGINE,
        table_engine=TABLE_ENGINE,
        formula_rec_enable=FORMULA_REC_ENABLE,
        use_async_pipeline=args.use_async,
    )
