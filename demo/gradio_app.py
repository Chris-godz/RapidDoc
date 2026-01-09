#!/usr/bin/env python3
# Copyright (c) Opendatalab. All rights reserved.
"""
RapidDoc Gradio Web UI - For Closed Environment
DX Engine based PDF parsing web interface

Usage:
    source venv/bin/activate
    source deepx_scripts/set_env.sh 1 2 1 3 2 4
    python demo/gradio_app.py
"""
import os
import sys
import time
import tempfile
import shutil
from pathlib import Path
from typing import List, Tuple

import gradio as gr
from loguru import logger

# =============================================================================
# Environment Setup Check: Verify if deepx_scripts/set_env.sh has been executed
# =============================================================================
def check_environment_setup():
    """
    Check if deepx_scripts/set_env.sh 1 2 1 3 2 4 has been executed
    Exit with warning if required environment variables are not set
    """
    required_env_vars = {
        "CUSTOM_INTER_OP_THREADS_COUNT": "1",
        "CUSTOM_INTRA_OP_THREADS_COUNT": "2",
        "DXRT_DYNAMIC_CPU_THREAD": "1",
        "DXRT_TASK_MAX_LOAD": "3",
        "NFH_INPUT_WORKER_THREADS": "2",
        "NFH_OUTPUT_WORKER_THREADS": "4",
    }
    
    missing_vars = []
    for var_name, expected_value in required_env_vars.items():
        if var_name not in os.environ:
            missing_vars.append(var_name)
    
    if missing_vars:
        logger.error("=" * 80)
        logger.error("❌ Environment setup is not complete!")
        logger.error("")
        logger.error("Please run the following command first:")
        logger.error("  $ source ./deepx_scripts/set_env.sh 1 2 1 3 2 4")
        logger.error("")
        logger.error(f"Missing environment variables: {', '.join(missing_vars)}")
        logger.error("=" * 80)
        sys.exit(1)
    
    logger.info("✅ Environment setup verified")
    for var_name, expected_value in required_env_vars.items():
        logger.info(f"  {var_name}={os.environ.get(var_name)}")

check_environment_setup()
# =============================================================================

# =============================================================================
# Closed Environment Setup: Block model downloads
# =============================================================================
os.environ['MINERU_MODEL_SOURCE'] = 'local'  # Use local models only

# Block requests and urllib to completely prevent external download attempts
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

from rapid_doc.cli.common import convert_pdf_bytes_to_bytes_by_pypdfium2, prepare_env, read_fn
from rapid_doc.data.data_reader_writer import FileBasedDataWriter
from rapid_doc.utils.draw_bbox import draw_layout_bbox, draw_span_bbox
from rapid_doc.utils.enum_class import MakeMode
from rapid_doc.backend.pipeline.pipeline_analyze import doc_analyze as pipeline_doc_analyze
from rapid_doc.backend.pipeline.pipeline_middle_json_mkcontent import union_make as pipeline_union_make
from rapid_doc.backend.pipeline.model_json_to_middle_json import result_to_middle_json as pipeline_result_to_middle_json

from rapidocr import EngineType as OCREngineType
from rapid_doc.model.layout.rapid_layout_self import ModelType as LayoutModelType
from rapid_doc.model.layout.rapid_layout_self.utils.typings import EngineType as LayoutEngineType
from rapid_doc.model.formula.rapid_formula_self import ModelType as FormulaModelType
from rapid_doc.model.formula.rapid_formula_self.utils.typings import EngineType as FormulaEngineType
from rapid_doc.model.table.rapid_table_self import ModelType as TableModelType

# Project root directory
__dir__ = os.path.dirname(os.path.abspath(__file__))
project_root = Path(__dir__).parent.absolute()
onnx_models_dir = project_root / "onnx_models"
dxnn_models_dir = project_root / "dxnn_models"

# Default output directory
DEFAULT_OUTPUT_DIR = project_root / "demo" / "output-gradio"
DEFAULT_OUTPUT_DIR.mkdir(exist_ok=True)


def extract_performance_summary(perf_logs: str) -> str:
    """
    Extract per-PDF performance statistics from performance logs and return as summary string
    """
    if not perf_logs:
        return ""
    
    # Find Performance Summary section only (not per-PDF statistics)
    lines = perf_logs.split('\n')
    in_summary_section = False
    summary_data = []
    total_time = None
    
    for line in lines:
        # Detect Performance Summary section (not "Performance by PDF")
        if "Performance Summary" in line:
            in_summary_section = True
            continue
        
        # Exit if we hit "Performance by PDF" or "per-PDF" section
        if in_summary_section and ("Performance by PDF" in line):
            break
        
        if in_summary_section:
            # Extract model performance lines
            if any(emoji in line for emoji in ['📊', '📐', '📄', '🔍', '📋', '✍️']):
                # Parse line format: "📊 Layout   [    dxengine] |    3.98s ( 14.6%) |   13it | 0.306 s/it |   3.26 it/s"
                try:
                    parts = line.split('|')
                    if len(parts) >= 5:
                        # Extract model name and engine
                        model_part = parts[0].strip()
                        model_name = model_part.split('[')[0].strip()
                        engine = model_part.split('[')[1].split(']')[0].strip() if '[' in model_part else 'N/A'
                        
                        # Extract metrics
                        time_part = parts[1].strip()  # "3.98s ( 14.6%)"
                        time_match = time_part.split('(')[0].strip()
                        percentage = time_part.split('(')[1].split(')')[0].strip() if '(' in time_part else 'N/A'
                        
                        items = parts[2].strip()  # "13it"
                        s_per_it = parts[3].strip()  # "0.306 s/it"
                        it_per_s = parts[4].strip()  # "3.26 it/s"
                        
                        summary_data.append({
                            'model': model_name,
                            'engine': engine,
                            'time': time_match,
                            'percentage': percentage,
                            'items': items,
                            's_per_it': s_per_it,
                            'it_per_s': it_per_s
                        })
                except:
                    pass
            
            # Extract total processing time
            if "Total processing time" in line:
                try:
                    # Format: "🔥 Total processing time: 27.31s"
                    total_time = line.split(':')[1].strip()
                except:
                    pass
    
    if not summary_data:
        return ""
    
    # Build markdown table
    table_lines = [
        "\n" + "=" * 80,
        "📈 **Performance Summary**",
        "=" * 80,
        "",
        "| Model | Engine | Time | % | Items | s/it | it/s |",
        "|-------|--------|------|---|-------|------|------|"
    ]
    
    for data in summary_data:
        row = f"| {data['model']} | {data['engine']} | {data['time']} | {data['percentage']} | {data['items']} | {data['s_per_it']} | {data['it_per_s']} |"
        table_lines.append(row)
    
    if total_time:
        table_lines.append("")
        table_lines.append(f"**🔥 Total: {total_time}**")
    
    table_lines.append("=" * 80)
    
    return '\n'.join(table_lines)


def get_model_config(
    layout_engine: str,
    ocr_engine: str,
    formula_engine: str,
    table_engine: str,
):
    """Generate model configuration"""
    # Layout configuration
    layout_config = {"model_type": LayoutModelType.PP_DOCLAYOUT_L}
    if layout_engine == "dxengine":
        layout_config["engine_type"] = LayoutEngineType.DXENGINE
        layout_config["model_dir_or_path"] = str(dxnn_models_dir / "pp_doclayout_l_part1.dxnn")
        layout_config["sub_model_path"] = str(onnx_models_dir / "pp_doclayout_l_part2.onnx")
    else:
        layout_config["engine_type"] = LayoutEngineType.ONNXRUNTIME
        layout_config["model_dir_or_path"] = str(onnx_models_dir / "pp_doclayout_l.onnx")

    # OCR configuration
    ocr_config = {}
    if ocr_engine == "dxengine":
        ocr_config["engine_type"] = "dxengine"
        ocr_config["Det.model_path"] = str(dxnn_models_dir / "ch_PP-OCRv5_server_det.dxnn")
        ocr_config["Rec.model_path"] = str(dxnn_models_dir / "ch_PP-OCRv5_rec_server_infer.dxnn")
        ocr_config["char_dict_path"] = str(project_root / "value_compare" / "recognition" / "character_dict_from_onnx.txt")
    elif ocr_engine == "paddle":
        ocr_config["Det.engine_type"] = OCREngineType.PADDLE
        ocr_config["Rec.engine_type"] = OCREngineType.PADDLE
    else:
        ocr_config["Det.engine_type"] = OCREngineType.ONNXRUNTIME
        ocr_config["Rec.engine_type"] = OCREngineType.ONNXRUNTIME
        ocr_config["Det.model_path"] = str(onnx_models_dir / "ch_PP-OCRv5_server_det.onnx")
        ocr_config["Rec.model_path"] = str(onnx_models_dir / "ch_PP-OCRv5_rec_server_infer.onnx")

    ocr_config.update({
        "use_det_mode": 'auto',
        "use_multi_det_model": True,
        "Det.model_paths": {
            1: str(dxnn_models_dir / "det_v5_640_640.dxnn"),
            2: str(dxnn_models_dir / "det_v5_320_640.dxnn"),
            4: str(dxnn_models_dir / "det_v5_160_640.dxnn"),
            10: str(dxnn_models_dir / "det_v5_64_640.dxnn"),
        },
        "use_multi_rec_model": True,
        "Rec.model_paths": {
            3: str(dxnn_models_dir / "rec_v5_ratio_3.dxnn"),
            5: str(dxnn_models_dir / "rec_v5_ratio_5.dxnn"),
            10: str(dxnn_models_dir / "rec_v5_ratio_10.dxnn"),
            15: str(dxnn_models_dir / "rec_v5_ratio_15.dxnn"),
            25: str(dxnn_models_dir / "rec_v5_ratio_25.dxnn"),
            35: str(dxnn_models_dir / "rec_v5_ratio_35.dxnn"),
        },
        "save_debug_images": False,
    })

    # Formula configuration
    formula_config = {"model_type": FormulaModelType.PP_FORMULANET_PLUS_M}
    if formula_engine == "dxengine":
        formula_config["engine_type"] = "dxengine"
        formula_config["model_dir_or_path"] = str(dxnn_models_dir / "pp_formulanet_plus_m.dxnn")
    else:
        formula_config["engine_type"] = FormulaEngineType.ONNXRUNTIME
        formula_config["model_dir_or_path"] = str(onnx_models_dir / "pp_formulanet_plus_m.onnx")

    # Table configuration
    table_config = {"model_type": TableModelType.UNET}
    if table_engine == "dxengine":
        table_config["engine_type"] = "dxengine"
        table_config["model_dir_or_path"] = str(dxnn_models_dir / "unet.dxnn")
    elif table_engine == "torch":
        table_config["engine_type"] = "torch"
    else:
        table_config["engine_type"] = "onnxruntime"
        table_config["model_dir_or_path"] = str(onnx_models_dir / "unet.onnx")

    checkbox_config = {"checkbox_enable": False}
    image_config = {
        "extract_original_image": False,
        "extract_original_image_iou_thresh": 0.5,
    }

    return layout_config, ocr_config, formula_config, table_config, checkbox_config, image_config


def parse_document(
    file_path: str,
    parse_method: str,
    formula_enable: bool,
    table_enable: bool,
    layout_engine: str,
    ocr_engine: str,
    formula_engine: str,
    table_engine: str,
    use_async_pipeline: bool,
    start_page: int,
    end_page: int,
    progress=gr.Progress(),
) -> Tuple[str, str, str, List[str]]:
    """
    Document parsing function
    
    Returns:
        (markdown_content, info_text, layout_pdf_path, image_list)
    """
    try:
        if not file_path:
            return "", "❌ Please upload a file.", None, []

        progress(0.1, desc="Preparing configuration...")
        
        # File information
        file_name = Path(file_path).stem
        file_suffix = Path(file_path).suffix.lower()
        
        # Check supported formats
        pdf_suffixes = [".pdf"]
        image_suffixes = [".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"]
        if file_suffix not in pdf_suffixes + image_suffixes:
            return "", f"❌ Unsupported file format: {file_suffix}", None, []

        progress(0.2, desc="Configuring models...")
        
        # Model configuration
        layout_config, ocr_config, formula_config, table_config, checkbox_config, image_config = get_model_config(
            layout_engine, ocr_engine, formula_engine, table_engine
        )

        progress(0.3, desc="Reading file...")
        
        # Read file
        pdf_bytes = read_fn(file_path)
        
        # Process page range
        end_page_id = end_page if end_page > 0 else None
        if end_page_id and end_page_id < start_page:
            return "", "❌ End page must be greater than start page.", None, []
        
        new_pdf_bytes = convert_pdf_bytes_to_bytes_by_pypdfium2(pdf_bytes, start_page, end_page_id)

        progress(0.4, desc="Starting document analysis...")
        logger.info("=" * 80)
        logger.info(f"Parsing started: {file_name}")
        logger.info(f"Formula recognition: {'Enabled' if formula_enable else 'Disabled'}")
        logger.info(f"Table recognition: {'Enabled' if table_enable else 'Disabled'}")
        logger.info(f"Engines: Layout={layout_engine}, OCR={ocr_engine}, Formula={formula_engine}, Table={table_engine}")
        logger.info("=" * 80)
        
        # Model inference (capture logger to collect performance statistics)
        start_time = time.time()
        
        # Add handler to capture logs
        import io
        log_stream = io.StringIO()
        log_handler = logger.add(log_stream, format="{message}", level="INFO")
        
        infer_results, all_image_lists, all_page_dicts, lang_list, ocr_enabled_list = pipeline_doc_analyze(
            [new_pdf_bytes],
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
        
        # Remove log handler and extract log content
        logger.remove(log_handler)
        perf_logs = log_stream.getvalue()
        log_stream.close()
        
        progress(0.7, desc="Generating results...")
        
        # Process results
        model_list = infer_results[0]
        images_list = all_image_lists[0]
        pdf_dict = all_page_dicts[0]
        _lang = lang_list[0]
        _ocr_enable = ocr_enabled_list[0]
        
        # Set output directory
        local_image_dir, local_md_dir = prepare_env(str(DEFAULT_OUTPUT_DIR), file_name, parse_method)
        image_writer, md_writer = FileBasedDataWriter(local_image_dir), FileBasedDataWriter(local_md_dir)

        # Generate Middle JSON
        middle_json = pipeline_result_to_middle_json(
            model_list, images_list, pdf_dict, image_writer, _lang, _ocr_enable,
            formula_enable, ocr_config=ocr_config, image_config=image_config
        )

        pdf_info = middle_json["pdf_info"]

        progress(0.85, desc="Visualizing layout...")
        
        # Draw layout bbox
        layout_pdf_path = os.path.join(local_md_dir, f"{file_name}_layout.pdf")
        draw_layout_bbox(pdf_info, new_pdf_bytes, local_md_dir, f"{file_name}_layout.pdf")

        progress(0.9, desc="Generating Markdown...")
        
        # Generate Markdown
        image_dir = str(os.path.basename(local_image_dir))
        md_content = pipeline_union_make(pdf_info, MakeMode.MM_MD, image_dir)
        
        # Save Markdown
        md_writer.write_string(f"{file_name}.md", md_content)
        
        total_time = time.time() - start_time
        total_pages = len(images_list)
        
        progress(1.0, desc="Complete!")
        
        # Parse and format performance statistics
        perf_summary = extract_performance_summary(perf_logs)
        
        # Debugging: Check if log is empty
        if not perf_summary:
            logger.warning("Failed to parse performance logs.")
            logger.debug(f"Captured log length: {len(perf_logs)} characters")
            # Output log sample (first 500 characters)
            if perf_logs:
                logger.debug(f"Log sample:\n{perf_logs[:500]}")
        
        # Generate info text
        info_text = f"""✅ Parsing Complete!

📄 File: {file_name}{file_suffix}
📊 Pages Processed: {total_pages} pages
⏱️ Total Time: {total_time:.2f}s
⚡ Average Speed: {total_time/total_pages:.3f} s/it | {total_pages/total_time:.2f} it/s

💾 Output Directory: {local_md_dir}

{perf_summary if perf_summary else '※ See terminal logs for detailed performance statistics.'}
"""
        
        logger.info(f"Parsing complete: {total_time:.2f}s, {total_pages} pages")
        
        # Collect extracted images (for Gradio Gallery)
        import glob
        extracted_images = []
        if os.path.exists(local_image_dir):
            # Find all image files
            for ext in ['*.png', '*.jpg', '*.jpeg']:
                image_files = glob.glob(os.path.join(local_image_dir, ext))
                extracted_images.extend(sorted(image_files))
        
        logger.info(f"Extracted images: {len(extracted_images)}")
        
        return md_content, info_text, layout_pdf_path, extracted_images

    except Exception as e:
        logger.exception(e)
        error_text = f"❌ Error occurred:\n{str(e)}"
        return "", error_text, None, []


# Create Gradio interface
def create_ui():
    with gr.Blocks(title="RapidDoc - DX Engine", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🚀 RapidDoc - Document Parsing (DX Engine)
        
        **High-Performance Document Parsing System for Closed Environments**
        
        Upload PDF or image files to extract text, tables, formulas, and more.
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                # File upload
                file_input = gr.File(
                    label="📁 File Upload (PDF or Image)",
                    file_types=[".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"]
                )
                
                # Basic settings
                with gr.Group():
                    gr.Markdown("### ⚙️ Basic Settings")
                    parse_method = gr.Radio(
                        choices=["auto", "ocr", "txt"],
                        value="auto",
                        label="Parsing Method",
                        info="auto: Auto select | ocr: Force OCR | txt: Text extraction only"
                    )
                    formula_enable = gr.Checkbox(
                        value=True,
                        label="Enable Formula Recognition"
                    )
                    table_enable = gr.Checkbox(
                        value=True,
                        label="Enable Table Recognition"
                    )
                    use_async_pipeline = gr.Checkbox(
                        value=True,
                        label="Enable Async Pipeline",
                        info="Run DX models with async scheduler for higher throughput"
                    )
                
                # Page range
                with gr.Group():
                    gr.Markdown("### 📄 Page Range")
                    with gr.Row():
                        start_page = gr.Number(
                            value=0,
                            label="Start Page",
                            precision=0,
                            minimum=0
                        )
                        end_page = gr.Number(
                            value=0,
                            label="End Page (0=All)",
                            precision=0,
                            minimum=0
                        )
                
                # Engine settings
                with gr.Accordion("🔧 Engine Settings", open=False):
                    layout_engine = gr.Dropdown(
                        choices=["dxengine", "onnxruntime"],
                        value="dxengine",
                        label="Layout Engine"
                    )
                    ocr_engine = gr.Dropdown(
                        choices=["dxengine", "onnxruntime", "paddle"],
                        value="dxengine",
                        label="OCR Engine"
                    )
                    formula_engine = gr.Dropdown(
                        choices=["onnxruntime", "dxengine"],
                        value="onnxruntime",
                        label="Formula Engine"
                    )
                    table_engine = gr.Dropdown(
                        choices=["dxengine", "onnxruntime"],
                        value="dxengine",
                        label="Table Engine"
                    )
                
                # Parse button
                parse_btn = gr.Button("🚀 Start Parsing", variant="primary", size="lg")
            
            with gr.Column(scale=2):
                # Information output
                with gr.Accordion("ℹ️ Parsing Information", open=True):
                    info_output = gr.Textbox(
                        label="",
                        lines=20,
                        max_lines=50,
                        show_copy_button=True
                    )
                
                # Separate results by tabs
                with gr.Tabs():
                    with gr.Tab("📖️ Markdown Preview"):
                        md_preview = gr.Markdown(
                            label="Rendered Markdown",
                            value="",
                            height=600
                        )
                    
                    with gr.Tab("📝 Markdown Source"):
                        md_output = gr.Textbox(
                            label="Markdown Text (for copy)",
                            lines=60,
                            show_copy_button=True
                        )
                    
                    with gr.Tab("🖼️ Extracted Images"):
                        image_gallery = gr.Gallery(
                            label="Images Extracted from Document",
                            columns=3,
                            height="auto",
                            object_fit="contain"
                        )
                    
                    with gr.Tab("🎨 Layout Visualization"):
                        layout_output = gr.File(
                            label="Layout PDF (Downloadable)",
                        )
        
        # Connect events
        parse_btn.click(
            fn=parse_document,
            inputs=[
                file_input,
                parse_method,
                formula_enable,
                table_enable,
                layout_engine,
                ocr_engine,
                formula_engine,
                table_engine,
                use_async_pipeline,
                start_page,
                end_page,
            ],
            outputs=[md_output, info_output, layout_output, image_gallery]
        ).then(
            fn=lambda md: md,  # Pass the same markdown to preview
            inputs=[md_output],
            outputs=[md_preview]
        )
        
    
    return demo


if __name__ == "__main__":
    
    logger.info("=" * 80)
    logger.info("RapidDoc Gradio UI Starting...")
    logger.info("Mode: Closed Environment (Network Blocked)")
    logger.info("=" * 80)
    
    demo = create_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
