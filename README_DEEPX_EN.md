# RapidDoc Usage Examples (DeepX)

These examples assume your virtual environment and models/env variables are already set (e.g., `source venv/bin/activate` and, if needed, run `source ./deepx_scripts/set_env.sh ...`).

Prerequisites
```shell
# 1) Activate your venv
source venv/bin/activate

# 2) Download sample models into this repo root
./setup.sh --force-remove-models

# 3) Build dx-rt (replace with your path)
cd /path/to/dx-rt
./build.sh --clean
cd -

# 4) Install RapidDoc dependencies and editable package
pip install -r requirements.gradio.txt
pip install -e .
```

---

## CLI Options Reference

### Pipeline Mode (mutually exclusive)

| Option | Description |
|:---|:---|
| `--finegrained` | **(Default)** 7-stage per-page streaming pipeline. Stages (Layout → Plan → Formula → PDF-det → OCR-det → Table → OCR-rec) run concurrently with inter-stage queues. Fastest mode — ~17% faster than `--use-async`. |
| `--use-async` | Batch async pipeline (TrueAsyncPipeline). Processes all pages through each stage before moving to the next. |
| `--no-async` | Sequential sync pipeline. Processes pages one at a time through all stages. Slowest but simplest. |

### Inference Options

| Option | Description |
|:---|:---|
| `--no-formula` | Skip formula recognition entirely. Formula regions detected by layout are kept as **cropped images** instead of LaTeX text. Removes the largest single-stage bottleneck (~99s). |
| `--force-ocr` | Ignore PDF text metadata and use image→OCR for all text extraction. Useful for scanned PDFs or model evaluation. Increases OCR workload but parallel pipeline absorbs most overhead. |

### Other Options

| Option | Description |
|:---|:---|
| `PATH` | PDF files or directories to process. Defaults to `demo/pdfs/` if omitted. |
| `--output-dir DIR` | Output directory. Defaults to `demo/output-offline-{mode}/`. |

---

## 1) Offline Demo (Default: Finegrained Pipeline)
- Command: `python demo/demo_offline.py`
- Description: CLI demo that processes local PDF/image files with the **finegrained streaming pipeline** (default). Outputs (markdown/images) are saved under `demo/output-offline-finegrained/`.
- With options: `python demo/demo_offline.py test_files --no-formula --force-ocr`

## 1-1) Offline Demo (Async Pipeline)
- Command: `python demo/demo_offline.py --use-async`
- Description: CLI demo using the batch async pipeline. Outputs are saved under `demo/output-offline-async/`.

After the run, a `performance_summary.md` file is automatically saved in the output directory with per-PDF stage latency and throughput, along with overall aggregated stats.

**Example output (`performance_summary.md`):**

```
FinegrainedStreamingPipeline PERFORMANCE SUMMARY
Pipeline Step       Avg Latency     Throughput
----------------------------------------------------------
 Layout               364.38 ms        2.7 FPS
 Formula              488.82 ms        2.0 FPS
 PDF-det                1.46 ms      683.2 FPS
 OCR-det               91.03 ms       11.0 FPS
 Table                479.04 ms        2.1 FPS
 OCR-rec               27.72 ms       36.1 FPS
----------------------------------------------------------
 Total Stages         231.31 s
----------------------------------------------------------
 Total Pages                  66
 Total Time              128.2 s
 Overall                 0.5 pages/s
```

The per-PDF breakdown (with actual filenames) is written to `performance_summary.md` in the output directory.

## 2) Offline API Server (DeepX Default)
- Start server: `python demo/app_offline.py --deepx-default`
  - Uses DeepX as the default engine (preferred over ONNXRuntime).
- Test: in another terminal run `python demo/test_api_offline.py`
  - Sends sample requests and checks the server responses.

## 3) Gradio Web UI
- Command: `python demo/gradio_app.py`
- Description: Launches the Gradio-based web UI (default port 7860) for upload/parse/preview.

## Notes
- If your environment needs dx-rt inference engine thread tuning, run `./deepx_scripts/set_env.sh` with the appropriate arguments before starting the demos/servers.
- Logs and output paths follow each script's internal settings; adjust inside the scripts if you need different locations.
