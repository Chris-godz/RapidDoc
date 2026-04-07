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

## 1) Offline Demo (Async Pipeline)
- Command: `python demo/demo_offline.py --use-async`
- Description: CLI demo that processes local PDF/image files with the async pipeline. Outputs (markdown/images) are saved under `demo/output-offline-async/`.

## 1-1) Offline Demo (Finegrained Streaming Pipeline)
- Command: `python demo/demo_offline.py --finegrained`
- Description: CLI demo using the 7-stage per-page streaming pipeline. Each page flows through Layout → Formula → PDF-det → OCR-det → Table → OCR-rec stages in a pipelined fashion. Outputs are saved under `demo/output-offline-finegrained/`.

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
