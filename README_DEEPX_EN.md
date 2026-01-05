# RapidDoc Usage Examples (DeepX)

These examples assume your virtual environment and models/env variables are already set (e.g., `source venv/bin/activate` and, if needed, run `source set_env.sh ...`).

Prerequisites
```shell
# 1) Activate your venv
source venv/bin/activate

# 2) Build dx-rt (replace with your path)
cd /path/to/dx-rt
./build.sh --clean
cd -

# 3) Install RapidDoc dependencies and editable package
pip install -r requirements.gradio.txt
pip install -e .
```

## 1) Offline Demo (Async Pipeline)
- Command: `python demo/demo_offline.py --use-async`
- Description: CLI demo that processes local PDF/image files with the async pipeline. Outputs (markdown/images) are saved under `demo/output-offline/`.

## 2) Offline API Server (DeepX Default)
- Start server: `python demo/app_offline.py --deepx-default`
  - Uses DeepX as the default engine (preferred over ONNXRuntime).
- Test: in another terminal run `python demo/test_api_offline.py`
  - Sends sample requests and checks the server responses.

## 3) Gradio Web UI
- Command: `python demo/app_gradio.py`
- Description: Launches the Gradio-based web UI (default port 7860) for upload/parse/preview.

## Notes
- If your environment needs dx-rt inference engine thread tuning, run `set_env.sh` with the appropriate arguments before starting the demos/servers.
- Logs and output paths follow each script's internal settings; adjust inside the scripts if you need different locations.
