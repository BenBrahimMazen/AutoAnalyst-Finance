"""Modal deployment for AutoAnalyst Finance (full stack: FastAPI + Streamlit).

Quick start
-----------
1.  pip install modal
2.  modal token new                 # one-time auth; opens a browser
3.  Create the runtime secret (provider + data keys):

        modal secret create autoanalyst-finance \
            LLM_API_KEY=gsk_... \
            LLM_BASE_URL=https://api.groq.com/openai/v1 \
            LLM_MODEL=llama-3.3-70b-versatile \
            TAVILY_API_KEY=tvly-... \
            FRED_API_KEY=...

4.  modal deploy modal_app.py       # prints two URLs: api + frontend

After the first deploy, copy the ``api`` URL Modal prints into ``API_URL`` below
and redeploy, so the Streamlit UI can reach the API:

        modal deploy modal_app.py

Notes
-----
* Two web endpoints are created (one subdomain each):
      api      -> FastAPI JSON API + auto /docs Swagger UI
      frontend -> Streamlit UI
* FinBERT is pre-downloaded into the image at build time, so cold starts load it
  from disk (~seconds) instead of re-downloading ~1.3 GB.
* The API scales to zero when idle (free while idle); the first request after
  idle takes ~10-20 s to warm up. For an always-warm demo, add
  ``min_containers=1`` to the ``@app.function`` on ``api`` (uses credit).
* Run state is in-memory (per-container) and does not survive a cold start —
  fine for a single-user demo; externalize to Redis for real scale.
"""

from __future__ import annotations

import sys

import modal

APP_NAME = "autoanalyst-finance"

# After your first `modal deploy`, replace this with the printed `api` URL, then
# redeploy. It tells the Streamlit UI where the FastAPI backend lives.
API_URL = "https://<workspace>--autoanalyst-finance-api.modal.run"


def _download_finbert() -> None:
    """Pre-download ProsusAI/finbert into the image cache at build time."""
    from transformers import pipeline  # noqa: PLC0415

    pipeline("text-classification", model="ProsusAI/finbert", top_k=None)


# Shared image: WeasyPrint system libs + full Python deps + project code +
# FinBERT baked in. The heavy pip layers are cached across deploys, so only the
# `add_local_dir` layer (your code) re-runs on most deploys.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "libpango-1.0-0", "libpangoft2-1.0-0", "libcairo2", "libgdk-pixbuf-2.0-0"
    )
    .add_local_file("requirements.txt", "/root/requirements.txt")
    .run_commands(
        "pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu",
        "pip install --no-cache-dir -r /root/requirements.txt",
    )
    .add_local_dir("src", "/root/src")
    .run_function(_download_finbert)
)

app = modal.App(APP_NAME, image=image)


@app.function(
    secrets=[
        modal.Secret.from_name(
            "autoanalyst-finance",
            required_keys=["LLM_API_KEY", "TAVILY_API_KEY"],
        )
    ],
    memory=3072,  # torch + FinBERT need ~3 GB at inference time
)
@modal.asgi_app()
def api():
    """FastAPI backend: JSON API + auto-generated /docs UI."""
    sys.path.insert(0, "/root")
    from src.api.main import app as fastapi_app  # noqa: PLC0415

    return fastapi_app


@app.function()
@modal.web_server(8501, startup_timeout=60)
def frontend() -> None:
    """Streamlit UI, pointed at the deployed ``api`` endpoint via ``API_URL``."""
    import os  # noqa: PLC0415
    import subprocess  # noqa: PLC0415

    env = dict(os.environ)
    env["API_URL"] = API_URL
    subprocess.Popen(
        [
            sys.executable, "-m", "streamlit", "run", "src/frontend/app.py",
            "--server.port=8501",
            "--server.address=0.0.0.0",
            "--server.headless=true",
            "--server.enableCORS=false",
            "--server.enableXsrfCORS=false",
            "--server.enableXsrfProtection=false",
        ],
        env=env,
    )
