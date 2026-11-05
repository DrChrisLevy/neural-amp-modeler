"""Deploy the studio and its isolated NAM worker with `uv run modal deploy modal_app.py`."""

from pathlib import Path

import modal

app = modal.App("tone-twin")
data = modal.Volume.from_name("tone-twin-data", create_if_missing=True, version=2)
auth = modal.Secret.from_name("tone-twin-auth")
package = Path(__file__).parent / "src" / "tone_twin"

web_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libsndfile1")
    .pip_install("fastapi==0.162.0", "numpy==2.5.4", "soundfile==0.14.0", "python-multipart==0.0.33", "uvicorn==0.54.0")
    .env({"TONE_TWIN_MODAL": "1", "TONE_TWIN_DATA": "/data"})
    .add_local_dir(package, "/root/tone_twin", ignore=["__pycache__", "*.pyc"])
)
training_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libsndfile1", "libportaudio2", "python3-tk")
    .pip_install("neural-amp-modeler==0.13.0", "pytorch-lightning<=2.6.1", "soundfile>=0.13,<1")
    .env({"MPLBACKEND": "Agg"})
    .run_commands("python -c 'import nam, tkinter, soundfile; from nam.train.lightning_module import LightningModule'")
    .add_local_dir(package, "/root/tone_twin", ignore=["__pycache__", "*.pyc", "static"])
)


@app.function(image=training_image, gpu=["L4", "A10G", "T4"], cpu=4, memory=16384, volumes={"/data": data}, timeout=3600, max_containers=1)
def train(job_id: str):
    from tone_twin.training import train_job

    data.reload()
    return train_job("/data", job_id, commit=data.commit)


# A single web container keeps the one-user capture connection and controls together.
# GPU jobs run in a separate function and don't interrupt the studio's audio ingestion.
@app.function(image=web_image, volumes={"/data": data}, secrets=[auth], cpu=1, memory=2048, timeout=3600, max_containers=1, scaledown_window=120)
@modal.concurrent(max_inputs=64)
@modal.asgi_app()
def web():
    from tone_twin.server import create_app

    async def submit(job_id):
        await train.spawn.aio(job_id)

    return create_app("/data", volume=data, submit=submit)
