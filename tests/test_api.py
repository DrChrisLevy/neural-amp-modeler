import io

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from nam_studio.server import create_app


def wav(seconds=2, channels=1):
    audio = 0.1 * np.sin(np.arange(48000 * seconds) * 2 * np.pi * 220 / 48000)
    if channels == 2:
        audio = np.stack([audio, audio], axis=-1)
    stream = io.BytesIO()
    sf.write(stream, audio, 48000, format="WAV", subtype="PCM_24")
    return stream.getvalue()


@pytest.mark.parametrize("legacy_key", ["", "unused-old-key"])
@pytest.mark.parametrize("base_url", ["http://localhost:8000", "https://another-workspace.example"])
def test_fresh_visitors_can_use_the_studio_without_credentials(tmp_path, monkeypatch, legacy_key, base_url):
    # A leftover environment variable must not bring back the retired login flow.
    monkeypatch.setenv("NAM_STUDIO_MODAL", "1")
    monkeypatch.setenv("NAM_STUDIO_KEY", legacy_key)
    server = create_app(tmp_path)
    with TestClient(server, base_url=base_url) as client:
        config = client.get("/api/config")
        assert config.status_code == 200
        assert config.json()["training_available"] is False
        assert "set-cookie" not in config.headers
        assert client.get("/api/state").json()["captures"] == []
        take = client.post("/api/captures/upload", files={
            "dry": ("input.wav", wav(), "audio/wav"),
            "wet": ("output.wav", wav(), "audio/wav"),
        }).json()
        assert not client.cookies
        assert client.get("/api/state").json()["captures"][0]["id"] == take["id"]
        assert take["dry_url"].startswith("/api/")
        assert take["wet_url"].startswith("/api/")
        audio = client.get(take["dry_url"], headers={"Range": "bytes=0-43"})
        assert audio.status_code == 206
        assert audio.content.startswith(b"RIFF")
        assert "cookie" not in audio.request.headers
        assert "authorization" not in audio.request.headers
        with client.websocket_connect("/ws/monitor") as monitor:
            assert monitor.receive_json()["type"] == "state"
        assert not client.cookies


def test_upload_roundtrip_preserves_audio_and_cannot_escape_store(tmp_path):
    client = TestClient(create_app(tmp_path))
    result = client.post("/api/captures/upload", data={"name": "First take"}, files={"dry": ("input-v3.wav", wav(), "audio/wav"), "wet": ("amp-output.wav", wav(), "audio/wav")})
    assert result.status_code == 200
    take = result.json()
    assert take["duration"] == 2
    assert take["name"] == "First take"
    assert take["filenames"] == {"dry": "input-v3.wav", "wet": "amp-output.wav"}
    assert client.get("/api/state").json()["captures"][0]["filenames"] == take["filenames"]
    data, sr = sf.read(io.BytesIO(client.get(take["dry_url"]).content))
    expected, _ = sf.read(io.BytesIO(wav()))
    assert sr == 48000
    np.testing.assert_array_equal(data, expected)
    assert client.get("/api/files/capture/not-an-id/dry.wav").status_code == 400
    assert client.get(f"/api/files/capture/{take['id']}/meta.json").status_code == 404


def test_rejects_mismatched_and_multichannel_uploads(tmp_path):
    client = TestClient(create_app(tmp_path))
    for wrong in (wav(seconds=3), wav(channels=2)):
        response = client.post("/api/captures/upload", files={"dry": ("dry.wav", wav()), "wet": ("wet.wav", wrong)})
        assert response.status_code == 400
    assert client.get("/api/state").json()["captures"] == []


def test_training_dispatch_allows_independent_runs(tmp_path):
    submitted = []

    async def submit(job_id):
        submitted.append(job_id)

    client = TestClient(create_app(tmp_path, submit=submit))
    take = client.post("/api/demo").json()
    response = client.post("/api/train", json={"capture_id": take["id"], "epochs": 50, "model_size": "full"})
    assert response.status_code == 200
    assert submitted == [response.json()["id"]]
    second = client.post("/api/train", json={"capture_id": take["id"]})
    assert second.status_code == 200
    assert second.json()["id"] != response.json()["id"]
    assert submitted == [response.json()["id"], second.json()["id"]]
    assert response.json()["model_size"] == second.json()["model_size"] == "full"
    assert response.json()["epochs"] == 50
    assert second.json()["epochs"] == 400
    assert all(job["status"] == "queued" for job in client.get("/api/state").json()["jobs"])


def test_new_runs_reject_lite_and_packed_without_dispatching(tmp_path):
    submitted = []

    async def submit(job_id):
        submitted.append(job_id)

    client = TestClient(create_app(tmp_path, submit=submit))
    take = client.post("/api/demo").json()
    for size in ("lite", "both", "unknown"):
        response = client.post("/api/train", json={"capture_id": take["id"], "model_size": size})
        assert response.status_code == 400
        assert "A2 Full" in response.json()["detail"]
    assert submitted == []
    assert client.get("/api/state").json()["jobs"] == []


def test_local_mode_does_not_fake_training(tmp_path):
    client = TestClient(create_app(tmp_path))
    take = client.post("/api/demo").json()
    assert client.post("/api/train", json={"capture_id": take["id"]}).status_code == 503


def test_extended_epoch_choices_and_full_default(tmp_path):
    from nam_studio.server import TrainCommand
    from pydantic import ValidationError
    import pytest

    assert TrainCommand(capture_id="x").model_size == "full"
    assert TrainCommand(capture_id="x").epochs == 400
    for epochs in (50, 100, 400, 500, 1000):
        assert TrainCommand(capture_id="x", epochs=epochs).epochs == epochs
    for epochs in (0, 1001):
        with pytest.raises(ValidationError):
            TrainCommand(capture_id="x", epochs=epochs)
