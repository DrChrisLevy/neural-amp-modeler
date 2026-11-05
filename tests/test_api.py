import io

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

from tone_twin.server import create_app


def wav(seconds=2, channels=1):
    audio = 0.1 * np.sin(np.arange(48000 * seconds) * 2 * np.pi * 220 / 48000)
    if channels == 2:
        audio = np.stack([audio, audio], axis=-1)
    stream = io.BytesIO()
    sf.write(stream, audio, 48000, format="WAV", subtype="PCM_24")
    return stream.getvalue()


def test_private_workspace_requires_key_for_audio_and_state(tmp_path):
    client = TestClient(create_app(tmp_path, key="test-key"))
    assert client.get("/api/config").json()["authenticated"] is False
    assert client.get("/api/state").status_code == 401
    assert client.post("/api/auth", json={"key": "wrong"}).status_code == 401
    assert client.post("/api/auth", json={"key": "test-key"}).status_code == 200
    assert client.get("/api/state").json()["captures"] == []


def test_launch_ticket_is_authenticated_single_use_and_sets_cookie(tmp_path):
    server = create_app(tmp_path, key="private-key")
    local = TestClient(server)
    assert local.post("/api/launch").status_code == 401
    url = local.post("/api/launch", headers={"Authorization": "Bearer private-key"}).json()["url"]
    assert "private-key" not in url
    browser = TestClient(server)
    assert browser.get(url, follow_redirects=False).status_code == 303
    assert browser.get("/api/state").status_code == 200
    assert browser.get(url).status_code == 401


def test_upload_roundtrip_preserves_audio_and_cannot_escape_store(tmp_path):
    client = TestClient(create_app(tmp_path))
    result = client.post("/api/captures/upload", data={"name": "First take"}, files={"dry": ("dry.wav", wav(), "audio/wav"), "wet": ("wet.wav", wav(), "audio/wav")})
    assert result.status_code == 200
    take = result.json()
    assert take["duration"] == 2
    assert take["name"] == "First take"
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


def test_training_dispatch_is_real_and_serialized(tmp_path):
    submitted = []

    async def submit(job_id):
        submitted.append(job_id)

    client = TestClient(create_app(tmp_path, submit=submit))
    take = client.post("/api/demo").json()
    response = client.post("/api/train", json={"capture_id": take["id"], "epochs": 10, "model_size": "lite"})
    assert response.status_code == 200
    assert submitted == [response.json()["id"]]
    assert client.post("/api/train", json={"capture_id": take["id"]}).status_code == 409
    assert client.get("/api/state").json()["jobs"][0]["status"] == "queued"


def test_local_mode_does_not_fake_training(tmp_path):
    client = TestClient(create_app(tmp_path))
    take = client.post("/api/demo").json()
    assert client.post("/api/train", json={"capture_id": take["id"]}).status_code == 503
