"""Audio fixtures enter the app through the same upload path as user files."""

import io

import numpy as np
import pytest
import soundfile as sf


@pytest.fixture(scope="session")
def pair_files():
    dry = (0.1 * np.sin(np.arange(48_000 * 12) * 0.03)).astype("float32")
    files = {}
    for name, audio in (("dry", dry), ("wet", np.tanh(4 * dry) * 0.6)):
        stream = io.BytesIO()
        sf.write(stream, audio, 48_000, format="WAV", subtype="PCM_24")
        files[name] = (f"{name}.wav", stream.getvalue(), "audio/wav")
    return files
