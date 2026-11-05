"""Audio integrity checks that run without PyTorch, NAM, or a GPU."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import soundfile as sf

from tone_twin.training import (
    ALIGNMENT_HEADROOM,
    SAMPLE_RATE,
    _select_model_config,
    _write_audition,
    align_pair,
    analyze_audio,
    estimate_delay,
    split_segments,
    train_job,
)


class AlignmentTests(unittest.TestCase):
    def test_delay_sign_and_nonlinear_polarity(self):
        rng = np.random.default_rng(12)
        x = rng.normal(0, 0.06, 3 * SAMPLE_RATE).astype(np.float32)
        distorted = -0.4 * np.tanh(4 * x)
        for lag in (237, -173):
            with self.subTest(lag=lag):
                if lag > 0:
                    wet = np.r_[np.zeros(lag), distorted[:-lag]]
                else:
                    wet = np.r_[distorted[-lag:], np.zeros(-lag)]
                result = estimate_delay(x, wet)
                self.assertEqual(result["estimated_delay_samples"], lag)
                self.assertEqual(result["applied_delay_samples"], lag - ALIGNMENT_HEADROOM)
                aligned_x, aligned_y = align_pair(x, wet, lag)
                np.testing.assert_allclose(aligned_y, -0.4 * np.tanh(4 * aligned_x), atol=1e-7)

    def test_alignment_rejects_unrelated_audio(self):
        rng = np.random.default_rng(12)
        x = rng.normal(0, 0.06, 2 * SAMPLE_RATE)
        y = rng.normal(0, 0.06, 2 * SAMPLE_RATE)
        with self.assertRaisesRegex(ValueError, "align"):
            estimate_delay(x, y)

    def test_validation_context_does_not_overlap_training(self):
        split = split_segments(12 * SAMPLE_RATE, 6327)
        self.assertEqual(split["train_stop"], split["validation_input_start"])
        self.assertEqual(split["validation_target_start"] - 6326, split["train_stop"])
        self.assertGreater(split["validation_stop"] - split["validation_target_start"], SAMPLE_RATE)


class QualityTests(unittest.TestCase):
    def test_silence_nonfinite_stereo_and_clipping_are_rejected(self):
        bad = {
            "silence": np.zeros(1000),
            "nonfinite": np.r_[np.full(999, 0.1), np.nan],
            "stereo": np.full((1000, 2), 0.1),
            "clipping": np.r_[np.full(990, 0.1), np.full(10, 0.99999)],
        }
        for name, audio in bad.items():
            with self.subTest(name=name), self.assertRaises(ValueError):
                analyze_audio(audio, "Dry audio")

    def test_clean_audio_level_is_preserved(self):
        audio = 0.2 * np.sin(np.arange(1000) * 0.1)
        copy = audio.copy()
        quality = analyze_audio(audio, "Wet audio")
        self.assertAlmostEqual(quality["peak"], np.abs(audio).max())
        self.assertEqual(quality["clipped_fraction"], 0)
        np.testing.assert_array_equal(audio, copy)

    def test_audition_uses_one_shared_gain(self):
        with tempfile.TemporaryDirectory() as path:
            directory = Path(path)
            dry = np.full(1000, 0.1)
            reference = np.full(1000, 0.4)
            prediction = np.full(1000, 2.0)
            gain = _write_audition(directory, dry, reference, {"lite": prediction}, "lite")
            self.assertAlmostEqual(gain, 0.475)
            for name, expected in [("dry", 0.1), ("reference", 0.4), ("prediction", 2.0)]:
                audio, rate = sf.read(directory / f"{name}.wav")
                self.assertEqual(rate, SAMPLE_RATE)
                np.testing.assert_allclose(audio, expected * gain, atol=2e-7)

    def test_failed_capture_updates_status_before_gpu_import(self):
        with tempfile.TemporaryDirectory() as path:
            root = Path(path)
            capture = root / "captures" / "capture-test"
            job = root / "jobs" / "job-test"
            capture.mkdir(parents=True)
            job.mkdir(parents=True)
            (capture / "meta.json").write_text("{}")
            zeros = np.zeros(12 * SAMPLE_RATE)
            for name in ("dry", "wet"):
                sf.write(capture / f"{name}.wav", zeros, SAMPLE_RATE, subtype="PCM_24")
            (job / "status.json").write_text(json.dumps({
                "id": "job-test", "capture_id": "capture-test", "epochs": 10,
                "model_size": "lite", "status": "queued", "created_at": "test",
            }))
            commits = []
            with self.assertRaisesRegex(ValueError, "silent"):
                train_job(path, "job-test", commit=lambda: commits.append(True))
            status = json.loads((job / "status.json").read_text())
            self.assertEqual(status["status"], "failed")
            self.assertIn("silent", status["error"])
            self.assertGreaterEqual(len(commits), 2)


class RecipeTests(unittest.TestCase):
    def test_single_model_keeps_training_recipe(self):
        recipe = {
            "net": {"name": "PackedWaveNet", "config": {"submodels": [
                {"config": {"layers_configs": [{"channels": 3}], "head_scale": 0.01}},
                {"config": {"layers_configs": [{"channels": 8}], "head_scale": 0.01}},
            ]}},
            "loss": {"val_loss": "esr", "mrstft_weight": 0.0005},
            "optimizer": {"lr": 0.004},
        }
        for label, width in [("lite", 3), ("full", 8)]:
            selected = _select_model_config(recipe, label)
            self.assertEqual(selected["net"]["name"], "WaveNet")
            self.assertEqual(selected["net"]["config"]["layers_configs"][0]["channels"], width)
            self.assertEqual(selected["loss"], recipe["loss"])
            self.assertEqual(selected["optimizer"], recipe["optimizer"])
        self.assertEqual(recipe["net"]["name"], "PackedWaveNet")


if __name__ == "__main__":
    unittest.main()
