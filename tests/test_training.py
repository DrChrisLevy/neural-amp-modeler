"""Audio integrity checks that run without PyTorch, NAM, or a GPU."""

import json
import hashlib
from unittest.mock import patch
from pathlib import Path
import tempfile
import unittest

import numpy as np
import soundfile as sf

from nam_studio.alignment import measure_v3_latency
from nam_studio.training import (
    ALIGNMENT_HEADROOM,
    SAMPLE_RATE,
    _LossHistory,
    _full_model_config,
    _write_audition,
    analyze_audio,
    estimate_delay,
    split_segments,
    is_standard_input,
    check_repeated_output,
    train_job,
)


class LossHistoryTests(unittest.TestCase):
    def test_long_runs_retain_early_middle_and_latest_measurements(self):
        history = _LossHistory()
        for step in range(1, 63_001):
            point = {"step": step, "epoch": step / 63, "loss": 1 / step}
            history.append(point)
            if step in (2001, 31_500, 63_000):
                points = history.snapshot()
                self.assertLessEqual(len(points), 2000)
                self.assertEqual(points[0]["step"], 1)
                self.assertEqual(points[-1], point)
                self.assertTrue(all(a["step"] < b["step"] for a, b in zip(points, points[1:])))
                self.assertLessEqual(max(b["step"] - a["step"] for a, b in zip(points, points[1:])), history.stride)
                for p in points:
                    self.assertEqual(p["loss"], 1 / p["step"])


class AlignmentTests(unittest.TestCase):
    @staticmethod
    def pulse_audio(delays):
        audio = np.zeros(13 * SAMPLE_RATE, dtype="float32")
        for position, delay in zip((504_000, 552_000), delays):
            if delay is not None:
                audio[position + delay:position + delay + 3] = [.001, .1, .01]
        return audio

    def test_buffered_usb_pulses_use_onset_not_peak(self):
        audio = self.pulse_audio((12_846, 12_846))
        for polarity in (1, -1):
            result = measure_v3_latency(audio * polarity)
            self.assertEqual(result["pulse_delays_samples"], [12_846, 12_846])
            self.assertEqual(result["applied_delay_samples"], 12_845)

    def test_aligned_and_short_latency_pulses_keep_one_sample_margin(self):
        for delay in (0, 1, 300, 9_999):
            result = measure_v3_latency(self.pulse_audio((delay, delay)))
            self.assertEqual(result["applied_delay_samples"], delay - 1)

    def test_extended_calibration_rejects_missing_inconsistent_and_late_pulses(self):
        for delays in ((12_846, None), (None, 12_846), (12_846, 12_866), (24_000, 24_000), (-1_000, -1_000)):
            with self.subTest(delays=delays), self.assertRaises(ValueError):
                measure_v3_latency(self.pulse_audio(delays))

    def test_extended_calibration_respects_background_noise(self):
        audio = self.pulse_audio((12_846, 12_846))
        audio[492_000:498_000] = .2
        with self.assertRaises(ValueError):
            measure_v3_latency(audio)

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
    def test_standard_split_excludes_the_repeated_opening_excerpt(self):
        split = split_segments(190 * SAMPLE_RATE, 6347, standard=True)
        self.assertEqual(split["train_start"], 10 * SAMPLE_RATE)
        self.assertEqual(split["train_stop"], 181 * SAMPLE_RATE)
        self.assertEqual(split["validation_input_start"], 181 * SAMPLE_RATE)
        self.assertEqual(split["validation_stop"], 190 * SAMPLE_RATE)
        with self.assertRaises(ValueError):
            split_segments(189 * SAMPLE_RATE, 6347, standard=True)

    def test_standard_detection_requires_matching_audio_not_just_duration(self):
        dry = np.zeros(190 * SAMPLE_RATE, dtype="float32")
        self.assertFalse(is_standard_input(dry))
        digest = hashlib.sha256(dry.astype("<f4").tobytes()).hexdigest()
        with patch("nam_studio.training.NAM_V3_AUDIO_SHA256", digest):
            self.assertTrue(is_standard_input(dry))
            dry[1] = .1
            self.assertFalse(is_standard_input(dry))

    def test_changed_repeated_tone_is_rejected(self):
        audio = np.full(190 * SAMPLE_RATE, .1, dtype="float32")
        self.assertEqual(check_repeated_output(audio), 0)
        audio[-9 * SAMPLE_RATE:] *= 2
        with self.assertRaisesRegex(ValueError, "tone changed"):
            check_repeated_output(audio)

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
            gain = _write_audition(directory, dry, reference, prediction)
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
                "model_size": "full", "status": "queued", "created_at": "test",
            }))
            commits = []
            with self.assertRaisesRegex(ValueError, "silent"):
                train_job(path, "job-test", commit=lambda: commits.append(True))
            status = json.loads((job / "status.json").read_text())
            self.assertEqual(status["status"], "failed")
            self.assertIn("silent", status["error"])
            self.assertGreaterEqual(len(commits), 2)


class RecipeTests(unittest.TestCase):
    def test_worker_rejects_unsupported_models_before_loading_audio_or_gpu(self):
        for model_size in ("lite", "both", "unknown"):
            with self.subTest(model_size=model_size), tempfile.TemporaryDirectory() as path:
                status_path = Path(path) / "jobs" / "job-test" / "status.json"
                status_path.parent.mkdir(parents=True)
                status_path.write_text(json.dumps({
                    "id": "job-test", "capture_id": "capture-test", "epochs": 10,
                    "model_size": model_size, "status": "queued",
                }))
                with self.assertRaisesRegex(ValueError, "A2 Full only"):
                    train_job(path, "job-test")
                self.assertEqual(json.loads(status_path.read_text())["status"], "failed")

    def test_full_model_keeps_training_recipe(self):
        recipe = {
            "net": {"name": "PackedWaveNet", "config": {"submodels": [
                {"config": {"layers_configs": [{"channels": 3}], "head_scale": 0.01}},
                {"config": {"layers_configs": [{"channels": 8}], "head_scale": 0.01}},
            ]}},
            "loss": {"val_loss": "esr", "mrstft_weight": 0.0005},
            "optimizer": {"lr": 0.004},
        }
        selected = _full_model_config(recipe)
        self.assertEqual(selected["net"]["name"], "WaveNet")
        self.assertEqual(selected["net"]["config"]["layers_configs"][0]["channels"], 8)
        self.assertEqual(selected["loss"], recipe["loss"])
        self.assertEqual(selected["optimizer"], recipe["optimizer"])
        self.assertEqual(recipe["net"]["name"], "PackedWaveNet")


if __name__ == "__main__":
    unittest.main()
