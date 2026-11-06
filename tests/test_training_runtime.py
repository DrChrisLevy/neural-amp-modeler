"""Run in the training image; skipped locally when NAM/PyTorch are absent."""

import unittest
import importlib.resources
import json

try:
    import torch
    from tone_twin.training_runtime import TrainingBatches, ValidationChunks, _WholeRecordingValidation, ValidatedLightningModule, prepare_datasets
    from tone_twin.training import _select_model_config, split_segments
    from nam.data import Dataset
except ModuleNotFoundError as exc:
    if exc.name not in {"torch", "nam"}:
        raise
    raise unittest.SkipTest("Requires the NAM training image") from exc


class ValidationHarness(_WholeRecordingValidation, torch.nn.Module):
    def __init__(self, context=6, packed=False):
        super().__init__()
        self.context, self.packed = context, packed
        self.device = torch.device("cpu")
        self.metrics = {}

    def forward(self, dry, pad_start=False):
        prediction = dry[:, self.context:] + 0.01
        return torch.stack((prediction, prediction * 2), dim=1) if self.packed else prediction

    def log(self, name, value, **kwargs):
        self.metrics[name] = value.item()


class ValidationRegressionTests(unittest.TestCase):
    def test_normalization_uses_only_training_energy_and_preserves_source(self):
        import numpy as np
        dry = np.sin(np.arange(48000 * 12) * .017).astype("float32") * .1
        wet = dry * 2
        split = split_segments(len(dry), 7)
        wet[split["train_stop"]:] *= .1
        original_dry, original_wet = dry.copy(), wet.copy()
        train, val, gain = prepare_datasets(dry, wet, split, 7, 13)
        self.assertAlmostEqual(float(train.y.square().mean().sqrt()), 10**(-18/20), places=6)
        np.testing.assert_array_equal(dry, original_dry)
        np.testing.assert_array_equal(wet, original_wet)
        np.testing.assert_allclose(train.x.numpy(), dry[:split["train_stop"]-13])
        np.testing.assert_allclose(val.y.numpy() / gain, wet[split["train_stop"]+13:], atol=1e-7)
        self.assertEqual(len(val.x), len(dry) - split["train_stop"] - 13)

    def test_export_hook_compensates_training_gain_exactly_once(self):
        import numpy as np
        recipe = json.loads(importlib.resources.files("nam.train._resources").joinpath("config_model_packed.json").read_text())
        net = ValidatedLightningModule.init_from_config(_select_model_config(recipe, "lite")).net
        net.sample_rate = 48000
        dry = np.sin(np.arange(48000 * 12) * .017).astype("float32") * .1
        train, val, gain = prepare_datasets(dry, dry * .2, split_segments(len(dry), net.receptive_field), net.receptive_field, 0)
        # Handshakes on both partitions must not double-apply compensation.
        train.handshake(net)
        val.handshake(net)
        hooks = [h for h in net.export_model_dict_post_hooks if isinstance(h, Dataset._ScaleOutputHook)]
        self.assertEqual(len(hooks), 1)
        self.assertAlmostEqual(hooks[0].scale, 1/gain, places=10)
        exported = {"architecture": "WaveNet", "config": {"head_scale": .01}, "weights": [1, 2, .01], "metadata": {}}
        hooks[0].apply(exported)
        self.assertAlmostEqual(exported["config"]["head_scale"], .01/gain)
        self.assertAlmostEqual(exported["weights"][-1], .01/gain)

    def test_resident_batches_match_nam_windows_and_preserve_small_final_batch(self):
        dry = torch.linspace(-0.1, 0.1, 111)
        wet = dry * 2
        reference = Dataset(dry, wet, nx=7, ny=10, sample_rate=48000, require_input_pre_silence=None)
        batches = TrainingBatches(dry, wet, 7, ny=10, batch_size=4, shuffle=False)
        actual = list(batches)
        self.assertEqual([len(x) for x, y in actual], [4, 4, 2])
        xs, ys = torch.cat([x for x, y in actual]), torch.cat([y for x, y in actual])
        for i in range(len(reference)):
            expected_x, expected_y = reference[i]
            torch.testing.assert_close(xs[i], expected_x)
            torch.testing.assert_close(ys[i], expected_y)

    def test_shuffling_visits_every_training_window_once(self):
        signal = torch.arange(111, dtype=torch.float32)
        batches = TrainingBatches(signal, signal, 7, ny=10, batch_size=4)
        starts = torch.cat([x[:, 0] for x, y in batches]).sort().values
        torch.testing.assert_close(starts, torch.arange(0, 100, 10, dtype=torch.float32))

    def test_every_target_including_partial_tail_is_scored_once(self):
        signal = torch.arange(111, dtype=torch.float32)
        chunks = ValidationChunks(signal, signal, receptive_field=7, chunk_size=32)
        self.assertEqual(len(chunks), 4)
        targets = []
        for i in range(len(chunks)):
            dry, target = chunks[i]
            self.assertEqual(dry.shape[-1] - target.shape[-1], 6)
            torch.testing.assert_close(dry[:, 6:], target)
            targets.append(target)
        torch.testing.assert_close(torch.cat(targets, dim=-1).squeeze(0), signal[6:])

    def test_quiet_passages_do_not_dominate_checkpoint_selection(self):
        # One audible sample surrounded by silence used to explode averaged ESR.
        target = torch.zeros(111)
        target[30] = 0.5
        expected = ((target[6:] + 0.01 - target[6:]).double().square().sum()
                    / target[6:].double().square().sum()).item()
        for chunk_size in (17, 32, 105):
            model = ValidationHarness()
            chunks = ValidationChunks(target, target, 7, chunk_size)
            model.on_validation_epoch_start()
            for i in range(len(chunks)):
                model.validation_step(chunks[i], i)
            model.on_validation_epoch_end()
            self.assertAlmostEqual(model.metrics["val_loss"], expected, places=7)

    def test_packed_metric_sums_the_two_whole_recording_errors(self):
        target = torch.linspace(-0.2, 0.2, 111)
        model = ValidationHarness(packed=True)
        chunks = ValidationChunks(target, target, 7, 32)
        model.on_validation_epoch_start()
        for i in range(len(chunks)):
            model.validation_step(chunks[i], i)
        model.on_validation_epoch_end()
        prediction = torch.stack((target[6:] + 0.01, (target[6:] + 0.01) * 2))
        expected = ((prediction - target[6:]).double().square().sum()
                    / target[6:].double().square().sum()).item()
        self.assertAlmostEqual(model.metrics["val_loss"], expected, places=6)

    def test_silent_validation_fails_instead_of_selecting_a_checkpoint(self):
        model = ValidationHarness()
        model.on_validation_epoch_start()
        model.validation_step((torch.zeros(1, 38), torch.zeros(1, 32)), 0)
        with self.assertRaisesRegex(ValueError, "signal energy"):
            model.on_validation_epoch_end()



class GraphRegressionTests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "Requires CUDA")
    def test_graph_matches_eager_loss_and_gradients_across_changed_inputs(self):
        recipe = json.loads(importlib.resources.files("nam.train._resources").joinpath("config_model_packed.json").read_text())
        config = _select_model_config(recipe, "lite")
        eager = ValidatedLightningModule.init_from_config(config).cuda()
        graphed = ValidatedLightningModule.init_from_config(config).cuda()
        graphed.load_state_dict(eager.state_dict())
        eager.use_cuda_graphs = False
        snapshots = []
        for batch_size, level in ((4, 0.02), (4, 0.08), (2, 0.04)):
            dry = torch.randn(batch_size, eager.net.receptive_field - 1 + 8192, device="cuda") * level
            wet = torch.tanh(dry[:, -8192:] * 4) * 0.6
            eager.zero_grad(set_to_none=True)
            graphed.zero_grad(set_to_none=True)
            loss = eager.training_step((dry, wet), 0)
            graph_loss = graphed.training_step((dry, wet), 0)
            snapshots.append(graph_loss.detach().clone())
            torch.testing.assert_close(graph_loss, loss, rtol=1e-4, atol=1e-6)
            loss.backward()
            graph_loss.backward()
            for a, b in zip(eager.parameters(), graphed.parameters()):
                if a.grad is None:
                    self.assertIsNone(b.grad)
                else:
                    torch.testing.assert_close(b.grad, a.grad, rtol=2e-3, atol=1e-6)
        self.assertEqual(set(eager.state_dict()), set(graphed.state_dict()))
        self.assertNotEqual(snapshots[0].item(), snapshots[1].item())
