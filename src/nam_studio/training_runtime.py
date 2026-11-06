"""GPU-only adapters around NAM's unchanged training objective and networks."""

import math

import torch
from nam.train.lightning_module import LightningModule, PackedLightningModule
from nam.data import Dataset as NAMDataset, NormalizeJointDatasetOutput
from torch.utils.data import Dataset, IterableDataset


def prepare_datasets(dry, wet, split, receptive_field, delay):
    """Use upstream slicing, alignment, RMS scaling, and export compensation.

    Split *before* alignment, as NAM does. The repeated opening v3 excerpt is
    excluded by the caller's split, and neither partition borrows context.
    """
    common = dict(x=torch.from_numpy(dry), y=torch.from_numpy(wet), nx=receptive_field,
                  sample_rate=48000, delay=delay, require_input_pre_silence=None)
    training = NAMDataset(**common, start_samples=split["train_start"], stop_samples=split["train_stop"], ny=8192)
    validation = NAMDataset(**common, start_samples=split["validation_input_start"], stop_samples=split["validation_stop"], ny=None)
    rms = (torch.sum(training.y.square()).item() / training.y.numel()) ** 0.5
    NormalizeJointDatasetOutput(level_rms_dbfs=-18.0).apply(training, validation)
    return training, validation, 10 ** (-18 / 20) / rms


class TrainingBatches(IterableDataset):
    """Gather complete batches from resident audio with two device operations.

    Windows and the final small batch match NAM Dataset's nx/ny convention.
    No decoded audio is copied across PCIe during an epoch.
    """

    def __init__(self, dry, wet, receptive_field, ny=8192, batch_size=16, shuffle=True):
        super().__init__()
        self.dry, self.wet = dry, wet
        self.context = receptive_field - 1
        self.ny, self.batch_size, self.shuffle = ny, batch_size, shuffle
        self.windows = (len(dry) - self.context) // ny
        self.x_offsets = torch.arange(self.context + ny, device=dry.device)
        self.y_offsets = torch.arange(ny, device=wet.device) + self.context
        if len(dry) != len(wet) or self.windows < 1:
            raise ValueError("Training audio must contain matching input and target windows")

    def __len__(self):
        return math.ceil(self.windows / self.batch_size)

    def __iter__(self):
        order = (torch.randperm(self.windows, device=self.dry.device) if self.shuffle
                 else torch.arange(self.windows, device=self.dry.device))
        for index in range(0, self.windows, self.batch_size):
            starts = (order[index:index + self.batch_size] * self.ny).unsqueeze(1)
            yield self.dry[starts + self.x_offsets], self.wet[starts + self.y_offsets]


class _TrainingObjective(torch.nn.Module):
    def __init__(self, net, compute):
        super().__init__()
        self.net = net
        self.compute = compute

    def forward(self, dry, wet):
        return self.compute((dry, wet), 0)


class _GraphedTraining:
    """Replay NAM's forward/loss/backward kernels without a compile cold start.

    Optimizer, clipping, scheduler, packed masks, and checkpoint format remain
    under Lightning/NAM. A separate graph covers the final smaller batch.
    """

    use_cuda_graphs = True

    def training_step(self, batch, batch_idx):
        compute = super().training_step
        if not self.use_cuda_graphs or not batch[0].is_cuda:
            return compute(batch, batch_idx)
        if not hasattr(self, "_training_graphs"):
            # Graph wrappers reference the existing net; they must not register a
            # second copy of its parameters or change the checkpoint state dict.
            object.__setattr__(self, "_training_graphs", {})
        key = (tuple(batch[0].shape), tuple(batch[1].shape), batch[0].dtype)
        if key not in self._training_graphs:
            objective = _TrainingObjective(self.net, compute)
            self._training_graphs[key] = torch.cuda.make_graphed_callables(
                objective, tuple(batch), allow_unused_input=True,
            )
        return self._training_graphs[key](*batch)

    def on_train_end(self):
        if hasattr(self, "_training_graphs"):
            self._training_graphs.clear()
        super().on_train_end()


class ValidationChunks(Dataset):
    """Score every held-out sample once, including the final partial chunk.

    Each item already has a batch dimension. Use DataLoader(batch_size=None).
    Context comes only from this validation partition, never the training set.
    """

    def __init__(self, dry, wet, receptive_field, chunk_size=131_072):
        self.dry, self.wet = dry, wet
        self.context = receptive_field - 1
        self.chunk_size = chunk_size
        self.samples = len(wet) - self.context
        if len(dry) != len(wet) or self.samples <= 0:
            raise ValueError("Validation audio must contain matching input and target context")

    def __len__(self):
        return math.ceil(self.samples / self.chunk_size)

    def __getitem__(self, index):
        start = self.context + index * self.chunk_size
        stop = min(start + self.chunk_size, len(self.wet))
        return self.dry[start - self.context:stop].unsqueeze(0), self.wet[start:stop].unsqueeze(0)


class _WholeRecordingValidation:
    """Energy-weighted ESR, rather than averaging ratios of nearly silent chunks.

    NAM's training loss remains untouched. Validation only needs the metric used
    to select checkpoints, so it also avoids computing unused spectral losses.
    The packed model retains NAM's summed metric convention.
    """

    def on_validation_epoch_start(self):
        self._validation_error = None
        self._validation_reference = torch.zeros((), dtype=torch.float64, device=self.device)

    def validation_step(self, batch, batch_idx):
        dry, target = batch
        prediction = self(dry, pad_start=False)
        if prediction.ndim == 2:
            prediction = prediction.unsqueeze(1)
        error = (prediction - target.unsqueeze(1)).square().sum(dim=(0, 2), dtype=torch.float64)
        self._validation_error = error if self._validation_error is None else self._validation_error + error
        self._validation_reference += target.square().sum(dtype=torch.float64)

    def on_validation_epoch_end(self):
        if self._validation_error is None or self._validation_reference.item() <= 0:
            raise ValueError("The held-out recording has no signal energy")
        esr = self._validation_error / self._validation_reference
        if not torch.isfinite(esr).all().item():
            raise RuntimeError("Model produced non-finite validation metrics")
        self.log("val_loss", esr.sum(), on_step=False, on_epoch=True, batch_size=1)


class ValidatedLightningModule(_GraphedTraining, _WholeRecordingValidation, LightningModule):
    pass


class ValidatedPackedLightningModule(_GraphedTraining, _WholeRecordingValidation, PackedLightningModule):
    pass
