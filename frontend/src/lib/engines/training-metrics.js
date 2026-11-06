'use strict';

window.NAMLossChart = class {
  static format(value) {
    if (!Number.isFinite(value)) return '—';
    if (value === 0) return '0';
    return Math.abs(value) < .001 || Math.abs(value) >= 1000 ? value.toExponential(2) : Number(value.toPrecision(4)).toString();
  }

  static trainingSeries(job) {
    const valid = (value) => Number.isFinite(value) && value >= 0;
    const points = (Array.isArray(job.metrics) ? job.metrics : [])
      .filter((p) => Number.isFinite(p.epoch) && valid(p.train_loss))
      .map((p) => ({epoch: p.epoch, loss: p.train_loss}));
    const batches = Array.isArray(job.training_metrics) ? job.training_metrics : [];
    const latest = batches.at(-1);
    if (job.status === 'training' && valid(job.train_epoch_loss)
        && Number.isFinite(latest?.epoch) && latest.epoch > (points.at(-1)?.epoch ?? 0)) {
      points.push({epoch: latest.epoch, loss: job.train_epoch_loss, partial: true});
    }
    // Saved epoch means repair older runs whose batch history was over-thinned.
    return points.length ? {points, averaged: true} : {points: batches, averaged: false};
  }

};
