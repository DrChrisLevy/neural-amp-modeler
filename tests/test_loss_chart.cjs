const assert = require('node:assert/strict');
const test = require('node:test');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');

const context = vm.createContext({window: {}});
vm.runInContext(fs.readFileSync(path.join(__dirname, '../frontend/src/lib/engines/training-metrics.js'), 'utf8'), context);
const series = (job) => JSON.parse(JSON.stringify(context.window.NAMLossChart.trainingSeries(job)));

test('restores all 500 epochs when old batch history jumps straight to epoch 334', () => {
  const metrics = Array.from({length: 500}, (_, i) => ({epoch: i + 1, loss: .5 / (i + 1), train_loss: .01 / (i + 1)}));
  const result = series({status: 'completed', metrics, training_metrics: [
    {step: 1, epoch: .015873, loss: .02014},
    {step: 21025, epoch: 333.730159, loss: .000139},
    {step: 31500, epoch: 500, loss: .000192},
  ]});
  assert.equal(result.averaged, true);
  assert.equal(result.points.length, 500);
  assert.deepEqual(result.points[0], {epoch: 1, loss: .01});
  assert.deepEqual(result.points[1], {epoch: 2, loss: .005});
  assert.deepEqual(result.points.at(-1), {epoch: 500, loss: .00002});
});

test('shows the live running average without mixing it with raw batch loss', () => {
  const job = {status: 'training', metrics: [{epoch: 1, train_loss: .01}],
    training_metrics: [{epoch: 1.5, loss: .9}], train_epoch_loss: .006};
  assert.deepEqual(series(job).points, [{epoch: 1, loss: .01}, {epoch: 1.5, loss: .006, partial: true}]);
  job.metrics.push({epoch: 2, train_loss: .005});
  job.training_metrics[0].epoch = 2;
  assert.deepEqual(series(job).points, [{epoch: 1, loss: .01}, {epoch: 2, loss: .005}]);
});

test('handles the first epoch, legacy measurements, and missing or invalid values', () => {
  assert.deepEqual(series({}), {points: [], averaged: false});
  const batches = [{epoch: .5, loss: .02}];
  assert.deepEqual(series({training_metrics: batches}), {points: batches, averaged: false});
  assert.deepEqual(series({status: 'training', training_metrics: batches, train_epoch_loss: .01}),
    {points: [{epoch: .5, loss: .01, partial: true}], averaged: true});
  assert.deepEqual(series({metrics: [{epoch: 1, loss: .1}, {epoch: 2, train_loss: NaN}]}).points, []);
  assert.deepEqual(series({metrics: [{epoch: 1, train_loss: 0}]}).points, [{epoch: 1, loss: 0}]);
});
