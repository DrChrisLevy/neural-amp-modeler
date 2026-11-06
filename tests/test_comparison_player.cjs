const test = require('node:test');
const assert = require('node:assert/strict');
const vm = require('node:vm');
const fs = require('node:fs');
const path = require('node:path');
const sandbox = {window: {}, AbortController};
vm.runInNewContext(fs.readFileSync(path.join(__dirname, '../src/nam_studio/static/comparison-player.js'), 'utf8'), sandbox);
const deferred = () => { let resolve; const promise = new Promise(r => resolve = r); return {promise, resolve}; };
function setup(fetcher) {
  const context = {
    currentTime: 10, destination: {}, sources: [], gains: [],
    async resume() {},
    async decodeAudioData(data) { return {duration: data.duration || 12, getChannelData: () => new Float32Array([0, .1, -.2])}; },
    createBufferSource() {
      const node = {connect() {}, disconnect() {}, start(...args) { this.startArgs = args; }, stop(when) { this.stoppedAt = when; }};
      this.sources.push(node); return node;
    },
    createGain() {
      const node = {connect() {}, disconnect() {}, gain: {value: 0, cancelScheduledValues() {}, setValueAtTime(value) { this.value = value; }, linearRampToValueAtTime(value) { this.value = value; }}};
      this.gains.push(node); return node;
    },
  };
  const requests = [];
  const engine = new sandbox.window.NAMComparisonAudio({context: () => context, fetcher: async (url, options) => {
    requests.push({url, signal: options.signal});
    return fetcher ? fetcher(url, options) : {ok: true, arrayBuffer: async () => ({duration: url.includes('short') ? 8 : 12})};
  }});
  return {engine, context, requests};
}
const urls = {dry: '/dry', reference: '/reference', model: '/model'};

test('all tracks start together; switching preserves the clock and buffers', async () => {
  const {engine, context, requests} = setup();
  await engine.load('run-a', urls);
  assert.equal(engine.playing, false);
  await engine.play('reference', 2);
  assert.equal(context.sources.length, 3);
  for (const node of context.sources) assert.deepEqual(node.startArgs, [10.006, 2, 10]);
  assert.deepEqual(context.gains.map(n => n.gain.value), [0, 1, 0]);
  context.currentTime = 13.006;
  assert.equal(engine.currentTime, 5);
  await engine.play('model');
  assert.equal(engine.currentTime, 5);
  assert.equal(context.sources.length, 3);
  assert.deepEqual(context.gains.map(n => n.gain.value), [0, 0, 1]);
  await engine.load('run-a', urls);
  assert.equal(requests.length, 3);
});

test('seek, pause, resume and end use a shared timeline', async () => {
  const {engine, context} = setup();
  await engine.load('run-a', {...urls, model: '/short'});
  assert.equal(engine.duration, 8);
  await engine.play('dry', 4);
  const oldNodes = [...context.sources];
  context.currentTime = 12.006;
  engine.pause();
  assert.equal(engine.currentTime, 6);
  assert.equal(engine.playing, false);
  oldNodes.forEach(n => n.onended()); // Fading old nodes must not advance the playhead.
  assert.equal(engine.currentTime, 6);
  await engine.play('model');
  context.sources.slice(-3).forEach(n => assert.equal(n.startArgs[1], 6));
  await engine.play('reference', 99);
  assert.equal(engine.currentTime, 8);
  assert.equal(engine.playing, false);
  await engine.play();
  context.sources.slice(-3).forEach(n => assert.equal(n.startArgs[1], 0));
  context.sources.at(-1).onended();
  assert.equal(engine.playing, false);
  assert.equal(engine.currentTime, 8);
});

test('last gesture wins while AudioContext is resuming; pause cancels autoplay', async () => {
  const {engine, context} = setup();
  await engine.load('run-a', urls);
  const pending = [];
  context.resume = () => { const d = deferred(); pending.push(d); return d.promise; };
  const first = engine.play('dry', 1), second = engine.play('model', 7);
  pending[1].resolve(); await second;
  pending[0].resolve(); await first;
  assert.equal(engine.active, 'model');
  assert.equal(engine.currentTime, 7);
  assert.equal(context.sources.length, 3);
  const late = engine.play('reference', 2);
  engine.pause(); pending[2].resolve(); await late;
  assert.equal(engine.playing, false);
  assert.equal(engine.currentTime, 7);
});

test('navigation cancels pending playback and resets selection without autoplay', async () => {
  const {engine, context} = setup();
  await engine.load('run-a', urls);
  const resume = deferred(); context.resume = () => resume.promise;
  const late = engine.play('model', 3);
  await engine.load('run-b', urls);
  resume.resolve(); await late;
  assert.equal(engine.playing, false);
  assert.equal(engine.active, 'reference');
  assert.equal(engine.currentTime, 0);
  assert.equal(context.sources.length, 0);
  engine.reset();
  assert.equal(engine.key, null);
  assert.equal(engine.duration, 0);
});

test('stale requests cannot replace the audio from a newly selected run', async () => {
  const old = deferred();
  const {engine, requests} = setup(async url => url === '/old' ? old.promise : {ok: true, arrayBuffer: async () => ({duration: 5})});
  const stale = engine.load('old', {dry: '/old'});
  await engine.load('new', {reference: '/new'});
  assert.equal(requests[0].signal.aborted, true);
  old.resolve({ok: true, arrayBuffer: async () => ({duration: 100})}); await stale;
  assert.equal(engine.key, 'new');
  assert.equal(engine.duration, 5);
  assert.equal(engine.tracks.dry.state, 'unavailable');
});

test('missing or failed models leave available original audio playable', async () => {
  const {engine} = setup(async url => ({ok: url !== '/model', arrayBuffer: async () => ({duration: 12})}));
  await engine.load('failed-model', urls);
  assert.equal(engine.tracks.model.state, 'error');
  await engine.play('reference');
  assert.equal(engine.playing, true);
  assert.equal(engine.nodes.length, 2);
  await engine.load('untrained', {dry: '/dry', reference: '/reference'});
  assert.equal(engine.tracks.model.state, 'unavailable');
  assert.equal(engine.playing, false);
});
