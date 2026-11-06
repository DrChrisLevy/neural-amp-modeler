'use strict';

(() => {
  const SOURCES = ['dry', 'reference', 'model'];
  const clamp = (value, max) => Math.max(0, Math.min(max, Number(value) || 0));
  const clock = (seconds) => {
    const value = Math.max(0, seconds || 0);
    return `${Math.floor(value / 60)}:${String(Math.floor(value % 60)).padStart(2, '0')}.${Math.floor(value % 1 * 10)}`;
  };

  // All clips share one audio clock and start sample. Switching only changes gains.
  class ComparisonAudio {
    constructor({changed = () => {}, unauthorized = () => {}, context, fetcher = (...args) => fetch(...args)} = {}) {
      this.changed = changed;
      this.unauthorized = unauthorized;
      this.contextFactory = context || (() => new (window.AudioContext || window.webkitAudioContext)());
      this.fetcher = fetcher;
      this.key = null;
      this.tracks = {};
      this.nodes = [];
      this.duration = this.offset = this.startedAt = 0;
      this.generation = this.intent = this.playGeneration = 0;
      this.active = 'reference';
      this.playing = this.loading = false;
    }
    get currentTime() {
      return clamp(this.offset + (this.playing ? Math.max(0, this.context.currentTime - this.startedAt) : 0), this.duration);
    }
    load(key, urls) {
      if (key === this.key) return this.pending;
      this.pause();
      this.abort?.abort();
      this.abort = new AbortController();
      const signal = this.abort.signal;
      const generation = ++this.generation;
      this.key = key;
      this.offset = this.duration = 0;
      this.active = 'reference';
      this.loading = true;
      this.tracks = Object.fromEntries(SOURCES.map((source) => [source, {state: urls[source] ? 'loading' : 'unavailable'}]));
      this.changed();
      this.pending = Promise.all(SOURCES.filter((source) => urls[source]).map(async (source) => {
        try {
          const response = await this.fetcher(urls[source], {credentials: 'same-origin', signal});
          if (generation !== this.generation) return;
          if (response.status === 401) { this.unauthorized(); throw new Error('Unlock your workspace to listen.'); }
          if (!response.ok) throw new Error('Audio could not load.');
          const bytes = await response.arrayBuffer();
          if (generation !== this.generation) return;
          this.context ||= this.contextFactory();
          const buffer = await this.context.decodeAudioData(bytes);
          if (generation !== this.generation) return;
          if (!Number.isFinite(buffer.duration) || buffer.duration <= 0) throw new Error('Audio is empty.');
          const samples = buffer.getChannelData(0);
          const peaks = Array.from({length: 600}, (_, bin) => {
            let peak = 0;
            const end = Math.floor((bin + 1) * samples.length / 600);
            for (let i = Math.floor(bin * samples.length / 600); i < end; i++) peak = Math.max(peak, Math.abs(samples[i]));
            return peak;
          });
          this.tracks[source] = {state: 'ready', buffer, peaks};
        } catch (error) {
          if (generation !== this.generation) return;
          this.tracks[source] = {state: 'error', error: error.message || 'Audio could not load.'};
        }
      })).then(() => {
        if (generation !== this.generation) return;
        const available = SOURCES.filter((source) => this.tracks[source].buffer);
        this.duration = available.length ? Math.min(...available.map((source) => this.tracks[source].buffer.duration)) : 0;
        if (!available.includes(this.active)) this.active = available[0] || 'reference';
        this.loading = false;
        this.changed();
      });
      return this.pending;
    }
    reset() {
      this.pause();
      this.abort?.abort();
      ++this.generation;
      this.key = null;
      this.tracks = {};
      this.offset = this.duration = 0;
      this.loading = false;
      this.changed();
    }
    async play(source = this.active, position = null) {
      if (this.loading || !this.tracks[source]?.buffer || !this.duration) return;
      const intent = ++this.intent;
      const generation = this.generation;
      // Resume directly from the gesture; a later pause/navigation cancels this intent.
      await this.context.resume();
      if (intent !== this.intent || generation !== this.generation) return;
      this.active = source;
      if (this.playing && position === null) {
        this._gains();
      } else {
        const offset = position === null ? (this.currentTime >= this.duration ? 0 : this.currentTime) : clamp(position, this.duration);
        this._stop();
        this.offset = offset;
        if (offset < this.duration) this._start();
      }
      this.changed();
    }
    pause() {
      ++this.intent;
      this.offset = this.currentTime;
      this._stop();
      this.changed();
    }
    _gains() {
      const now = this.context.currentTime;
      for (const {source, gain} of this.nodes) {
        gain.gain.cancelScheduledValues(now);
        gain.gain.setValueAtTime(gain.gain.value, now);
        gain.gain.linearRampToValueAtTime(source === this.active ? 1 : 0, now + .005);
      }
    }
    _start() {
      const generation = ++this.playGeneration;
      this.startedAt = this.context.currentTime + .006;
      this.playing = true;
      for (const source of SOURCES) {
        if (!this.tracks[source]?.buffer) continue;
        const node = this.context.createBufferSource();
        const gain = this.context.createGain();
        node.buffer = this.tracks[source].buffer;
        gain.gain.setValueAtTime(0, this.startedAt);
        gain.gain.linearRampToValueAtTime(source === this.active ? 1 : 0, this.startedAt + .005);
        node.connect(gain);
        gain.connect(this.context.destination);
        node.onended = () => {
          node.disconnect(); gain.disconnect();
          if (generation !== this.playGeneration) return;
          ++this.playGeneration;
          this.playing = false;
          this.offset = this.duration;
          this.nodes = [];
          this.changed();
        };
        node.start(this.startedAt, this.offset, this.duration - this.offset);
        this.nodes.push({source, node, gain});
      }
    }
    _stop() {
      ++this.playGeneration;
      for (const {node, gain} of this.nodes) {
        const now = this.context.currentTime;
        gain.gain.cancelScheduledValues(now);
        gain.gain.setValueAtTime(gain.gain.value, now);
        gain.gain.linearRampToValueAtTime(0, now + .005);
        node.stop(now + .006);
      }
      this.nodes = [];
      this.playing = false;
    }
  }

  class ComparisonPlayer {
    constructor(root, {unauthorized, error = () => {}} = {}) {
      this.root = root;
      this.error = error;
      this.rows = Object.fromEntries(SOURCES.map((source) => [source, root.querySelector(`[data-track="${source}"]`)]));
      this.toggle = root.querySelector('.comparison-toggle');
      this.time = root.querySelector('.comparison-time');
      this.retry = root.querySelector('.comparison-retry');
      this.engine = new ComparisonAudio({changed: () => this.render(), unauthorized});
      this.toggle.addEventListener('click', () => this.togglePlayback());
      this.retry.addEventListener('click', () => { const {key} = this.engine; this.engine.key = null; this.load(key, this.urls); });
      for (const [source, row] of Object.entries(this.rows)) {
        const wave = row.querySelector('.comparison-wave');
        const label = row.querySelector('.comparison-label');
        label.addEventListener('click', () => this.play(source));
        label.addEventListener('keydown', (event) => {
          if (event.code !== 'Space' || event.ctrlKey || event.metaKey || event.altKey) return;
          event.preventDefault(); event.stopPropagation();
          if (!event.repeat) this.togglePlayback();
        });
        let pointer = null;
        const seek = (event) => {
          const rect = wave.getBoundingClientRect();
          this.play(source, clamp((event.clientX - rect.left) / rect.width, 1) * this.engine.duration);
        };
        wave.addEventListener('pointerdown', (event) => {
          if (event.button !== 0 || !this.available(source)) return;
          event.preventDefault();
          pointer = event.pointerId;
          wave.focus({preventScroll: true});
          wave.setPointerCapture(pointer);
          seek(event);
        });
        wave.addEventListener('pointermove', (event) => { if (pointer === event.pointerId) seek(event); });
        const release = (event) => { if (pointer === event.pointerId) pointer = null; };
        wave.addEventListener('pointerup', release);
        wave.addEventListener('pointercancel', release);
        wave.addEventListener('lostpointercapture', release);
        wave.addEventListener('keydown', (event) => {
          if (!this.available(source) || event.ctrlKey || event.metaKey || event.altKey) return;
          if (event.code === 'Space') { event.preventDefault(); event.stopPropagation(); if (!event.repeat) this.togglePlayback(); }
          else if (event.key === 'Enter') { event.preventDefault(); this.play(source); }
          else if (['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) {
            event.preventDefault();
            const time = this.engine.currentTime;
            const next = event.key === 'Home' ? 0 : event.key === 'End' ? this.engine.duration : time + (event.key === 'ArrowRight' ? 1 : -1) * (event.shiftKey ? 5 : .25);
            this.play(source, clamp(next, this.engine.duration));
          }
        });
      }
      this.resize = () => { this.draw(); cancelAnimationFrame(this.frame); this.tick(); };
      if ('ResizeObserver' in window) new ResizeObserver(this.resize).observe(root);
      else window.addEventListener('resize', this.resize);
      this.render();
    }
    available(source) { return !this.engine.loading && this.engine.tracks[source]?.state === 'ready'; }
    load(key, urls) { this.urls = urls; return this.engine.load(key, urls); }
    pause() { this.engine.pause(); }
    reset() { this.engine.reset(); }
    play(source, time = null) { this.engine.play(source, time).catch(() => this.error('Audio could not play. Click a waveform to try again.')); }
    togglePlayback() { this.engine.playing ? this.pause() : this.play(this.engine.active); }
    render() {
      const engine = this.engine;
      this.root.dataset.playing = String(engine.playing);
      this.root.setAttribute('aria-busy', String(engine.loading));
      for (const [source, row] of Object.entries(this.rows)) {
        const available = this.available(source);
        const selected = available && engine.active === source;
        const track = engine.tracks[source];
        row.classList.toggle('active', selected);
        row.classList.toggle('unavailable', !available);
        row.querySelector('.comparison-label').disabled = !available;
        row.querySelector('.comparison-label').setAttribute('aria-pressed', String(selected));
        row.querySelector('.comparison-track-state').textContent = track?.state === 'error' ? 'Could not load' : engine.loading && track?.state !== 'unavailable' ? 'Loading…' : !available ? (source === 'model' ? 'Train to compare' : 'Unavailable') : selected ? (engine.playing ? 'Playing' : 'Selected') : '';
        const wave = row.querySelector('.comparison-wave');
        wave.setAttribute('aria-disabled', String(!available));
        wave.tabIndex = available ? 0 : -1;
      }
      this.toggle.disabled = !engine.duration || engine.loading;
      this.toggle.setAttribute('aria-label', engine.playing ? 'Pause comparison' : 'Play comparison');
      this.toggle.title = engine.playing ? 'Pause (Space)' : 'Play (Space)';
      this.toggle.innerHTML = engine.playing ? '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8 5v14M16 5v14"/></svg>' : '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="m8 5 11 7-11 7Z"/></svg>';
      this.retry.hidden = engine.loading || !Object.values(engine.tracks).some((track) => track.state === 'error');
      this.draw();
      cancelAnimationFrame(this.frame);
      this.tick();
    }
    draw() {
      const peakScale = Math.max(.04, ...Object.values(this.engine.tracks).flatMap((track) => track.peaks || []));
      for (const [source, row] of Object.entries(this.rows)) {
        const canvas = row.querySelector('canvas');
        const rect = canvas.getBoundingClientRect();
        if (!rect.width || !rect.height) continue;
        const dpr = Math.min(window.devicePixelRatio || 1, 2);
        canvas.width = Math.round(rect.width * dpr); canvas.height = Math.round(rect.height * dpr);
        const ctx = canvas.getContext('2d');
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        const track = this.engine.tracks[source];
        const peaks = track?.peaks || [];
        const count = Math.floor(rect.width / 4);
        ctx.strokeStyle = this.available(source) && this.engine.active === source ? '#eeeae2' : '#626861';
        ctx.lineWidth = 1.6; ctx.lineCap = 'round';
        for (let bar = 0; bar < count; bar++) {
          // The same time and amplitude scales apply to every row, including unequal lengths.
          const ratio = this.engine.duration && track?.buffer ? this.engine.duration / track.buffer.duration : 1;
          const from = Math.floor(bar / count * peaks.length * ratio);
          const to = Math.max(from + 1, Math.floor((bar + 1) / count * peaks.length * ratio));
          let peak = 0;
          for (let i = from; i < to; i++) peak = Math.max(peak, peaks[i] || 0);
          const height = Math.max(.7, peak / peakScale * rect.height * .42);
          const x = (bar + .5) * rect.width / count;
          ctx.beginPath(); ctx.moveTo(x, rect.height / 2 - height); ctx.lineTo(x, rect.height / 2 + height); ctx.stroke();
        }
      }
    }
    tick() {
      const time = this.engine.currentTime;
      const duration = this.engine.duration;
      this.time.textContent = `${clock(time)} / ${clock(duration)}`;
      for (const row of Object.values(this.rows)) {
        row.querySelector('.comparison-cursor').style.left = `${duration ? time / duration * 100 : 0}%`;
        const wave = row.querySelector('.comparison-wave');
        wave.setAttribute('aria-valuemax', String(duration));
        wave.setAttribute('aria-valuenow', time.toFixed(2));
        wave.setAttribute('aria-valuetext', `${clock(time)} of ${clock(duration)}`);
      }
      if (this.engine.playing) this.frame = requestAnimationFrame(() => this.tick());
    }
  }
  window.NAMComparisonAudio = ComparisonAudio;
  window.NAMComparisonPlayer = ComparisonPlayer;
})();
