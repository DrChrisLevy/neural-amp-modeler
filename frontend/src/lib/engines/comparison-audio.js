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
    constructor({changed = () => {}, context, fetcher = (...args) => fetch(...args)} = {}) {
      this.changed = changed;
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

  window.NAMComparisonAudio = ComparisonAudio;
})();
