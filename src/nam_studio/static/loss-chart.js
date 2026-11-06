'use strict';

// Small SVG plots: only measured points are drawn, with separate metric scales.
window.NAMLossChart = class {
  constructor(prefix, color) {
    this.svg = document.getElementById(`${prefix}-chart`);
    this.empty = document.getElementById(`${prefix}-empty`);
    this.tooltip = document.getElementById(`${prefix}-tooltip`);
    this.value = document.getElementById(`${prefix}-value`);
    this.color = color;
    this.prefix = prefix;
    this.points = [];
    this.selected = null;
    this.svg.addEventListener('pointermove', (event) => {
      if (!this.points.length) return;
      const rect = this.svg.getBoundingClientRect();
      const epoch = (event.clientX - rect.left - this.left) / this.plotWidth * this.epochs;
      this.selected = this.points.reduce((best, point, index) => Math.abs(point.epoch - epoch) < Math.abs(this.points[best].epoch - epoch) ? index : best, 0);
      this.inspect();
    });
    this.svg.addEventListener('pointerleave', () => this.clearSelection());
    this.svg.addEventListener('blur', () => this.clearSelection());
    this.svg.addEventListener('keydown', (event) => {
      if (!['ArrowLeft', 'ArrowRight', 'Home', 'End', 'Escape'].includes(event.key) || !this.points.length) return;
      event.preventDefault();
      if (event.key === 'Escape') { this.clearSelection(); return; }
      const last = this.points.length - 1;
      this.selected = event.key === 'Home' ? 0 : event.key === 'End' ? last : Math.max(0, Math.min(last, (this.selected ?? last) + (event.key === 'ArrowLeft' ? -1 : 1)));
      this.inspect();
    });
    new ResizeObserver(() => this.draw()).observe(this.svg);
  }

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

  update(points, epochs, logarithmic, emptyMessage) {
    this.points = points.filter((point) => Number.isFinite(point.epoch) && Number.isFinite(point.loss) && point.loss >= 0);
    this.epochs = Math.max(1, Number(epochs) || 1, ...this.points.map((point) => point.epoch));
    this.logarithmic = logarithmic;
    this.empty.textContent = emptyMessage;
    this.empty.hidden = this.points.length > 0;
    this.value.textContent = this.constructor.format(this.points.at(-1)?.loss);
    this.clearSelection();
    this.draw();
  }

  draw() {
    const width = this.svg.clientWidth;
    const height = this.svg.clientHeight;
    if (!width || !height) return;
    this.left = 48;
    this.plotWidth = width - this.left - 12;
    const top = 15, bottom = height - 29;
    const positive = this.points.map((p) => p.loss).filter((n) => n > 0);
    const floor = positive.length ? Math.max(Number.MIN_VALUE, Math.min(...positive) / 10) : 1e-6;
    const transform = (n) => this.logarithmic ? Math.log10(Math.max(floor, n)) : n;
    const values = this.points.map((p) => transform(p.loss));
    let min = values.length ? Math.min(...values) : (this.logarithmic ? -3 : 0);
    let max = values.length ? Math.max(...values) : (this.logarithmic ? 0 : 1);
    const pad = Math.max((max - min) * .13, this.logarithmic ? .15 : Math.max(max * .1, 1e-6));
    min = this.logarithmic ? min - pad : Math.max(0, min - pad);
    max += pad;
    this.x = (epoch) => this.left + epoch / this.epochs * this.plotWidth;
    this.y = (loss) => bottom - (transform(loss) - min) / (max - min) * (bottom - top);
    this.svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
    let grid = '';
    for (let i = 0; i < 4; i++) {
      const value = min + (max - min) * i / 3;
      const y = bottom - (bottom - top) * i / 3;
      grid += `<line x1="${this.left}" y1="${y}" x2="${width - 12}" y2="${y}" class="chart-grid"/>`;
      if (this.points.length) grid += `<text x="${this.left - 9}" y="${y + 3}" text-anchor="end" class="chart-axis">${this.constructor.format(this.logarithmic ? 10 ** value : value)}</text>`;
    }
    for (let i = 0; i < 3; i++) {
      const epoch = this.epochs * i / 2;
      grid += `<text x="${this.x(epoch)}" y="${height - 6}" text-anchor="${i === 0 ? 'start' : i === 2 ? 'end' : 'middle'}" class="chart-axis">${Number(epoch.toFixed(1))}${i === 2 ? ' epochs' : ''}</text>`;
    }
    let curve = '';
    if (this.points.length) {
      const path = this.points.map((point, index) => `${index ? 'L' : 'M'}${this.x(point.epoch).toFixed(2)},${this.y(point.loss).toFixed(2)}`).join(' ');
      const first = this.points[0], last = this.points.at(-1);
      curve = `<defs><linearGradient id="${this.prefix}-gradient" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="${this.color}" stop-opacity=".13"/><stop offset="1" stop-color="${this.color}" stop-opacity="0"/></linearGradient></defs><path d="${path} L${this.x(last.epoch)},${bottom} L${this.x(first.epoch)},${bottom} Z" fill="url(#${this.prefix}-gradient)"/><path d="${path}" fill="none" stroke="${this.color}" stroke-width="1.7" stroke-linejoin="round"/><circle cx="${this.x(last.epoch)}" cy="${this.y(last.loss)}" r="3" fill="${this.color}"/>`;
    }
    this.svg.innerHTML = grid + curve + '<g class="chart-crosshair" hidden><line stroke="#b9b7af" stroke-width=".7" stroke-dasharray="3 4"/><circle r="4" fill="' + this.color + '" stroke="#101112" stroke-width="2"/></g>';
    this.svg.setAttribute('aria-label', `${this.prefix === 'train-loss' ? 'Training loss' : 'Validation ESR'}, ${this.logarithmic ? 'logarithmic' : 'linear'} scale. ${this.points.length ? `${this.points.length} measurements. Latest ${this.constructor.format(this.points.at(-1).loss)}. Use arrow keys to inspect.` : 'Waiting for measurements.'}`);
  }

  clearSelection() {
    this.selected = null;
    this.tooltip.hidden = true;
    this.svg.querySelector('.chart-crosshair')?.setAttribute('hidden', '');
  }

  inspect() {
    const point = this.points[this.selected];
    if (!point) return;
    const x = this.x(point.epoch), y = this.y(point.loss);
    const group = this.svg.querySelector('.chart-crosshair');
    if (!group) return;
    group.removeAttribute('hidden');
    const line = group.querySelector('line'), dot = group.querySelector('circle');
    for (const [key, value] of Object.entries({x1: x, x2: x, y1: 12, y2: this.svg.clientHeight - 29})) line.setAttribute(key, value);
    dot.setAttribute('cx', x); dot.setAttribute('cy', y);
    this.tooltip.textContent = `Epoch ${Number(point.epoch.toFixed(2))}${point.partial ? ' so far' : ''} · ${this.constructor.format(point.loss)}${point.step ? ` · step ${point.step}` : ''}`;
    this.tooltip.hidden = false;
    this.tooltip.style.left = `${Math.max(4, Math.min(this.svg.clientWidth - this.tooltip.offsetWidth - 4, x - this.tooltip.offsetWidth / 2))}px`;
    this.tooltip.style.top = `${Math.max(0, y - 34)}px`;
  }
};
