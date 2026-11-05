'use strict';

(() => {
  const $ = (id) => document.getElementById(id);
  const audio = $('audio-player');
  const canvas = $('waveform');
  const ctx = canvas.getContext('2d');
  const icons = {
    connect: '<svg viewBox="0 0 24 24"><path d="M5 12h14m-6-6 6 6-6 6"/></svg>',
    record: '<svg class="solid-icon" viewBox="0 0 24 24"><circle cx="12" cy="12" r="5"/></svg>',
    stop: '<svg class="solid-icon" viewBox="0 0 24 24"><rect x="6" y="6" width="12" height="12" rx="1"/></svg>',
    train: '<svg viewBox="0 0 24 24"><path d="m12 3 2.2 6.8L21 12l-6.8 2.2L12 21l-2.2-6.8L3 12l6.8-2.2L12 3Z"/></svg>',
    play: '<svg class="solid-icon play-icon" viewBox="0 0 24 24"><path d="m7 4 13 8-13 8V4Z"/></svg>',
    pause: '<svg class="solid-icon" viewBox="0 0 24 24"><rect x="6" y="5" width="4" height="14" rx=".7"/><rect x="14" y="5" width="4" height="14" rx=".7"/></svg>',
  };
  let config = { auth_required: false, authenticated: true, training_available: false };
  let state = { captures: [], jobs: [], bridge: { connected: false, recording: false, samples: 0, sample_rate: 48000 } };
  let catalogRevision = null;
  let mode = ['record', 'train', 'compare'].includes(safeSessionRead('tone-twin.mode')) ? safeSessionRead('tone-twin.mode') : 'record';
  let selectedId = safeSessionRead('tone-twin.take');
  let selectedJobId = null;
  let makingNewTake = !selectedId;
  let comparisonSource = 'reference';
  let busy = false;
  let offline = false;
  let polling = false;
  let initialized = false;
  let socket = null;
  let socketTimer = null;
  let reconnectTimer = null;
  let pendingSeek = null;
  let audioGeneration = 0;
  let loadedAudioKey = '';
  let livePeaks = [];
  let lastLiveAt = 0;
  let lastRenderKey = '';
  let animationFrame = null;
  let waveformSize = { width: 0, height: 0, dpr: 1 };
  let decodedPeaks = new Map();
  let pendingPeaks = new Set();
  let audioContext = null;
  let previousJobStatuses = new Map();
  let logarithmicLoss = true;
  const trainingChart = new window.ToneTwinLossChart('train-loss', '#f56855');
  const validationChart = new window.ToneTwinLossChart('validation-loss', '#dedbd1');

  function safeSessionRead(key) { try { return sessionStorage.getItem(key); } catch { return null; } }
  function safeSessionWrite(key, value) { try { value ? sessionStorage.setItem(key, value) : sessionStorage.removeItem(key); } catch { /* optional persistence */ } }
  function capture() { return state.captures.find((item) => item.id === selectedId) || null; }
  function job() {
    if (mode === 'train' && selectedJobId === '__new__') return null;
    const jobs = state.jobs.filter((item) => item.capture_id === selectedId);
    return jobs.find((item) => item.id === selectedJobId) || jobs.find((item) => ['queued', 'training'].includes(item.status)) || jobs.find((item) => item.status === 'completed') || jobs[0] || null;
  }
  function activeJob() { return state.jobs.find((item) => ['queued', 'training'].includes(item.status)); }
  function formatTime(seconds, padded = true) {
    const value = Number.isFinite(seconds) ? Math.max(0, Math.floor(seconds)) : 0;
    const minutes = Math.floor(value / 60);
    return `${padded ? String(minutes).padStart(2, '0') : minutes}:${String(value % 60).padStart(2, '0')}`;
  }
  function duration() { return state.bridge.recording ? Number(state.bridge.samples || 0) / Number(state.bridge.sample_rate || 48000) : Number(capture()?.duration || 0); }
  function progress(item) { return Math.round(Math.max(0, Math.min(1, Number(item?.progress || 0))) * 100); }
  function escapeHtml(value) { return String(value ?? '').replace(/[&<>"']/g, (character) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[character])); }
  function showError(id, message) { $(id).textContent = message; $(id).hidden = !message; }
  function status(message, kind = 'error') { $('inline-status-message').textContent = message; $('inline-status').className = `inline-status ${kind}`; $('inline-status').hidden = !message; }
  function clearStatus() { status(''); }
  function errorMessage(payload, fallback) {
    if (typeof payload?.detail === 'string') return payload.detail;
    if (Array.isArray(payload?.detail)) return payload.detail.map((item) => item.msg || 'Check the input.').join(' ');
    return payload?.message || fallback;
  }
  async function api(path, options = {}) {
    const headers = { ...options.headers };
    if (options.body && !(options.body instanceof FormData)) headers['Content-Type'] = 'application/json';
    let response;
    try { response = await fetch(path, { ...options, headers, credentials: 'same-origin', cache: 'no-store' }); }
    catch { throw new Error('Could not reach your studio. Check your connection and try again.'); }
    let payload;
    try { payload = await response.json(); } catch { payload = null; }
    if (!response.ok) {
      const error = new Error(errorMessage(payload, `The studio could not complete that request (${response.status}).`));
      error.status = response.status;
      if (response.status === 401 && path !== '/api/auth') lockWorkspace();
      throw error;
    }
    return payload;
  }
  function openDialog(id) {
    if (id !== 'auth-dialog' && $('auth-dialog').open) return;
    document.querySelectorAll('dialog[open]').forEach((dialog) => { if (dialog.id !== id) dialog.close(); });
    if (!$(id).open) $(id).showModal();
    if (id === 'takes-dialog') renderTakes();
    if (id === 'settings-dialog') updateSource();
  }
  function lockWorkspace() {
    config.authenticated = false;
    if (socket) socket.close();
    clearTimeout(reconnectTimer);
    openDialog('auth-dialog');
  }
  function updateSource() {
    const connected = Boolean(state.bridge.connected);
    const device = connected ? (state.bridge.device || 'Audio interface') : 'Connect source';
    $('source-name').textContent = device;
    $('source-button').setAttribute('aria-label', connected ? `Audio source: ${device}. Open connection settings.` : 'Connect an audio source');
    $('source-dot').className = `status-dot${connected ? ' connected' : ''}`;
    $('settings-source-dot').className = `status-dot${connected ? ' connected' : ''}`;
    $('settings-source-label').textContent = connected ? `${device} is connected` : 'Waiting for your interface';
    $('bridge-command').textContent = `uv run${config.auth_required ? ' --env-file .env' : ''} tone-twin bridge --url ${location.origin}`;
    $('bridge-auth-note').hidden = !config.auth_required;
    $('bridge-auth-note').innerHTML = 'The project’s <code>.env</code> file supplies your private workspace key. Keep it on your computer.';
    $('local-training-note').hidden = config.training_available;
    $('app-version').textContent = `Tone Twin${config.version ? ` · ${config.version}` : ''}`;
    const synthetic = state.bridge.source === 'demo' || capture()?.source === 'demo';
    $('footer-source').textContent = synthetic ? 'Synthetic demo · 48 kHz' : 'Dry + processed · 48 kHz';
  }
  function selectTake(id, nextMode = null) {
    if (state.bridge.recording) { status('Finish your recording before choosing another take.'); return; }
    selectedId = id;
    selectedJobId = null;
    makingNewTake = false;
    safeSessionWrite('tone-twin.take', id);
    comparisonSource = 'reference';
    pauseAudio();
    loadedAudioKey = '';
    clearStatus();
    if (nextMode) mode = nextMode;
    render();
    if (mode === 'compare') syncAudioSource();
  }
  function setMode(next) {
    if (!['record', 'train', 'compare'].includes(next)) return;
    if (state.bridge.recording && next !== 'record') { status('Finish this take before moving on.'); return; }
    if (next !== 'compare') pauseAudio();
    mode = next;
    clearStatus();
    render();
    if (next === 'compare') syncAudioSource();
  }
  function newTake() {
    pauseAudio();
    selectedId = null;
    selectedJobId = null;
    makingNewTake = true;
    safeSessionWrite('tone-twin.take', null);
    $('take-name').value = 'Untitled take';
    livePeaks = [];
    mode = 'record';
    clearStatus();
    render();
  }
  function primary(label, icon, disabled = false, spinner = false) {
    $('primary-label').textContent = label;
    $('primary-action').disabled = disabled || busy;
    $('transport-icon').innerHTML = icons[icon] || icons.connect;
    $('transport-icon').className = `transport-icon${spinner || busy ? ' busy' : ''}`;
  }
  function secondary(label, handler) {
    $('secondary-action').hidden = !label;
    $('secondary-action').textContent = label || '';
    $('secondary-action').onclick = handler || null;
  }
  function setTitle(text, editable = false) {
    $('take-name').hidden = !editable;
    $('stage-title').hidden = editable;
    $('stage-title').textContent = text;
  }
  function validTrainingTake(item) {
    if (!item) return 'Choose or record a take first.';
    if (item.status !== 'ready') return 'This recording was interrupted. Record a fresh take.';
    if (item.clipped) return 'This take is clipping. Lower the input level and record again.';
    if (item.duration < 12) return 'Record at least 12 seconds. A few varied minutes is even better.';
    return null;
  }
  function render() {
    const take = capture();
    const currentJob = job();
    const recording = state.bridge.recording;
    safeSessionWrite('tone-twin.mode', mode);
    document.body.classList.toggle('recording', Boolean(recording));
    const showLearning = mode === 'train' && Boolean(currentJob);
    document.body.classList.toggle('training-view', showLearning);
    $('learning-panel').hidden = !showLearning;
    canvas.parentElement.hidden = showLearning;
    document.querySelectorAll('[data-mode]').forEach((button) => {
      const active = button.dataset.mode === mode;
      button.classList.toggle('active', active);
      if (active) button.setAttribute('aria-current', 'page'); else button.removeAttribute('aria-current');
    });
    $('recording-dot').hidden = !recording;
    $('stage-value').className = 'stage-value';
    $('compare-switch').hidden = mode !== 'compare';
    $('download-model').hidden = true;
    $('waveform-message').hidden = true;
    $('waveform-seek').hidden = mode !== 'compare' || !take;
    $('takes-count').textContent = state.captures.length ? `(${state.captures.length})` : '';
    updateSource();
    if (mode === 'record') {
      setTitle(recording ? $('take-name').value || 'Untitled take' : take?.name || 'Untitled take', !take);
      $('take-name').disabled = Boolean(recording || busy);
      $('stage-value').textContent = formatTime(duration());
      if (recording) {
        $('stage-detail').textContent = 'Capturing dry + processed. Keep playing.';
        primary('Stop recording', 'stop');
        secondary('', null);
      } else if (take) {
        $('stage-detail').textContent = take.source === 'demo' ? 'Synthetic plucks · for exploring the workflow.' : take.status === 'interrupted' ? 'This take was interrupted. Record a fresh one.' : 'Take saved. Every detail, together.';
        primary(currentJob?.status === 'completed' ? 'Compare your tone' : 'Train this take', currentJob?.status === 'completed' ? 'play' : 'train');
        secondary('Record another', newTake);
      } else {
        $('stage-detail').textContent = state.bridge.connected ? 'Dry + processed, together. Make it yours.' : 'A clean input. Your processed tone. One take.';
        primary(state.bridge.connected ? 'Start recording' : 'Connect source', state.bridge.connected ? 'record' : 'connect');
        secondary('', null);
        if (!state.bridge.connected || !livePeaks.length) { $('waveform-message').textContent = 'Ready when you are.'; $('waveform-message').hidden = false; }
      }
    } else if (mode === 'train') {
      const running = currentJob && ['queued', 'training'].includes(currentJob.status);
      setTitle(take?.name || 'A little learning.');
      if (running) {
        $('stage-value').classList.add('percent');
        $('stage-value').textContent = currentJob.status === 'queued' ? '—' : `${progress(currentJob)}%`;
        const stages = { preparing: 'Checking your audio and lining things up.', training: `Epoch ${currentJob.epoch || 0} of ${currentJob.epochs}`, exporting: 'Saving your model and rendering the comparison.' };
        $('stage-detail').textContent = currentJob.status === 'queued' ? 'Waiting for a training worker.' : stages[currentJob.stage] || `Epoch ${currentJob.epoch || 0} of ${currentJob.epochs}`;
        primary(currentJob.status === 'queued' ? 'Getting ready' : 'Learning your tone', 'train', true, true);
        secondary('You can leave this page. Training keeps going.', null);
      } else if (currentJob?.status === 'completed') {
        $('stage-value').classList.add('word');
        $('stage-value').textContent = 'Ready.';
        $('stage-detail').textContent = 'Your model is ready to meet the original.';
        primary('Hear the difference', 'play');
        secondary('Train a new version', () => { selectedJobId = '__new__'; render(); });
      } else {
        renderNewTraining();
        if (currentJob?.status === 'failed' && $('inline-status').hidden) status(currentJob.error || 'Training could not finish. Try a fresh take or a new run.');
      }
    } else {
      setTitle(take?.name || 'Hear the difference.');
      $('stage-value').textContent = formatTime(audio.currentTime);
      const completed = currentJob?.status === 'completed' && currentJob.prediction_url;
      $('stage-detail').textContent = !take ? 'Choose a take to start listening.' : take.source === 'demo' ? (completed ? 'Synthetic demo · comparing held-out audio.' : 'Synthetic plucks · this is not a guitar recording.') : completed ? 'Same performance. Original tone, learned tone.' : 'Listen to the recorded pair. Train to add your model.';
      primary(audio.paused ? 'Listen' : 'Pause', audio.paused ? 'play' : 'pause', !take);
      secondary(take ? `${formatTime(Number.isFinite(audio.duration) ? audio.duration : take.duration, false)}${completed ? ' · held-out comparison' : ' · recorded pair'}` : 'Choose a take', take ? null : () => openDialog('takes-dialog'));
      if (completed && currentJob.model_url) { $('download-model').hidden = false; $('download-model').href = currentJob.model_url; }
      document.querySelectorAll('[data-source]').forEach((button) => {
        button.disabled = !take || (button.dataset.source === 'model' && !completed);
        button.classList.toggle('active', button.dataset.source === comparisonSource);
        button.setAttribute('aria-pressed', String(button.dataset.source === comparisonSource));
      });
      if (!take) { $('waveform-message').textContent = 'Your sound will live here.'; $('waveform-message').hidden = false; }
    }
    if (offline) {
      $('stage-detail').textContent = 'The studio is offline. Reconnect to continue.';
      primary('Reconnect studio', 'connect');
    }
    if (showLearning) renderLearning(currentJob);
    drawWaveform();
    if ($('takes-dialog').open) renderTakes();
  }
  function renderLearning(item) {
    const running = ['queued', 'training'].includes(item.status);
    const training = Array.isArray(item.training_metrics) ? item.training_metrics : [];
    const validation = Array.isArray(item.metrics) ? item.metrics : [];
    const oldRun = !item.metric_schema;
    const stateLabel = offline ? 'Connection lost' : item.status === 'queued' ? 'Queued' : item.stage === 'preparing' ? 'Preparing' : item.stage === 'exporting' ? 'Exporting' : item.status === 'training' ? 'Live' : item.status === 'completed' ? 'Complete' : 'Stopped';
    $('learning-state').textContent = stateLabel;
    $('learning-state').classList.toggle('is-live', item.status === 'training' && !offline);
    $('learning-progress-fill').style.width = `${progress(item)}%`;
    $('learning-progress').setAttribute('aria-valuenow', String(progress(item)));
    $('learning-model').textContent = `NAM · ${item.model_size === 'both' ? 'Lite + Full' : item.model_size === 'full' ? 'Full' : 'Lite'}`;
    $('learning-epoch').textContent = `Epoch ${item.epoch || 0} / ${item.epochs}`;
    $('learning-elapsed').textContent = Number.isFinite(item.elapsed_seconds) ? `${formatTime(item.elapsed_seconds)} training` : '';
    trainingChart.update(training, item.epochs, logarithmicLoss, oldRun ? 'Training loss wasn’t logged for this older run.' : running ? 'Waiting for the first batch.' : 'No training measurements recorded.');
    validationChart.update(validation, item.epochs, logarithmicLoss, running ? 'Measured after the first epoch.' : 'No validation measurements recorded.');
    $('learning-best').textContent = Number.isFinite(item.best_loss) ? `Best validation ${window.ToneTwinLossChart.format(item.best_loss)} · epoch ${item.best_epoch}` : 'Lower is better · separate scales';
    $('learning-note').textContent = oldRun ? 'Saved validation history. New runs also log training loss live.' : running ? 'Training updates about every 2 seconds; validation after each epoch. Lower is better · separate scales.' : 'Measured training batches and validation epochs. Lower is better · separate scales.';
  }
  function renderNewTraining() {
    const take = capture();
    const issue = validTrainingTake(take);
    setTitle(take?.name || 'A little learning.');
    $('stage-value').className = 'stage-value word';
    $('stage-value').textContent = take ? 'Your turn.' : 'First, play.';
    $('stage-detail').textContent = !config.training_available ? 'Training happens in your deployed Modal workspace.' : issue || 'Teach a small network the shape of your sound.';
    const other = activeJob();
    const unavailable = Boolean(take && (!config.training_available || issue || (other && other.capture_id !== selectedId)));
    if (other && other.capture_id !== selectedId) $('stage-detail').textContent = 'Another model is training. You can follow its take below.';
    primary(take ? 'Train a model' : 'Choose a take', 'train', unavailable);
    secondary(`${$('model-size').selectedOptions[0].text.split(' — ')[0]} model · ${$('epochs').value} epochs`, () => { openDialog('settings-dialog'); $('training-settings').open = true; $('training-settings').scrollIntoView({ block: 'nearest' }); });
    if (!take) { $('waveform-message').textContent = 'Start with one sound you love.'; $('waveform-message').hidden = false; }
  }
  function renderTakes() {
    const target = $('takes-list');
    if (!state.captures.length) {
      target.innerHTML = '<div class="take-empty"><strong>Nothing on the tape. Yet.</strong>Record your first take, or upload a matched dry / wet pair.</div>';
      return;
    }
    target.innerHTML = state.captures.map((take) => {
      const jobs = state.jobs.filter((item) => item.capture_id === take.id);
      const running = jobs.find((item) => ['queued', 'training'].includes(item.status));
      const complete = jobs.find((item) => item.status === 'completed');
      const detail = take.status === 'interrupted' ? 'Interrupted' : take.clipped ? 'Clipping' : running ? `${running.status === 'queued' ? 'Queued' : `Training ${progress(running)}%`}` : complete ? 'Model ready' : 'Ready to train';
      let date = '';
      try { date = new Date(take.created_at).toLocaleDateString(undefined, { month: 'short', day: 'numeric' }); } catch { /* optional metadata */ }
      return `<button class="take-item${take.id === selectedId ? ' selected' : ''}" data-take="${escapeHtml(take.id)}"><span class="take-item-info"><span class="take-item-title">${escapeHtml(take.name)}</span><span class="take-item-meta"><span>${escapeHtml(date)}</span><span>${escapeHtml(detail)}</span>${take.source === 'demo' ? '<span class="synthetic-label">Synthetic demo</span>' : ''}</span></span><span class="take-item-right"><span class="take-item-duration">${formatTime(take.duration)}</span><span class="take-item-arrow">↗</span></span></button>`;
    }).join('');
    target.querySelectorAll('[data-take]').forEach((button) => button.addEventListener('click', () => {
      const current = state.jobs.find((item) => item.capture_id === button.dataset.take && ['queued', 'training'].includes(item.status));
      selectTake(button.dataset.take, current ? 'train' : 'compare');
      $('takes-dialog').close();
    }));
  }
  async function updateState() {
    if (polling || !config.authenticated) return;
    polling = true;
    try {
      const next = await api(`/api/state${catalogRevision ? `?revision=${encodeURIComponent(catalogRevision)}` : ''}`);
      const wasOffline = offline;
      const wasRecording = state.bridge.recording;
      state = { captures: next.captures || state.captures, jobs: next.jobs || state.jobs, bridge: next.bridge || state.bridge };
      catalogRevision = next.revision || null;
      offline = false;
      if (selectedId && !capture() && !wasRecording) { selectedId = null; makingNewTake = true; }
      if (wasRecording && !state.bridge.recording && state.captures.length) {
        const newest = state.captures[0];
        if (Date.now() - Date.parse(newest.created_at) < 20000) { selectedId = newest.id; makingNewTake = false; safeSessionWrite('tone-twin.take', selectedId); }
      }
      for (const item of state.jobs) {
        if (previousJobStatuses.has(item.id) && previousJobStatuses.get(item.id) !== item.status && item.capture_id === selectedId) {
          if (item.status === 'completed' && mode === 'train') { status('Training complete. Your best model is ready to compare.', 'success'); }
          if (item.status === 'failed') status(item.error || 'Training stopped. Try again.');
        }
      }
      previousJobStatuses = new Map(state.jobs.map((item) => [item.id, item.status]));
      const key = JSON.stringify([mode, selectedId, selectedJobId, state.captures.map((item) => [item.id, item.status]), state.jobs, state.bridge.connected, state.bridge.recording, state.bridge.device]);
      if (key !== lastRenderKey || wasOffline) { lastRenderKey = key; if (wasOffline) clearStatus(); render(); if (mode === 'compare') syncAudioSource(); }
      else if (mode === 'record' && state.bridge.recording) $('stage-value').textContent = formatTime(duration());
    } catch (error) {
      if (error.status !== 401) { offline = true; status(error.message); render(); }
    } finally { polling = false; }
  }
  function connectMonitor() {
    if (!config.authenticated || (socket && [WebSocket.OPEN, WebSocket.CONNECTING].includes(socket.readyState))) return;
    clearTimeout(reconnectTimer);
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    socket = new WebSocket(`${protocol}//${location.host}/ws/monitor`);
    socket.addEventListener('open', () => {
      clearInterval(socketTimer);
      socketTimer = setInterval(() => { if (socket?.readyState === WebSocket.OPEN) socket.send('ping'); }, 25000);
    });
    socket.addEventListener('message', (event) => {
      let message;
      try { message = JSON.parse(event.data); } catch { return; }
      if (message.type === 'state' && message.bridge) {
        const stopped = state.bridge.recording && !message.bridge.recording;
        const stoppedCaptureId = state.bridge.capture_id;
        state.bridge = { ...state.bridge, ...message.bridge };
        if (!state.bridge.connected) livePeaks = [];
        render();
        if (stopped) updateState().then(() => {
          if (stoppedCaptureId && state.captures.some((item) => item.id === stoppedCaptureId)) selectTake(stoppedCaptureId, 'record');
        });
      }
      if (message.type === 'levels') {
        state.bridge.levels = { dry: Number(message.dry || 0), wet: Number(message.wet || 0) };
        if (Number.isFinite(message.samples)) state.bridge.samples = message.samples;
        const values = Array.isArray(message.peaks) ? message.peaks : [];
        const stride = Math.max(1, Math.ceil(values.length / 4));
        for (let start = 0; start < values.length; start += stride) livePeaks.push(Math.max(0, ...values.slice(start, start + stride).filter(Number.isFinite)));
        livePeaks = livePeaks.slice(-320);
        lastLiveAt = performance.now();
        if (mode === 'record') {
          if (state.bridge.recording) $('stage-value').textContent = formatTime(duration());
          if (!capture()) $('waveform-message').hidden = Boolean(livePeaks.length);
          drawWaveform();
        }
      }
    });
    socket.addEventListener('close', () => {
      clearInterval(socketTimer);
      socket = null;
      if (config.authenticated) reconnectTimer = setTimeout(connectMonitor, 3000);
    });
    socket.addEventListener('error', () => { /* Polling reports connection errors; socket retries quietly. */ });
  }
  async function initialize() {
    try {
      config = await api('/api/config');
      offline = false;
      updateSource();
      if (!config.authenticated) { lockWorkspace(); return; }
      initialized = true;
      await updateState();
      connectMonitor();
      render();
      if (mode === 'compare') syncAudioSource();
    } catch (error) { offline = true; status(error.message); render(); }
  }
  async function runAction(action, { errorTarget = null, button = null } = {}) {
    if (busy) return;
    busy = true;
    if (button) { button.disabled = true; button.classList.add('loading'); }
    if (errorTarget) showError(errorTarget, ''); else clearStatus();
    render();
    try { await action(); }
    catch (error) { if (error.status !== 401 || errorTarget === 'auth-error') { if (errorTarget) showError(errorTarget, error.message); else status(error.message); } }
    finally { busy = false; if (button) { button.disabled = false; button.classList.remove('loading'); } render(); }
  }
  async function record() {
    if (!state.bridge.connected && !state.bridge.recording) { openDialog('settings-dialog'); return; }
    await runAction(async () => {
      if (state.bridge.recording) {
        const result = await api('/api/captures/record', { method: 'POST', body: JSON.stringify({ action: 'stop' }) });
        state.bridge.recording = false;
        await updateState();
        if (result.capture) {
          if (!state.captures.some((item) => item.id === result.capture.id)) state.captures.unshift(result.capture);
          selectTake(result.capture.id, 'record');
          if (result.capture.duration < 12) status('Take saved. Record at least 12 seconds when you’re ready to train.', 'neutral');
        } else { newTake(); status('That take was under a second. Give it another go.', 'neutral'); }
      } else {
        pauseAudio();
        mode = 'record';
        selectedId = null;
        makingNewTake = true;
        livePeaks = [];
        await api('/api/captures/record', { method: 'POST', body: JSON.stringify({ action: 'start', name: $('take-name').value.trim() || 'Untitled take' }) });
        state.bridge.recording = true;
        state.bridge.samples = 0;
        await updateState();
      }
    });
  }
  async function train() {
    const take = capture();
    if (!take) { openDialog('takes-dialog'); return; }
    const issue = validTrainingTake(take);
    if (issue) { status(issue); return; }
    if (!config.training_available) { status('Open your deployed Modal workspace to train this model.', 'neutral'); return; }
    await runAction(async () => {
      const result = await api('/api/train', { method: 'POST', body: JSON.stringify({ capture_id: take.id, epochs: Number($('epochs').value), model_size: $('model-size').value }) });
      state.jobs.unshift(result);
      selectedJobId = result.id;
      previousJobStatuses.set(result.id, result.status);
      mode = 'train';
      await updateState();
    });
  }
  function comparison() {
    const take = capture();
    const item = job();
    const completed = item?.status === 'completed' && item.prediction_url;
    if (!take) return null;
    if (comparisonSource === 'model' && !completed) comparisonSource = 'reference';
    const url = completed ? ({ dry: item.dry_url, reference: item.reference_url, model: item.prediction_url })[comparisonSource] : ({ dry: take.dry_url, reference: take.wet_url })[comparisonSource];
    return { url, key: `${take.id}:${completed ? item.id : 'capture'}:${comparisonSource}`, take, job: completed ? item : null };
  }
  function pauseAudio() { audio.pause(); }
  function syncAudioSource(preserve = false) {
    if (mode !== 'compare') return;
    const current = comparison();
    if (!current?.url || current.key === loadedAudioKey) return;
    const position = preserve ? audio.currentTime : 0;
    const playing = preserve && !audio.paused;
    audio.pause();
    const generation = ++audioGeneration;
    loadedAudioKey = current.key;
    pendingSeek = { position, playing, generation };
    audio.src = current.url;
    audio.load();
    const knownPeaks = !current.job && current.take.peaks?.[comparisonSource === 'dry' ? 'dry' : 'wet'];
    if (knownPeaks) decodedPeaks.set(current.url, knownPeaks);
    else loadPeaks(current.url);
    render();
  }
  async function togglePlayback() {
    if (!capture()) { openDialog('takes-dialog'); return; }
    syncAudioSource();
    if (audio.paused) {
      try { await audio.play(); } catch (error) { if (error.name !== 'AbortError') status('Audio could not play. Try again, or check that your workspace is unlocked.'); }
    } else audio.pause();
    render();
  }
  async function loadPeaks(url) {
    if (!url || decodedPeaks.has(url) || pendingPeaks.has(url)) return;
    pendingPeaks.add(url);
    try {
      const response = await fetch(url, { credentials: 'same-origin' });
      if (response.status === 401) { lockWorkspace(); return; }
      if (!response.ok) return;
      const bytes = await response.arrayBuffer();
      const AudioContextClass = window.AudioContext || window.webkitAudioContext;
      if (!AudioContextClass) return;
      audioContext ||= new AudioContextClass();
      const buffer = await audioContext.decodeAudioData(bytes);
      const signal = buffer.getChannelData(0);
      const bins = 400;
      const peaks = [];
      for (let bin = 0; bin < bins; bin++) {
        const from = Math.floor(bin * signal.length / bins);
        const to = Math.floor((bin + 1) * signal.length / bins);
        let peak = 0;
        for (let index = from; index < to; index++) peak = Math.max(peak, Math.abs(signal[index]));
        peaks.push(peak);
      }
      decodedPeaks.set(url, peaks);
      if (decodedPeaks.size > 10) decodedPeaks.delete(decodedPeaks.keys().next().value);
      drawWaveform();
    } catch { /* Audio remains playable if waveform decoding is unavailable. */ }
    finally { pendingPeaks.delete(url); }
  }
  function resizeCanvas() {
    const rect = canvas.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    waveformSize = { width: rect.width, height: rect.height, dpr };
    canvas.width = Math.round(rect.width * dpr);
    canvas.height = Math.round(rect.height * dpr);
    drawWaveform();
  }
  function drawWaveform() {
    const { width, height, dpr } = waveformSize;
    if (!width || !height) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);
    const take = capture();
    const currentJob = job();
    let peaks = [];
    let cursor = null;
    let isLive = false;
    if (mode === 'record' && (state.bridge.recording || !take) && state.bridge.connected) {
      isLive = true;
      peaks = livePeaks;
      if (performance.now() - lastLiveAt > 3500 && !state.bridge.recording) peaks = [];
      cursor = state.bridge.recording && peaks.length ? 0.74 : null;
    } else if (mode === 'compare' && take) {
      const current = comparison();
      peaks = decodedPeaks.get(current?.url) || (current?.job ? [] : take.peaks?.[comparisonSource === 'dry' ? 'dry' : 'wet']) || [];
      if (Number.isFinite(audio.duration) && audio.duration > 0) cursor = audio.currentTime / audio.duration;
    } else if (take) {
      peaks = take.peaks?.wet || take.peaks?.dry || [];
      if (mode === 'train' && ['queued', 'training'].includes(currentJob?.status)) cursor = Number(currentJob.progress || 0);
    }
    const count = Math.max(20, Math.floor(width / (width < 600 ? 5 : 6.8)));
    const spacing = width / count;
    const max = Math.max(0.04, ...peaks);
    const center = height / 2;
    ctx.lineWidth = width < 600 ? 1.8 : 2.1;
    ctx.lineCap = 'round';
    for (let bar = 0; bar < count; bar++) {
      const from = Math.floor(bar * peaks.length / count);
      const to = Math.max(from + 1, Math.floor((bar + 1) * peaks.length / count));
      let value = 0;
      for (let index = from; index < to; index++) value = Math.max(value, peaks[index] || 0);
      const amplitude = Math.max(peaks.length ? 1 : 0.65, Math.min(1, value / max) * height * 0.37);
      const x = spacing * bar + spacing / 2;
      const past = mode !== 'compare' || cursor === null || bar / count <= cursor;
      ctx.strokeStyle = !peaks.length ? '#444741' : mode === 'train' ? '#84867c' : past ? '#eeeae2' : '#64675d';
      ctx.beginPath(); ctx.moveTo(x, center - amplitude); ctx.lineTo(x, center + amplitude); ctx.stroke();
    }
    if (cursor !== null && peaks.length) {
      const x = Math.max(2, Math.min(width - 2, cursor * width));
      ctx.strokeStyle = '#f56855'; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(x, center - height * .43); ctx.lineTo(x, center + height * .43); ctx.stroke();
      ctx.fillStyle = '#f56855'; ctx.beginPath(); ctx.arc(x, center - height * .43, 3.5, 0, Math.PI * 2); ctx.fill();
    }
    canvas.setAttribute('aria-label', isLive ? 'Live input waveform' : take ? `${comparisonSource === 'dry' ? 'Dry input' : comparisonSource === 'model' ? 'Model output' : 'Processed tone'} waveform` : 'No audio captured yet');
  }
  function animatePlayback() {
    cancelAnimationFrame(animationFrame);
    const step = () => {
      if (mode === 'compare') { $('stage-value').textContent = formatTime(audio.currentTime); drawWaveform(); }
      if (!audio.paused) animationFrame = requestAnimationFrame(step);
    };
    step();
  }

  document.querySelectorAll('[data-mode]').forEach((button) => button.addEventListener('click', () => setMode(button.dataset.mode)));
  $('loss-scale').addEventListener('click', () => {
    logarithmicLoss = !logarithmicLoss;
    $('loss-scale').setAttribute('aria-pressed', String(logarithmicLoss));
    $('loss-scale').textContent = logarithmicLoss ? 'Log scale' : 'Linear scale';
    if (job()) renderLearning(job());
  });
  document.querySelector('.wordmark').addEventListener('click', (event) => { event.preventDefault(); setMode('record'); });
  $('source-button').addEventListener('click', () => openDialog('settings-dialog'));
  $('settings-button').addEventListener('click', () => openDialog('settings-dialog'));
  $('takes-button').addEventListener('click', () => openDialog('takes-dialog'));
  $('dismiss-status').addEventListener('click', clearStatus);
  $('primary-action').addEventListener('click', async () => {
    if (offline) { clearStatus(); await initialize(); return; }
    if (mode === 'record') {
      if (state.bridge.recording) await record();
      else if (capture()) setMode(job()?.status === 'completed' ? 'compare' : 'train');
      else await record();
    } else if (mode === 'train') {
      if (job()?.status === 'completed' && selectedJobId !== '__new__') setMode('compare'); else await train();
    } else await togglePlayback();
  });
  document.querySelectorAll('.close-dialog').forEach((button) => button.addEventListener('click', () => button.closest('dialog').close()));
  document.querySelectorAll('dialog').forEach((dialog) => dialog.addEventListener('click', (event) => { if (event.target !== dialog || dialog.id === 'auth-dialog') return; const rect = dialog.getBoundingClientRect(); if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) dialog.close(); }));
  $('auth-dialog').addEventListener('cancel', (event) => event.preventDefault());
  $('copy-command').addEventListener('click', async () => {
    try { await navigator.clipboard.writeText($('bridge-command').textContent); $('copy-command').innerHTML = '<svg viewBox="0 0 24 24"><path d="m5 12 4 4L19 6"/></svg>'; setTimeout(() => { $('copy-command').innerHTML = '<svg viewBox="0 0 24 24"><rect x="8" y="8" width="11" height="12" rx="2"/><path d="M15 8V4H4v12h4"/></svg>'; }, 2000); }
    catch { showError('settings-error', 'Copy is unavailable in this browser. Select and copy the command above.'); }
  });
  function openUpload() { showError('upload-error', ''); openDialog('upload-dialog'); }
  $('open-upload').addEventListener('click', openUpload);
  document.querySelectorAll('.upload-trigger').forEach((button) => button.addEventListener('click', openUpload));
  ['dry', 'wet'].forEach((channel) => $(channel + '-file').addEventListener('change', () => {
    const file = $(channel + '-file').files[0];
    $(channel + '-file-label').textContent = file?.name || 'Choose WAV ↑';
    $(channel + '-file').closest('.file-picker').classList.toggle('has-file', Boolean(file));
  }));
  $('upload-form').addEventListener('submit', (event) => {
    event.preventDefault();
    const dry = $('dry-file').files[0], wet = $('wet-file').files[0];
    if (!dry || !wet) { showError('upload-error', 'Choose both the dry and processed WAV files.'); return; }
    if (dry.size > 128 * 1024 * 1024 || wet.size > 128 * 1024 * 1024) { showError('upload-error', 'Each WAV must be smaller than 128 MB.'); return; }
    const form = new FormData(); form.set('dry', dry); form.set('wet', wet); form.set('name', $('upload-name').value.trim() || 'Untitled take');
    runAction(async () => {
      const result = await api('/api/captures/upload', { method: 'POST', body: form });
      state.captures.unshift(result);
      selectTake(result.id, 'record');
      $('upload-dialog').close();
      $('upload-form').reset();
      ['dry', 'wet'].forEach((channel) => { $(channel + '-file-label').textContent = 'Choose WAV ↑'; $(channel + '-file').closest('.file-picker').classList.remove('has-file'); });
      await updateState();
    }, { errorTarget: 'upload-error', button: $('upload-submit') });
  });
  $('demo-button').addEventListener('click', () => runAction(async () => {
    const result = await api('/api/demo', { method: 'POST' });
    state.captures.unshift(result);
    selectTake(result.id, 'compare');
    $('settings-dialog').close();
    await updateState();
  }, { errorTarget: 'settings-error', button: $('demo-button') }));
  $('auth-form').addEventListener('submit', async (event) => {
    event.preventDefault();
    const button = $('auth-submit'); button.disabled = true; button.classList.add('loading'); showError('auth-error', '');
    try {
      await api('/api/auth', { method: 'POST', body: JSON.stringify({ key: $('workspace-key').value }) });
      $('workspace-key').value = '';
      config.authenticated = true;
      $('auth-dialog').close();
      clearStatus();
      await initialize();
    } catch (error) { showError('auth-error', error.message); }
    finally { button.disabled = false; button.classList.remove('loading'); }
  });
  $('model-size').addEventListener('change', () => { if (mode === 'train') render(); });
  $('epochs').addEventListener('change', () => { if (mode === 'train') render(); });
  document.querySelectorAll('[data-source]').forEach((button) => button.addEventListener('click', () => { comparisonSource = button.dataset.source; syncAudioSource(true); render(); }));
  $('waveform-seek').addEventListener('click', (event) => { if (!Number.isFinite(audio.duration) || !audio.duration) return; const rect = canvas.getBoundingClientRect(); audio.currentTime = Math.max(0, Math.min(audio.duration, (event.clientX - rect.left) / rect.width * audio.duration)); $('stage-value').textContent = formatTime(audio.currentTime); drawWaveform(); });
  audio.addEventListener('loadedmetadata', () => {
    if (pendingSeek?.generation === audioGeneration) {
      const pending = pendingSeek; pendingSeek = null;
      audio.currentTime = Math.min(pending.position, Number.isFinite(audio.duration) ? Math.max(0, audio.duration - .01) : pending.position);
      if (pending.playing) audio.play().catch((error) => { if (error.name !== 'AbortError') status('Tap Listen to continue playback.', 'neutral'); });
    }
    if (mode === 'compare') render();
  });
  audio.addEventListener('play', () => { if (mode === 'compare') render(); animatePlayback(); });
  audio.addEventListener('pause', () => { cancelAnimationFrame(animationFrame); if (mode === 'compare') render(); });
  audio.addEventListener('ended', () => { audio.currentTime = 0; render(); });
  audio.addEventListener('error', () => { if (audio.src && loadedAudioKey) status('This audio file could not be loaded. Refresh the studio or unlock your workspace again.'); });
  document.addEventListener('keydown', (event) => {
    const target = event.target;
    if (target instanceof HTMLElement && (target.matches('input, textarea, select, button, summary') || target.isContentEditable)) return;
    if (document.querySelector('dialog[open]') || event.ctrlKey || event.metaKey || event.altKey || event.repeat) return;
    if (event.code === 'Space' && mode === 'compare' && capture()) { event.preventDefault(); togglePlayback(); }
    if (event.key.toLowerCase() === 'r' && (state.bridge.connected || state.bridge.recording)) { event.preventDefault(); record(); }
  });
  document.addEventListener('visibilitychange', () => { if (!document.hidden && initialized) { updateState(); connectMonitor(); } });
  window.addEventListener('beforeunload', (event) => { if (state.bridge.recording) { event.preventDefault(); event.returnValue = ''; } });
  if ('ResizeObserver' in window) new ResizeObserver(resizeCanvas).observe(canvas); else window.addEventListener('resize', resizeCanvas);
  resizeCanvas();
  render();
  initialize();
  async function poll() {
    if (initialized && !document.hidden) await updateState();
    setTimeout(poll, activeJob() ? 1000 : 3000);
  }
  setTimeout(poll, 1000);
})();
