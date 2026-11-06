'use strict';

(() => {
  const $ = (id) => document.getElementById(id);
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
  let mode = 'train';
  let view = 'library';
  let trainingPairId = null;
  let deletingPairId = null;
  let libraryRenderKey = '';
  const expandedPairs = new Set();
  let selectedId = safeSessionRead('neural-amp-modeler.take');
  let selectedJobId = null;
  let makingNewTake = !selectedId;
  let busy = false;
  let offline = false;
  let polling = false;
  let initialized = false;
  let socket = null;
  let socketTimer = null;
  let reconnectTimer = null;
  let livePeaks = [];
  let lastLiveAt = 0;
  let lastRenderKey = '';
  let waveformSize = { width: 0, height: 0, dpr: 1 };
  let previousJobStatuses = new Map();
  let logarithmicLoss = true;
  const trainingChart = new window.NAMLossChart('train-loss', '#f56855');
  const validationChart = new window.NAMLossChart('validation-loss', '#dedbd1');
  const comparisonPlayer = new window.NAMComparisonPlayer($('comparison-player'), {unauthorized: lockWorkspace, error: status});
  const uploads = new window.NAMUploads({
    changed: () => renderLibrary(),
    completed: (take) => {
      state.captures = [take, ...state.captures.filter((item) => item.id !== take.id)];
      updateState();
    },
    unauthorized: lockWorkspace,
  });

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
  function status(message, kind = 'error') { $('library-notice').textContent = message; $('library-notice').hidden = !message; $('inline-status-message').textContent = message; $('inline-status').className = `inline-status ${kind}`; $('inline-status').hidden = !message; }
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
    if (id === 'connection-dialog') updateSource();
  }
  function lockWorkspace() {
    config.authenticated = false;
    comparisonPlayer.reset();
    if (socket) socket.close();
    clearTimeout(reconnectTimer);
    openDialog('auth-dialog');
  }
  function updateSource() {
    const connected = Boolean(state.bridge.connected);
    const device = connected ? (state.bridge.device || 'Audio interface') : 'No live input';
    $('source-name').textContent = device;
    $('source-dot').className = `status-dot${connected ? ' connected' : ''}`;
    $('connection-dot').className = `status-dot${connected ? ' connected' : ''}`;
    $('connection-status').textContent = connected ? `${device} is connected` : 'Waiting for your guitar';
    $('connection-title').textContent = connected ? 'You’re connected.' : 'Connect your guitar.';
    $('connection-intro').textContent = connected ? 'Your live input is ready.' : 'Plug in your Quad Cortex Mini by USB, then run this in Terminal from your Neural Amp Modeler folder.';
    $('connection-instructions').hidden = connected;
    $('connection-record').hidden = !connected || state.bridge.recording;
    $('connection-record').disabled = busy || offline;
    $('bridge-command').textContent = `uv run${config.auth_required ? ' --env-file .env' : ''} neural-amp-modeler bridge --url ${location.origin}`;
    $('bridge-auth-note').hidden = !config.auth_required;
    $('bridge-auth-note').innerHTML = 'The project’s <code>.env</code> file supplies your private workspace key. Keep it on your computer.';
    $('local-training-note').hidden = config.training_available;
    $('source-status').hidden = view === 'library' || mode !== 'record';
  }
  function selectTake(id, nextMode = null) {
    view = 'detail';
    if (state.bridge.recording) { status('Finish your recording before choosing another take.'); return; }
    selectedId = id;
    selectedJobId = null;
    makingNewTake = false;
    safeSessionWrite('neural-amp-modeler.take', id);
    pauseAudio();
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
    view = 'detail';
    document.querySelectorAll('dialog[open]').forEach((dialog) => dialog.close());
    pauseAudio();
    selectedId = null;
    selectedJobId = null;
    makingNewTake = true;
    safeSessionWrite('neural-amp-modeler.take', null);
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
    $('library-view').hidden = view !== 'library';
    $('detail-view').hidden = view === 'library';
    document.body.classList.toggle('library-open', view === 'library');
    renderLibrary();
    const runs = state.jobs.filter((item) => item.capture_id === selectedId);
    $('run-picker').hidden = mode === 'record' || runs.length < 2;
    const options = runs.map((item) => `<option value="${escapeHtml(item.id)}">${item.model_size === 'both' ? 'Full + Lite' : item.model_size === 'full' ? 'Full' : 'Lite'} · ${item.epochs} epochs · ${escapeHtml(item.status)} · ${escapeHtml(new Date(item.created_at).toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' }))}</option>`).join('') + (mode === 'train' ? '<option value="__new__">New training run…</option>' : '');
    if ($('training-run').innerHTML !== options) $('training-run').innerHTML = options;
    $('training-run').value = selectedJobId === '__new__' && mode === 'train' ? '__new__' : currentJob?.id || '';
    const recording = state.bridge.recording;
    safeSessionWrite('neural-amp-modeler.mode', mode);
    document.body.classList.toggle('recording', Boolean(recording));
    const showLearning = mode === 'train' && Boolean(currentJob);
    document.body.classList.toggle('training-view', showLearning && view === 'detail');
    $('learning-panel').hidden = !showLearning;
    canvas.parentElement.hidden = showLearning || mode === 'compare';
    document.body.classList.toggle('comparison-view', mode === 'compare' && view === 'detail');
    $('comparison-player').hidden = mode !== 'compare';
    const finishedTraining = mode === 'train' && currentJob?.status === 'completed';
    $('stage-value').hidden = finishedTraining || mode === 'compare';
    $('stage-detail').hidden = finishedTraining || mode === 'compare';
    $('primary-action').parentElement.hidden = mode === 'compare';
    $('secondary-action').parentElement.hidden = mode === 'compare';
    $('recording-dot').hidden = !recording;
    $('stage-value').className = 'stage-value';
    const downloadable = !recording && mode !== 'record' && currentJob?.status === 'completed' && currentJob.model_url;
    $('download-model').hidden = !downloadable;
    if (downloadable) $('download-model').href = downloadUrl(currentJob.model_url);
    else $('download-model').removeAttribute('href');
    $('waveform-message').hidden = true;
    $('detail-compare').hidden = !take || mode === 'compare' || recording;
    $('detail-training').hidden = !take || !runs.length || mode === 'train' || recording;
    $('detail-train').hidden = !take || recording;
    $('back-library').disabled = Boolean(recording);
    document.body.classList.toggle('live-recording-view', mode === 'record');
    updateSource();
    if (mode === 'record') {
      setTitle(recording ? $('take-name').value || 'Untitled take' : take?.name || 'Untitled take', !take);
      $('take-name').disabled = Boolean(recording || busy);
      $('stage-value').textContent = formatTime(duration());
      if (recording) {
        $('stage-detail').textContent = 'Recording dry input and processed output.';
        primary('Stop recording', 'stop');
        secondary('', null);
      } else if (take) {
        $('stage-detail').textContent = take.source === 'demo' ? 'Synthetic plucks · for exploring the workflow.' : take.status === 'interrupted' ? 'This take was interrupted. Record a fresh one.' : 'Recording saved. Train a model or compare the audio.';
        primary(currentJob?.status === 'completed' ? 'Compare your tone' : 'Train this take', currentJob?.status === 'completed' ? 'play' : 'train');
        secondary('Record another', newTake);
      } else {
        $('stage-detail').textContent = 'Record your dry guitar and processed output at the same time. Use a mix of soft and hard notes and chords.';
        primary('Start recording', 'record');
        secondary('', null);
        if (!state.bridge.connected || !livePeaks.length) { $('waveform-message').textContent = state.bridge.connected ? 'Play to check your input.' : 'Connect your interface to record.'; $('waveform-message').hidden = false; }
      }
    } else if (mode === 'train') {
      const running = currentJob && ['queued', 'training'].includes(currentJob.status);
      setTitle(take?.name || 'Training');
      if (running) {
        $('stage-value').classList.add('percent');
        $('stage-value').textContent = currentJob.status === 'queued' ? '—' : `${progress(currentJob)}%`;
        const stages = { preparing: 'Checking your audio and lining things up.', training: `Epoch ${currentJob.epoch || 0} of ${currentJob.epochs}`, exporting: 'Saving your model and rendering the comparison.' };
        $('stage-detail').textContent = currentJob.status === 'queued' ? 'Waiting for a training worker.' : stages[currentJob.stage] || `Epoch ${currentJob.epoch || 0} of ${currentJob.epochs}`;
        primary(currentJob.status === 'queued' ? 'Getting ready' : 'Learning your tone', 'train', true, true);
        secondary('You can leave this page. Training keeps going.', null);
      } else if (currentJob?.status === 'completed') {
        $('stage-value').textContent = '';
        $('stage-detail').textContent = '';
        primary('Hear the difference', 'play');
        secondary('', null);
      } else {
        renderNewTraining();
        if (currentJob?.status === 'failed' && $('inline-status').hidden) status(currentJob.error || 'Training could not finish. Try a fresh take or a new run.');
      }
    } else {
      setTitle(take?.name || 'Compare audio');
      secondary('', null);
      syncAudioSource();
    }
    if (offline) {
      $('stage-detail').textContent = 'The studio is offline. Reconnect to continue.';
      primary('Reconnect studio', 'connect');
    }
    if (showLearning) renderLearning(currentJob);
    drawWaveform();

  }
  function renderLearning(item) {
    const running = ['queued', 'training'].includes(item.status);
    const training = window.NAMLossChart.trainingSeries(item);
    const validation = Array.isArray(item.metrics) ? item.metrics : [];
    const oldRun = !running && !item.metric_schema;
    const stateLabel = offline ? 'Connection lost' : item.status === 'queued' ? 'Queued' : item.stage === 'preparing' ? 'Preparing' : item.stage === 'exporting' ? 'Exporting' : item.status === 'training' ? 'Live' : item.status === 'completed' ? 'Complete' : 'Stopped';
    $('learning-state').textContent = stateLabel;
    $('learning-state').parentElement.hidden = item.status === 'completed' && !offline;
    $('learning-state').classList.toggle('is-live', item.status === 'training' && !offline);
    $('learning-progress-fill').style.width = `${progress(item)}%`;
    $('learning-progress').setAttribute('aria-valuenow', String(progress(item)));
    $('learning-model').textContent = `NAM · ${item.model_size === 'both' ? 'Full + Lite' : item.model_size === 'full' ? 'Full' : 'Lite'}`;
    $('learning-epoch').textContent = `Epoch ${item.epoch || 0} / ${item.epochs}`;
    $('learning-elapsed').textContent = Number.isFinite(item.elapsed_seconds) ? `${formatTime(item.elapsed_seconds)} training` : '';
    $('train-loss-description').textContent = training.averaged ? 'Waveform + spectral · epoch average' : 'Waveform + spectral · batch loss';
    trainingChart.update(training.points, item.epochs, logarithmicLoss, oldRun ? 'Training loss wasn’t logged for this older run.' : running ? 'Waiting for the first batch.' : 'No training measurements recorded.');
    validationChart.update(validation, item.epochs, logarithmicLoss, running ? 'Measured after the first epoch.' : 'No validation measurements recorded.');
    $('learning-best').textContent = Number.isFinite(item.best_loss) ? `Best validation ${window.NAMLossChart.format(item.best_loss)} · epoch ${item.best_epoch}` : 'Lower is better · separate scales';
    $('learning-note').textContent = training.averaged
      ? (running ? 'Average training loss per epoch; the current epoch updates live. Lower is better · separate scales.' : 'Measured epoch averages and held-out validation error. Lower is better · separate scales.')
      : oldRun ? 'Saved validation history. New runs also log training loss live.' : running ? 'Training updates about every 2 seconds; validation after each epoch. Lower is better · separate scales.' : 'Measured training batches and validation epochs. Lower is better · separate scales.';
    if (item.model_size === 'both') {
      $('train-loss-description').textContent += ' · Full + Lite average';
      $('validation-loss-description').textContent = 'ESR · Full + Lite average';
      $('learning-note').textContent = 'Both models train together and download in one .nam file. Curves show their average; Compare plays Full.';
      $('learning-best').textContent = $('learning-best').textContent.replace('Best validation', 'Best average validation');
    } else { $('validation-loss-description').textContent = 'ESR · audio held out from training'; }
    if (item.status === 'completed' && !item.data_recipe) $('learning-note').textContent += ' Earlier training recipe; scores may differ from new runs.';
  }
  function renderNewTraining() {
    const take = capture();
    const issue = validTrainingTake(take);
    setTitle(take?.name || 'Training');
    $('stage-value').className = 'stage-value word';
    $('stage-value').textContent = take ? 'Ready to train.' : 'Choose a pair.';
    $('stage-detail').textContent = !config.training_available ? 'Training happens in your deployed Modal workspace.' : issue || 'Train a model using this input and output pair.';
    const unavailable = Boolean(take && (!config.training_available || issue));
    primary(take ? 'Train a model' : 'Choose a pair', 'train', unavailable);
    secondary('Choose the number of epochs before starting.', () => openTraining(selectedId));
    if (!take) { $('waveform-message').textContent = 'Upload or record an audio pair.'; $('waveform-message').hidden = false; }
  }
  function showLibrary() {
    if (state.bridge.recording) { status('Finish your recording before returning to the library.'); return; }
    pauseAudio();
    view = 'library';
    clearStatus();
    document.querySelectorAll('dialog[open]').forEach((dialog) => dialog.close());
    render();
  }
  // Version the URL so cached responses cannot retain an older download filename.
  function downloadUrl(url) { return `${url}${url.includes('?') ? '&' : '?'}download=1&v=2`; }
  function modelName(item) { return item.model_size === 'both' ? 'Full + Lite' : item.model_size === 'lite' ? 'Lite' : 'Full'; }
  function runLabel(item) { return item.status === 'training' ? `Training ${progress(item)}%` : item.status === 'completed' ? 'Complete' : item.status === 'failed' ? 'Failed' : 'Queued'; }
  function dateLabel(value, time = false) { return new Date(value).toLocaleString([], { month: 'short', day: 'numeric', ...(time ? { hour: '2-digit', minute: '2-digit' } : {}) }); }
  function openTraining(id) {
    const take = state.captures.find((item) => item.id === id);
    if (!take) { showLibrary(); return; }
    trainingPairId = id;
    $('training-pair-name').textContent = take.name;
    const issue = validTrainingTake(take) || (!config.training_available ? 'Open your deployed studio to train.' : '');
    showError('training-error', issue);
    $('training-submit').disabled = Boolean(issue || busy || offline);
    openDialog('training-dialog');
  }
  function openDelete(id) {
    const take = state.captures.find((item) => item.id === id);
    if (!take) return;
    deletingPairId = id;
    const runs = state.jobs.filter((item) => item.capture_id === id);
    $('delete-description').textContent = `Delete “${take.name}” and its two WAV files${runs.length ? `, plus ${runs.length} training ${runs.length === 1 ? 'run' : 'runs'} with their models` : ''}? This cannot be undone.`;
    const active = runs.some((item) => ['queued', 'training'].includes(item.status));
    $('delete-submit').disabled = active || busy;
    showError('delete-error', active ? 'Wait for this pair’s training run to finish before deleting it.' : '');
    openDialog('delete-dialog');
    $('delete-cancel').focus();
  }
  async function deletePair() {
    const id = deletingPairId;
    if (!id) return;
    await runAction(async () => {
      await api(`/api/captures/${encodeURIComponent(id)}`, {method: 'DELETE'});
      state.captures = state.captures.filter((item) => item.id !== id);
      state.jobs = state.jobs.filter((item) => item.capture_id !== id);
      expandedPairs.delete(id);
      if (selectedId === id) {
        comparisonPlayer.reset();
        selectedId = selectedJobId = null;
        safeSessionWrite('neural-amp-modeler.take', null);
      }
      deletingPairId = null;
      $('delete-dialog').close();
      showLibrary();
      await updateState();
    }, {errorTarget: 'delete-error', button: $('delete-submit')});
  }
  function uploadRow(item) {
    const failed = item.status === 'failed';
    const label = failed ? 'Upload failed' : item.status === 'processing' ? 'Processing…' : `Uploading ${item.progress}%`;
    const file = (value) => `<span class="pair-file"><span class="file-type">WAV</span><span title="${escapeHtml(value.name)}">${escapeHtml(value.name)}</span></span>`;
    return `<tr class="pair-row upload-row"><td class="pair-name-cell"><span class="pair-name">${escapeHtml(item.name)}</span><div class="pair-meta${failed ? ' pair-warning' : ''}">${escapeHtml(failed ? item.error : 'Uploading pair')}</div></td><td data-label="Input">${file(item.dry)}</td><td data-label="Output">${file(item.wet)}</td><td data-label="Status"><div class="upload-state${failed ? ' pair-warning' : ''}" role="status">${failed ? '' : '<span class="upload-spinner" aria-hidden="true"></span>'}${label}</div>${failed ? '' : `<progress class="upload-progress" max="100" value="${item.progress}" aria-label="Upload progress for ${escapeHtml(item.name)}"></progress>`}</td><td class="pair-row-action">${failed ? `<button class="text-link" data-retry-upload="${item.id}" data-library-focus="retry-${item.id}">Retry upload</button>` : ''}</td></tr>`;
  }
  function renderLibrary() {
    const query = $('library-search').value.trim().toLowerCase();
    const pending = [...uploads.items.values()].filter((item) => !state.captures.some((take) => take.id === item.id));
    const pendingRows = pending.filter((item) => item.name.toLowerCase().includes(query));
    const pairs = state.captures.filter((item) => item.name.toLowerCase().includes(query));
    const count = state.captures.length;
    $('library-count').textContent = initialized ? `${query ? `${pairs.length} of ` : ''}${count} ${count === 1 ? 'pair' : 'pairs'} · ${state.jobs.length} training ${state.jobs.length === 1 ? 'run' : 'runs'}${pending.length ? ` · ${pending.length} ${pending.every((item) => item.status === 'failed') ? 'failed' : 'uploading'}` : ''}` : 'Loading your pairs…';
    $('resume-recording').hidden = !state.bridge.recording;
    $('library-empty').hidden = Boolean(count || pending.length) || !initialized;
    const key = JSON.stringify([query, pending.map(({id, status, progress, error}) => ({id, status, progress, error})), [...expandedPairs], state.captures.map(({id, name, source, filenames, status, clipped}) => ({id, name, source, filenames, status, clipped})), state.jobs.map(({id, status, epoch, epochs, model_size, progress, best_loss}) => ({id, status, epoch, epochs, model_size, progress, best_loss}))]);
    if (key === libraryRenderKey) return;
    libraryRenderKey = key;
    const focus = $('library-rows').contains(document.activeElement) ? document.activeElement.dataset.libraryFocus : null;
    $('library-rows').innerHTML = pendingRows.map(uploadRow).join('') + (pairs.map((take) => {
      const runs = state.jobs.filter((item) => item.capture_id === take.id);
      const activeRuns = runs.filter((item) => ['queued', 'training'].includes(item.status));
      const running = activeRuns[0];
      const displayedRun = running || runs[0];
      const ready = runs.filter((item) => item.status === 'completed').length;
      const expanded = expandedPairs.has(take.id);
      const source = take.source === 'upload' ? 'Uploaded pair' : take.source === 'demo' ? 'Synthetic demo' : 'Live recording';
      const problem = take.status !== 'ready' ? 'Interrupted' : take.clipped ? 'Clipping' : '';
      const file = (channel) => `<a class="pair-file" href="${escapeHtml(downloadUrl(take[channel + '_url']))}" download aria-label="Download ${channel === 'dry' ? 'input' : 'output'} for ${escapeHtml(take.name)}"><span class="file-type">WAV</span><span title="${escapeHtml(take.filenames?.[channel] || channel + '.wav')}">${escapeHtml(take.filenames?.[channel] || channel + '.wav')}</span></a>`;
      const summary = activeRuns.length > 1 ? `${activeRuns.length} active runs` : running ? runLabel(running) : ready ? `${ready} ${ready === 1 ? 'run' : 'runs'} complete` : runs.length ? 'No completed models' : 'Not trained';
      const history = expanded ? `<tr class="pair-history"><td colspan="5"><div class="run-history"><div class="run-history-heading">Training runs <span>${runs.length}</span></div>${runs.map((run) => `<div class="history-entry"><button class="history-run" data-run="${run.id}" data-pair="${take.id}" data-library-focus="run-${run.id}"><span class="run-model">${modelName(run)}<small>${run.epochs} epochs</small></span><span class="run-state ${run.status === 'completed' ? 'ready' : run.status === 'training' || run.status === 'queued' ? 'training' : ''}">${runLabel(run)}</span><span class="run-score">${Number.isFinite(run.best_loss) ? `ESR ${Number(run.best_loss).toPrecision(3)}` : '—'}</span><time>${escapeHtml(dateLabel(run.created_at, true))}</time><span aria-hidden="true">↗</span></button>${run.status === 'completed' && run.model_url ? `<a class="history-download" href="${escapeHtml(downloadUrl(run.model_url))}" download data-library-focus="download-${run.id}" aria-label="Download ${modelName(run)} ${run.epochs} epochs for ${escapeHtml(take.name)}, ${escapeHtml(dateLabel(run.created_at, true))}">Download .nam <span aria-hidden="true">↓</span></a>` : ''}</div>`).join('')}</div></td></tr>` : '';
      return `<tr class="pair-row${expanded ? ' expanded' : ''}"><td class="pair-name-cell"><button class="pair-name" data-open-pair="${take.id}" data-library-focus="pair-${take.id}">${escapeHtml(take.name)}</button><div class="pair-meta"><span>${formatTime(take.duration, false)}</span><span>${escapeHtml(dateLabel(take.created_at))}</span><span>${source}</span>${problem ? `<span class="pair-warning">${problem}</span>` : ''}</div></td><td data-label="Input">${file('dry')}</td><td data-label="Output">${file('wet')}</td><td class="pair-training" data-label="Training">${runs.length ? `<button class="runs-toggle" data-expand="${take.id}" data-library-focus="expand-${take.id}" aria-expanded="${expanded}" aria-label="${expanded ? 'Hide' : 'Show'} training runs for ${escapeHtml(take.name)}"><span class="run-state ${running ? 'training' : ready ? 'ready' : ''}">${summary}</span><span class="run-settings">${modelName(displayedRun)} · ${displayedRun.epochs} epochs</span><span class="run-count">${expanded ? 'Hide' : 'View'} ${runs.length} ${runs.length === 1 ? 'run' : 'runs'}<svg class="expand-chevron" viewBox="0 0 24 24" aria-hidden="true"><path d="m9 5 7 7-7 7"/></svg></span></button>` : '<span class="not-trained">Not trained</span>'}</td><td class="pair-row-action"><div class="row-actions"><button class="train-pair" data-train-pair="${take.id}" data-library-focus="train-${take.id}" aria-label="Train ${escapeHtml(take.name)}">Train <span>↗</span></button><button class="delete-pair icon-button" data-delete-pair="${take.id}" data-library-focus="delete-${take.id}" aria-label="Delete ${escapeHtml(take.name)}" title="${running ? 'Training in progress' : 'Delete pair'}" ${running ? 'disabled' : ''}><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 7h16M9 7V4h6v3M6 7l1 13h10l1-13M10 10v7M14 10v7"/></svg></button></div></td></tr>${history}`;
    }).join('') || (query && !pendingRows.length ? '<tr><td colspan="5" class="no-results">No pairs match your search.</td></tr>' : ''));
    if (focus) [...$('library-rows').querySelectorAll('[data-library-focus]')].find((el) => el.dataset.libraryFocus === focus)?.focus({preventScroll: true});
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
        if (Date.now() - Date.parse(newest.created_at) < 20000) { selectedId = newest.id; makingNewTake = false; safeSessionWrite('neural-amp-modeler.take', selectedId); }
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
          if (stoppedCaptureId && state.captures.some((item) => item.id === stoppedCaptureId)) { selectedId = stoppedCaptureId; if (mode === 'record') showLibrary(); }
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
    if (busy || offline) return;
    if (!state.bridge.connected && !state.bridge.recording) { openDialog('connection-dialog'); return; }
    await runAction(async () => {
      if (state.bridge.recording) {
        const result = await api('/api/captures/record', { method: 'POST', body: JSON.stringify({ action: 'stop' }) });
        state.bridge.recording = false;
        await updateState();
        if (result.capture) {
          if (!state.captures.some((item) => item.id === result.capture.id)) state.captures.unshift(result.capture);
          selectedId = result.capture.id;
          $('library-search').value = '';
          showLibrary();
          if (result.capture.duration < 12) status('Take saved. Record at least 12 seconds when you’re ready to train.', 'neutral');
        } else { newTake(); status('That take was under a second. Give it another go.', 'neutral'); }
      } else {
        pauseAudio();
        mode = 'record';
        selectedId = null;
        makingNewTake = true;
        livePeaks = [];
        try {
          await api('/api/captures/record', { method: 'POST', body: JSON.stringify({ action: 'start', name: $('take-name').value.trim() || 'Untitled take' }) });
        } catch (error) {
          // The bridge may disconnect between the last status update and click.
          if (error.status === 409) {
            await updateState();
            if (!offline && !state.bridge.connected) { openDialog('connection-dialog'); return; }
          }
          throw error;
        }
        state.bridge.recording = true;
        state.bridge.samples = 0;
        await updateState();
      }
    });
  }
  async function train() {
    const take = state.captures.find((item) => item.id === trainingPairId);
    const issue = validTrainingTake(take);
    if (issue) { showError('training-error', issue); return; }
    await runAction(async () => {
      const result = await api('/api/train', { method: 'POST', body: JSON.stringify({ capture_id: take.id, epochs: Number($('epochs').value), model_size: 'full' }) });
      state.jobs.unshift(result);
      selectTake(take.id, 'train');
      selectedJobId = result.id;
      previousJobStatuses.set(result.id, result.status);
      $('training-dialog').close();
      await updateState();
    }, { errorTarget: 'training-error', button: $('training-submit') });
  }

  function pauseAudio() { comparisonPlayer.pause(); }
  function syncAudioSource() {
    if (mode !== 'compare' || view !== 'detail') return;
    const take = capture();
    const item = job();
    if (!take) { comparisonPlayer.reset(); return; }
    const completed = item?.status === 'completed' && item.prediction_url;
    const urls = completed
      ? {dry: item.dry_url, reference: item.reference_url, model: item.prediction_url}
      : {dry: take.dry_url, reference: take.wet_url};
    comparisonPlayer.load(`${take.id}:${completed ? item.id : 'capture'}`, urls);
  }
  function togglePlayback() { comparisonPlayer.togglePlayback(); }
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
    if (!width || !height || mode === 'compare') return;
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
      ctx.strokeStyle = !peaks.length ? '#444741' : mode === 'train' ? '#84867c' : '#eeeae2';
      ctx.beginPath(); ctx.moveTo(x, center - amplitude); ctx.lineTo(x, center + amplitude); ctx.stroke();
    }
    if (cursor !== null && peaks.length) {
      const x = Math.max(2, Math.min(width - 2, cursor * width));
      ctx.strokeStyle = '#f56855'; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(x, center - height * .43); ctx.lineTo(x, center + height * .43); ctx.stroke();
      ctx.fillStyle = '#f56855'; ctx.beginPath(); ctx.arc(x, center - height * .43, 3.5, 0, Math.PI * 2); ctx.fill();
    }
    canvas.setAttribute('aria-label', isLive ? 'Live input waveform' : take ? 'Processed tone waveform' : 'No audio captured yet');
  }
  $('training-run').addEventListener('change', () => {
    if ($('training-run').value === '__new__') { openTraining(selectedId); render(); return; }
    selectedJobId = $('training-run').value;
    pauseAudio();
    clearStatus();
    render();
    if (mode === 'compare') syncAudioSource();
  });
  $('loss-scale').addEventListener('click', () => {
    logarithmicLoss = !logarithmicLoss;
    $('loss-scale').setAttribute('aria-pressed', String(logarithmicLoss));
    $('loss-scale').textContent = logarithmicLoss ? 'Log scale' : 'Linear scale';
    if (job()) renderLearning(job());
  });
  document.querySelector('.wordmark').addEventListener('click', (event) => { event.preventDefault(); showLibrary(); });

  $('connection-record').addEventListener('click', async () => { $('connection-dialog').close(); await record(); });
  $('connection-dialog').addEventListener('close', () => showError('connection-error', ''));
  $('back-library').addEventListener('click', showLibrary);
  $('record-live').addEventListener('click', () => { const name = $('upload-name').value.trim(); newTake(); if (name) $('take-name').value = name; });
  $('resume-recording').addEventListener('click', () => { view = 'detail'; mode = 'record'; render(); });
  $('detail-compare').addEventListener('click', () => setMode('compare'));
  $('detail-training').addEventListener('click', () => setMode('train'));
  $('detail-train').addEventListener('click', () => openTraining(selectedId));
  $('library-search').addEventListener('input', renderLibrary);
  $('library-rows').addEventListener('click', (event) => {
    const button = event.target.closest('button');
    if (!button) return;
    if (button.dataset.expand) { const id = button.dataset.expand; expandedPairs.has(id) ? expandedPairs.delete(id) : expandedPairs.add(id); renderLibrary(); }
    if (button.dataset.openPair) selectTake(button.dataset.openPair, 'train');
    if (button.dataset.trainPair) openTraining(button.dataset.trainPair);
    if (button.dataset.deletePair) openDelete(button.dataset.deletePair);
    if (button.dataset.retryUpload) uploads.start(button.dataset.retryUpload);
    if (button.dataset.run) { selectTake(button.dataset.pair, 'train'); selectedJobId = button.dataset.run; render(); }
  });
  $('delete-submit').addEventListener('click', deletePair);
  $('training-form').addEventListener('submit', (event) => { event.preventDefault(); train(); });
  $('dismiss-status').addEventListener('click', clearStatus);
  $('primary-action').addEventListener('click', async () => {
    if (offline) { clearStatus(); await initialize(); return; }
    if (mode === 'record') {
      if (state.bridge.recording) await record();
      else if (capture()) setMode(job()?.status === 'completed' ? 'compare' : 'train');
      else await record();
    } else if (mode === 'train') {
      if (job()?.status === 'completed' && selectedJobId !== '__new__') setMode('compare'); else openTraining(selectedId);
    } else await togglePlayback();
  });
  document.querySelectorAll('.close-dialog').forEach((button) => button.addEventListener('click', () => button.closest('dialog').close()));
  document.querySelectorAll('dialog').forEach((dialog) => dialog.addEventListener('click', (event) => { if (event.target !== dialog || dialog.id === 'auth-dialog') return; const rect = dialog.getBoundingClientRect(); if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) dialog.close(); }));
  $('auth-dialog').addEventListener('cancel', (event) => event.preventDefault());
  $('copy-command').addEventListener('click', async () => {
    try { await navigator.clipboard.writeText($('bridge-command').textContent); $('copy-command').innerHTML = '<svg viewBox="0 0 24 24"><path d="m5 12 4 4L19 6"/></svg>'; setTimeout(() => { $('copy-command').innerHTML = '<svg viewBox="0 0 24 24"><rect x="8" y="8" width="11" height="12" rx="2"/><path d="M15 8V4H4v12h4"/></svg>'; }, 2000); }
    catch { showError('connection-error', 'Copy is unavailable in this browser. Select and copy the command above.'); }
  });
  function openUpload() { showError('upload-error', ''); openDialog('upload-dialog'); }
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
    const name = $('upload-name').value.trim() || 'Untitled take';
    $('upload-dialog').close();
    $('upload-form').reset();
    ['dry', 'wet'].forEach((channel) => { $(channel + '-file-label').textContent = 'Choose WAV ↑'; $(channel + '-file').closest('.file-picker').classList.remove('has-file'); });
    $('library-search').value = '';
    showLibrary();
    uploads.add(name, dry, wet);
  });
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
  $('epochs').addEventListener('change', () => { if (mode === 'train') render(); });
  document.addEventListener('keydown', (event) => {
    const target = event.target;
    if (target instanceof HTMLElement && (target.matches('input, textarea, select, button, summary') || target.isContentEditable)) return;
    if (document.querySelector('dialog[open]') || event.ctrlKey || event.metaKey || event.altKey || event.repeat) return;
    if (event.code === 'Space' && mode === 'compare' && view === 'detail' && capture()) { event.preventDefault(); togglePlayback(); }
    if (view === 'detail' && mode === 'record' && event.key.toLowerCase() === 'r') { event.preventDefault(); record(); }
  });
  document.addEventListener('visibilitychange', () => { if (!document.hidden && initialized) { updateState(); connectMonitor(); } });
  window.addEventListener('beforeunload', (event) => { if (state.bridge.recording || uploads.active) { event.preventDefault(); event.returnValue = ''; } });
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
