'use strict';

// Each upload owns its request and status; none takes the studio's action lock.
window.NAMUploads = class {
  constructor({changed, completed, request = () => new XMLHttpRequest()}) {
    this.items = new Map();
    Object.assign(this, {changed, completed, request});
  }
  add(name, dry, wet) {
    const id = crypto.randomUUID().replaceAll('-', '');
    this.items.set(id, {id, name, dry, wet, status: 'pending', progress: 0, error: ''});
    this.start(id);
    return id;
  }
  start(id) {
    const item = this.items.get(id);
    if (!item || ['uploading', 'processing'].includes(item.status)) return;
    Object.assign(item, {status: 'uploading', progress: 0, error: ''});
    this.changed();
    const xhr = this.request();
    const fail = (message) => { item.status = 'failed'; item.error = message; this.changed(); };
    xhr.open('POST', '/api/captures/upload');
    xhr.responseType = 'json';
    xhr.withCredentials = true;
    xhr.timeout = 10 * 60 * 1000;
    xhr.upload.onprogress = ({lengthComputable, loaded, total}) => {
      if (!lengthComputable) return;
      item.progress = Math.round(100 * loaded / total);
      if (item.progress >= 100) item.status = 'processing';
      this.changed();
    };
    xhr.upload.onload = () => { item.progress = 100; item.status = 'processing'; this.changed(); };
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300 && xhr.response?.id) {
        this.items.delete(id);
        this.completed(xhr.response);
        this.changed();
      } else {
        fail(typeof xhr.response?.detail === 'string' ? xhr.response.detail : `Upload failed (${xhr.status}). Try again.`);
      }
    };
    xhr.onerror = () => fail('Connection lost. Retry this upload.');
    xhr.ontimeout = () => fail('Upload timed out. Retry this upload.');
    xhr.onabort = () => fail('Upload stopped. Retry this upload.');
    const body = new FormData();
    body.set('name', item.name); body.set('dry', item.dry); body.set('wet', item.wet);
    body.set('upload_id', id); // Stable across retries, so the server can deduplicate.
    try { xhr.send(body); } catch { fail('Upload could not start. Try again.'); }
  }
  get active() {
    return [...this.items.values()].some((item) => ['uploading', 'processing'].includes(item.status));
  }
};
