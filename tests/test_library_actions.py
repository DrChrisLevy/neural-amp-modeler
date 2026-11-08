import asyncio
import io
import json
import threading

import httpx
import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from nam_studio.server import create_app
from nam_studio.storage import atomic_json


def files():
    out = io.BytesIO()
    sf.write(out, np.zeros(48000), 48000, format='WAV', subtype='PCM_24')
    return {channel: (f'{channel}.wav', out.getvalue(), 'audio/wav') for channel in ('dry', 'wet')}


@pytest.mark.parametrize(('name', 'stem', 'epochs'), [
    ("Victor's Mega Squid", 'victors_mega_squid', 400),
    ('Victor’s Mega Squid', 'victors_mega_squid', 1000),
    ('../Café / Clean!\r\n', 'cafe_clean', 50),
    ('🎸', 'capture', 100),
])
def test_downloads_use_parent_pair_name_and_keep_audio_playable(tmp_path, name, stem, epochs):
    app = create_app(tmp_path)
    with TestClient(app) as c:
        take = c.post('/api/captures/upload', files=files(), data={'name': name}).json()
        c.post('/api/captures/upload', files=files(), data={'name': 'Another pair'})
        job_id = 'f' * 32
        job_path = app.state.store.path('jobs', job_id)
        atomic_json(job_path / 'status.json', {'id': job_id, 'capture_id': take['id'], 'name': 'Old name', 'status': 'completed', 'epochs': epochs})
        (job_path / 'model.nam').write_text('{}')
        model_url = c.get('/api/state').json()['jobs'][0]['model_url']
        for url, suffix in [(model_url, f'_{epochs}_epochs.nam'), (take['dry_url'], '_dry.wav'), (take['wet_url'], '_wet.wav')]:
            response = c.get(url, params={'download': 1})
            assert response.status_code == 200
            assert response.headers['content-disposition'] == f'attachment; filename="{stem}{suffix}"'
            assert response.content == c.get(url).content
        playback = c.get(take['dry_url'], headers={'Range': 'bytes=0-43'})
        assert playback.status_code == 206
        assert playback.headers['content-disposition'] == f'inline; filename="{stem}_dry.wav"'
        assert playback.content.startswith(b'RIFF')


def test_delete_cascades_runs_and_revokes_cached_downloads(tmp_path):
    app = create_app(tmp_path)
    with TestClient(app) as c:
        first = c.post('/api/captures/upload', files=files(), data={'name':'Delete'}).json()
        other = c.post('/api/captures/upload', files=files(), data={'name':'Keep'}).json()
        job_id = 'a' * 32
        job_path = app.state.store.path('jobs', job_id)
        atomic_json(job_path / 'status.json', {'id':job_id,'capture_id':first['id'],'status':'completed','created_at':'2026-11-06'})
        (job_path / 'model.nam').write_text('{}')
        state = c.get('/api/state').json()
        model_url = state['jobs'][0]['model_url']
        assert c.get(first['dry_url']).status_code == c.get(model_url).status_code == 200
        assert c.delete(f"/api/captures/{first['id']}").status_code == 200
        current = c.get('/api/state', params={'revision':state['revision']}).json()
        assert [x['id'] for x in current['captures']] == [other['id']]
        assert current['jobs'] == []
        assert not app.state.store.path('captures',first['id']).exists()
        assert not job_path.exists()
        assert c.get(first['dry_url']).status_code == c.get(model_url).status_code == 404
        assert c.get(other['dry_url']).status_code == 200


@pytest.mark.parametrize('status', ['queued','training'])
def test_delete_refuses_active_jobs(tmp_path, status):
    app = create_app(tmp_path)
    with TestClient(app) as c:
        take = c.post('/api/captures/upload',files=files()).json()
        job_id = 'b' * 32
        atomic_json(app.state.store.path('jobs',job_id)/'status.json', {'id':job_id,'capture_id':take['id'],'status':status})
        assert c.delete(f"/api/captures/{take['id']}").status_code == 409
        assert app.state.store.path('captures',take['id']).exists()
        assert app.state.store.path('jobs',job_id).exists()


def test_delete_rejects_invalid_ids_and_missing_rows(tmp_path):
    app = create_app(tmp_path)
    with TestClient(app) as c:
        assert c.delete('/api/captures/invalid').status_code == 400
        assert c.delete('/api/captures/'+'a'*32).status_code == 404


def test_upload_retries_reuse_the_same_capture_without_overwriting(tmp_path):
    app=create_app(tmp_path)
    with TestClient(app) as c:
        data={'name':'Original','upload_id':'c'*32}
        first=c.post('/api/captures/upload',files=files(),data=data).json()
        again=c.post('/api/captures/upload',files=files(),data={**data,'name':'Duplicate request'}).json()
        assert first['id']==again['id']=='c'*32
        assert again['name']=='Original'
        assert len(c.get('/api/state').json()['captures'])==1
        assert c.post('/api/captures/upload',files=files(),data={'upload_id':'bad'}).status_code==400


def test_concurrent_uploads_keep_controls_responsive_and_deduplicate(tmp_path, monkeypatch):
    async def run():
        app=create_app(tmp_path)
        started, release=threading.Event(), threading.Event()
        original=app.state.store.capture
        def slow_save(*args,**kwargs):
            started.set()
            assert release.wait(5)
            return original(*args,**kwargs)
        monkeypatch.setattr(app.state.store,'capture',slow_save)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://test') as c:
                requests=[asyncio.create_task(c.post('/api/captures/upload',files=files(),data={'name':name,'upload_id':i*32})) for name,i in [('A','d'),('B','e'),('Retry A','d')]]
                try:
                    assert await asyncio.to_thread(started.wait,1)
                    assert (await asyncio.wait_for(c.get('/api/config'),.5)).status_code==200
                finally:
                    release.set()
                responses=await asyncio.gather(*requests)
                assert all(r.status_code==200 for r in responses)
                assert len((await c.get('/api/state')).json()['captures'])==2
    asyncio.run(run())
