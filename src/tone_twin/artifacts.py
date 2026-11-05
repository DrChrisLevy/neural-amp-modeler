"""Bounded local copies for range-capable playback without holding Volume files open."""

import asyncio
import shutil
import tempfile
from collections import OrderedDict
from pathlib import Path
from starlette.responses import FileResponse


class ArtifactResponse(FileResponse):
    def __init__(self, *args, release, **kwargs):
        super().__init__(*args, **kwargs)
        self.release = release

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            await self.release()


class Artifacts:
    def __init__(self, max_bytes=512 * 1024 * 1024):
        self.directory = tempfile.TemporaryDirectory(prefix="tone-twin-playback-")
        self.entries = OrderedDict()
        self.pending = {}
        self.waiters = {}
        self.max_bytes = max_bytes

    async def acquire(self, key, source):
        self.waiters[key] = self.waiters.get(key, 0) + 1
        try:
            if key not in self.entries:
                if key not in self.pending:
                    self.pending[key] = asyncio.create_task(self._copy(key, source))
                    self.pending[key].add_done_callback(lambda task: task.exception() if not task.cancelled() else None)
                await asyncio.shield(self.pending[key])
            entry = self.entries[key]
            entry["users"] += 1
            self.entries.move_to_end(key)
            return entry["path"]
        finally:
            self.waiters[key] -= 1
            if not self.waiters[key]:
                del self.waiters[key]
            self._trim()

    async def _copy(self, key, source):
        try:
            destination = Path(self.directory.name) / key
            await source(destination)
            self.entries[key] = {"path": destination, "size": destination.stat().st_size, "users": 0}
        finally:
            self.pending.pop(key, None)
            self._trim()

    async def release(self, key):
        self.entries[key]["users"] -= 1
        self._trim()

    def _trim(self):
        size = sum(e["size"] for e in self.entries.values())
        for candidate, entry in list(self.entries.items()):
            if size <= self.max_bytes:
                break
            if entry["users"] == 0 and not self.waiters.get(candidate):
                entry["path"].unlink(missing_ok=True)
                size -= entry["size"]
                del self.entries[candidate]

    async def close(self):
        await asyncio.gather(*list(self.pending.values()), return_exceptions=True)
        await asyncio.to_thread(self.directory.cleanup)

    @staticmethod
    def copy(source, destination):
        # A failed transfer never becomes a playable cache entry.
        try:
            shutil.copyfile(source, destination)
        except BaseException:
            destination.unlink(missing_ok=True)
            raise
