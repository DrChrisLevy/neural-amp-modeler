"""A shared metadata snapshot; cloud filesystem reads never run on the event loop."""

import asyncio
import logging
import time
import uuid


class Catalog:
    def __init__(self, store, lock, reload, *, active_interval=1.0, idle_interval=30.0):
        self.store, self.lock, self.reload = store, lock, reload
        self.active_interval, self.idle_interval = active_interval, idle_interval
        self.captures, self.jobs = [], []
        self.loaded = False
        self.updated_at = self.scanned_at = float("-inf")
        self.task = None
        self.generation = uuid.uuid4().hex
        self.version = 0

    @property
    def revision(self):
        return f"{self.generation}:{self.version}"

    def put_capture(self, capture):
        self.captures = sorted([capture, *(c for c in self.captures if c["id"] != capture["id"])], key=lambda c: c["created_at"], reverse=True)
        self.version += 1

    def put_job(self, job):
        self.jobs = sorted([job, *(j for j in self.jobs if j["id"] != job["id"])], key=lambda j: j.get("created_at", ""), reverse=True)
        self.version += 1

    async def get(self, *, fresh=False):
        active = any(j["status"] in {"queued", "training"} for j in self.jobs)
        interval = self.active_interval if active else self.idle_interval
        if (fresh or not self.loaded or time.monotonic() - self.updated_at >= interval) and (self.task is None or self.task.done()):
            self.task = asyncio.create_task(self._refresh())
            self.task.add_done_callback(self._report_error)
        if not self.loaded or fresh:
            # Disconnecting a browser must not cancel the shared refresh.
            await asyncio.shield(self.task)
        return self

    @staticmethod
    def _report_error(task):
        if not task.cancelled() and task.exception() is not None:
            logging.getLogger(__name__).error("Studio metadata refresh failed", exc_info=task.exception())

    async def _refresh(self):
        async with self.lock:
            await self.reload()
            full = not self.loaded or time.monotonic() - self.scanned_at >= self.idle_interval
            if full:
                captures, jobs = await asyncio.to_thread(self.store.snapshot)
                self.scanned_at = time.monotonic()
            else:
                captures = self.captures
                # Completed models are immutable. Only the running job changes.
                active = [j["id"] for j in self.jobs if j["status"] in {"queued", "training"}]
                updates = await asyncio.to_thread(lambda: {job_id: self.store.job(job_id) for job_id in active})
                jobs = [updates.get(j["id"], j) for j in self.jobs]
            if captures != self.captures or jobs != self.jobs or not self.loaded:
                self.captures, self.jobs = captures, jobs
                self.version += 1
            self.loaded = True
            self.updated_at = time.monotonic()

    async def close(self):
        if self.task is not None:
            await asyncio.gather(self.task, return_exceptions=True)
