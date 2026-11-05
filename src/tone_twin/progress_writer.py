"""Publish the newest progress snapshot without stalling the training thread."""

import copy
import threading


class ProgressWriter:
    """One writer, at most one pending snapshot, and a mandatory final flush.

    Each snapshot contains the full measured history, so coalescing intermediate
    snapshots never loses measurements. Write errors are surfaced to the caller.
    """

    def __init__(self, write):
        self.write = write
        self.condition = threading.Condition()
        self.pending = None
        self.busy = False
        self.closed = False
        self.error = None
        self.thread = threading.Thread(target=self._run, name="training-progress", daemon=True)
        self.thread.start()

    def submit(self, snapshot):
        with self.condition:
            if self.error:
                raise self.error
            if self.closed:
                raise RuntimeError("Progress writer is closed")
            self.pending = copy.deepcopy(snapshot)
            self.condition.notify_all()

    def _run(self):
        while True:
            with self.condition:
                self.condition.wait_for(lambda: self.pending is not None or self.closed)
                if self.pending is None:
                    return
                snapshot, self.pending = self.pending, None
                self.busy = True
            try:
                self.write(snapshot)
            except Exception as exc:
                with self.condition:
                    self.error = exc
                    self.pending = None
                    self.busy = False
                    self.condition.notify_all()
                return
            with self.condition:
                self.busy = False
                self.condition.notify_all()

    def close(self):
        with self.condition:
            self.closed = True
            self.condition.notify_all()
        self.thread.join()
        if self.error:
            raise self.error
