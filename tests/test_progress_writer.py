import threading
import unittest

from nam_studio.progress_writer import ProgressWriter


class ProgressWriterTests(unittest.TestCase):
    def test_slow_storage_coalesces_updates_and_flushes_final_history(self):
        started, release = threading.Event(), threading.Event()
        written = []

        def write(snapshot):
            started.set()
            if not release.wait(2):
                raise RuntimeError("Test did not release the writer")
            written.append(snapshot)

        writer = ProgressWriter(write)
        try:
            writer.submit({"epoch": 1, "metrics": [1]})
            self.assertTrue(started.wait(2))
            writer.submit({"epoch": 2, "metrics": [1, 2]})
            final = {"epoch": 3, "metrics": [1, 2, 3], "status": "completed"}
            writer.submit(final)
            final["metrics"].append(99)
        finally:
            release.set()
            writer.close()
        self.assertEqual([item["epoch"] for item in written], [1, 3])
        self.assertEqual(written[-1]["metrics"], [1, 2, 3])
        self.assertEqual(written[-1]["status"], "completed")
        self.assertFalse(writer.thread.is_alive())

    def test_background_write_failure_is_reported(self):
        def fail(snapshot):
            raise OSError("Storage unavailable")

        writer = ProgressWriter(fail)
        writer.submit({"status": "training"})
        with self.assertRaisesRegex(OSError, "Storage unavailable"):
            writer.close()
        self.assertFalse(writer.thread.is_alive())
