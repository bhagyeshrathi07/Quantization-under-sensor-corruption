"""Peak GPU memory sampling via nvidia-smi. Measures total device use, which
includes the vLLM server's reserved pool; that is the number the proposal asks
for (total observed peak), not model allocations alone."""

import subprocess
import threading
import time


def query_memory_used_mib(gpu_index: int = 0) -> int | None:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", f"--id={gpu_index}", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
        )
        return int(out.strip().splitlines()[0])
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError, IndexError):
        return None


class PeakMemorySampler:
    def __init__(self, gpu_index: int = 0, interval_s: float = 0.5):
        self.gpu_index = gpu_index
        self.interval_s = interval_s
        self.peak_mib: int | None = None
        self.baseline_mib: int | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            v = query_memory_used_mib(self.gpu_index)
            if v is not None and (self.peak_mib is None or v > self.peak_mib):
                self.peak_mib = v
            time.sleep(self.interval_s)

    def __enter__(self):
        self.baseline_mib = query_memory_used_mib(self.gpu_index)
        self.peak_mib = self.baseline_mib
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
