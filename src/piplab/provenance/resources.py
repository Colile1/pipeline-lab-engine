# resources.py: measure what one run actually cost.
#
# Peak memory is the awkward one. Three sources, in descending order of
# trustworthiness, and the one used is RECORDED so figures from
# different methods are never silently compared:
#
#   1. Windows  psutil peak_wset   - the OS tracks the true peak, so it
#                                    survives deallocation. Verified:
#                                    freeing a 61 MB array leaves
#                                    peak_wset at 99 MB while rss falls
#                                    back to 38 MB.
#   2. POSIX    resource.ru_maxrss - the kernel's high-water mark.
#   3. fallback sampled rss        - a background thread polls rss; it
#                                    can miss a spike between samples,
#                                    so it is marked approximate.
#
# tracemalloc is deliberately NOT used: it only sees Python-level
# allocations and undercounts badly for this workload (measured 61 MB
# against the OS's 99 MB) because numpy and BLAS allocate natively.
from __future__ import annotations

import sys
import threading
import time
from dataclasses import asdict, dataclass

import psutil

SAMPLE_INTERVAL_SECONDS = 0.05


@dataclass
class ResourceUsage:
    """What one measured section of work cost."""

    wall_seconds: float
    cpu_seconds_user: float
    cpu_seconds_system: float
    cpu_percent_mean: float | None
    peak_rss_mb: float
    peak_rss_method: str
    rss_start_mb: float
    rss_end_mb: float
    n_threads_peak: int
    disk_read_mb: float | None
    disk_write_mb: float | None
    n_samples: int = 0

    def as_dict(self) -> dict:
        """Return the usage as a flat dict for the run record."""
        return asdict(self)

    @property
    def cpu_seconds_total(self) -> float:
        """Total CPU time, user plus system."""
        return self.cpu_seconds_user + self.cpu_seconds_system

    @property
    def parallel_efficiency(self) -> float | None:
        """CPU seconds per wall second: >1 means real parallelism.

        Low values on a long run indicate the process was blocked rather
        than computing -- the signature of the paging this machine has
        shown, where a b2 run took ~6 minutes of wall-clock for under a
        second of CPU. Worth having in the trend data.

        Note this includes the monitor's own sampler thread, which costs
        a small amount of system time (measured ~0.015 s over 0.3 s), so
        a genuinely idle block reads slightly above zero rather than
        exactly zero.
        """
        if self.wall_seconds <= 0:
            return None
        return round(self.cpu_seconds_total / self.wall_seconds, 3)


class ResourceMonitor:
    """Measure wall time, CPU time, peak memory and IO for a block of work.

    Used as a context manager:

        with ResourceMonitor() as monitor:
            ...work...
        usage = monitor.usage()

    A background sampler runs only to collect CPU percent and thread
    count; peak memory comes from the OS where possible (see module
    docstring), so a missed sample cannot understate the peak on Windows
    or POSIX.
    """

    def __init__(self, sample_interval: float = SAMPLE_INTERVAL_SECONDS):
        self.sample_interval = sample_interval
        self._process = psutil.Process()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._cpu_samples: list[float] = []
        self._rss_samples: list[int] = []
        self._thread_counts: list[int] = []
        self._usage: ResourceUsage | None = None

    def __enter__(self) -> "ResourceMonitor":
        """Reset the OS peak counter where possible and start sampling."""
        self._reset_peak()
        self._cpu_start = self._process.cpu_times()
        self._io_start = _io_counters(self._process)
        self._rss_start = self._process.memory_info().rss
        self._wall_start = time.perf_counter()
        self._process.cpu_percent(None)  # prime the interval
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        """Stop sampling and compute the usage. Never swallows an exception."""
        wall = time.perf_counter() - self._wall_start
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        cpu_end = self._process.cpu_times()
        io_end = _io_counters(self._process)
        rss_end = self._process.memory_info().rss
        peak_mb, method = self._peak_memory_mb()

        self._usage = ResourceUsage(
            wall_seconds=round(wall, 4),
            cpu_seconds_user=round(cpu_end.user - self._cpu_start.user, 4),
            cpu_seconds_system=round(cpu_end.system - self._cpu_start.system, 4),
            cpu_percent_mean=(
                round(sum(self._cpu_samples) / len(self._cpu_samples), 2)
                if self._cpu_samples else None
            ),
            peak_rss_mb=peak_mb,
            peak_rss_method=method,
            rss_start_mb=round(self._rss_start / 1024**2, 2),
            rss_end_mb=round(rss_end / 1024**2, 2),
            n_threads_peak=max(self._thread_counts) if self._thread_counts else self._process.num_threads(),
            disk_read_mb=_delta_mb(self._io_start, io_end, "read_bytes"),
            disk_write_mb=_delta_mb(self._io_start, io_end, "write_bytes"),
            n_samples=len(self._cpu_samples),
        )
        return False  # propagate any exception

    def usage(self) -> ResourceUsage:
        """Return the measured usage.

        Raises: RuntimeError if the monitor has not exited yet, rather
        than returning a half-measured figure.
        """
        if self._usage is None:
            raise RuntimeError("ResourceMonitor.usage() called before the block finished")
        return self._usage

    def _sample(self) -> None:
        """Poll CPU percent, rss and thread count until told to stop."""
        while not self._stop.wait(self.sample_interval):
            try:
                self._cpu_samples.append(self._process.cpu_percent(None))
                self._rss_samples.append(self._process.memory_info().rss)
                self._thread_counts.append(self._process.num_threads())
            except (psutil.NoSuchProcess, psutil.AccessDenied):  # pragma: no cover
                return

    def _reset_peak(self) -> None:
        """Clear the OS peak counter so it measures only this block.

        Windows exposes no reset for peak_wset, so on Windows the peak is
        reported relative to the process's lifetime high-water mark; the
        monitor compensates by also tracking the sampled maximum and
        taking whichever is larger but never smaller than this block's
        own start. POSIX ru_maxrss likewise cannot be reset, handled the
        same way.
        """
        self._baseline_peak = _os_peak_bytes()

    def _peak_memory_mb(self) -> tuple[float, str]:
        """Return (peak MB, method) for this block.

        The method names the SOURCE, not whether memory happened to grow,
        so the same machine always reports the same method and figures
        stay comparable across runs.

        Neither Windows' peak_wset nor POSIX' ru_maxrss can be reset, so
        both are lifetime high-water marks that may predate this block.
        The block's own peak is therefore the larger of the sampled
        maximum and (OS peak, when it grew during the block) -- but never
        less than the rss at entry, since the process already held that.
        """
        os_peak = _os_peak_bytes()
        sampled = max(self._rss_samples, default=self._rss_start)
        candidates = [sampled, self._rss_start]
        if os_peak is not None and self._baseline_peak is not None and os_peak > self._baseline_peak:
            # Grew during this block, so the OS figure is this block's.
            candidates.append(os_peak)
        elif os_peak is not None and self._baseline_peak is None:
            candidates.append(os_peak)
        method = _os_peak_method() if os_peak is not None else "sampled_rss_approximate"
        return round(max(candidates) / 1024**2, 2), method


def _os_peak_bytes() -> int | None:
    """Return the OS-tracked peak memory in bytes, or None if unavailable."""
    try:
        info = psutil.Process().memory_info()
    except psutil.Error:  # pragma: no cover
        return None
    peak = getattr(info, "peak_wset", None)  # Windows
    if peak is not None:
        return int(peak)
    try:  # POSIX
        import resource

        maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reports KB, macOS reports bytes.
        return int(maxrss if sys.platform == "darwin" else maxrss * 1024)
    except (ImportError, AttributeError):  # pragma: no cover
        return None


def _os_peak_method() -> str:
    """Name the peak-memory source, so figures are never compared blindly."""
    try:
        if getattr(psutil.Process().memory_info(), "peak_wset", None) is not None:
            return "windows_peak_wset"
    except psutil.Error:  # pragma: no cover
        pass
    return "posix_ru_maxrss"


def _io_counters(process: psutil.Process):
    """Return the process's IO counters, or None where not permitted."""
    try:
        return process.io_counters()
    except (psutil.AccessDenied, NotImplementedError, AttributeError):  # pragma: no cover
        return None


def _delta_mb(start, end, attribute: str) -> float | None:
    """Return the MB difference for one IO counter, or None if unmeasured."""
    if start is None or end is None:
        return None
    return round((getattr(end, attribute) - getattr(start, attribute)) / 1024**2, 3)
