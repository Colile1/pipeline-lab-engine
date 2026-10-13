# Confirms resource and environment capture against KNOWN-COST workloads,
# per PLAN.md step 4's verification: wall/cpu/peak-RSS must land within
# sane bounds and every environment field must be non-null (except the
# two that are legitimately absent, which are asserted as such).
import platform
import time

import numpy as np
import pytest

from piplab.provenance import (
    ResourceMonitor,
    ResourceUsage,
    capture_environment,
    git_sha,
    machine_fingerprint,
)
from piplab.provenance.environment import TRACKED_LIBRARIES

MB = 1024**2


# --- environment ----------------------------------------------------------

def test_environment_captures_every_machine_field():
    env = capture_environment()
    for field in ("hostname", "cpu_model", "cpu_arch", "os_name", "os_version",
                  "python_version", "python_implementation"):
        assert env[field], f"{field} must be captured, not blank"


def test_core_counts_and_ram_are_plausible():
    env = capture_environment()
    assert env["cpu_cores_physical"] >= 1
    assert env["cpu_cores_logical"] >= env["cpu_cores_physical"]
    assert env["ram_total_gb"] > 0.5


def test_the_libraries_that_can_change_a_result_are_versioned():
    env = capture_environment()
    # These four are required to run at all, so a missing version means
    # the environment is broken, not that the field is optional.
    for name in ("scikit_learn", "numpy", "pandas", "scipy"):
        assert env[f"version_{name}"], f"version_{name} must be recorded"
    assert all(f"version_{n.replace('-', '_')}" in env for n in TRACKED_LIBRARIES)


def test_git_sha_is_none_rather_than_a_placeholder_outside_a_repo(tmp_path):
    # Recording None is honest; a placeholder would imply a provenance
    # the record does not have.
    assert git_sha(tmp_path) is None


def test_machine_fingerprint_is_stable_and_identifies_the_machine():
    env = capture_environment()
    first = machine_fingerprint(env)
    assert first == machine_fingerprint(env)
    assert env["hostname"] in first
    assert str(env["cpu_cores_physical"]) in first


def test_machine_fingerprint_differs_when_the_machine_differs():
    env = capture_environment()
    other = {**env, "hostname": "some-other-box"}
    assert machine_fingerprint(env) != machine_fingerprint(other)


# --- resources, against known costs ---------------------------------------

def test_wall_time_matches_a_known_sleep():
    with ResourceMonitor(sample_interval=0.01) as monitor:
        time.sleep(0.30)
    usage = monitor.usage()
    assert 0.28 <= usage.wall_seconds <= 1.5, usage.wall_seconds


def test_cpu_time_tracks_real_work_and_not_sleeping():
    """A sleep burns wall time but almost no CPU; a busy loop burns both."""
    with ResourceMonitor(sample_interval=0.01) as idle:
        time.sleep(0.30)
    with ResourceMonitor(sample_interval=0.01) as busy:
        end = time.perf_counter() + 0.30
        while time.perf_counter() < end:
            sum(i * i for i in range(5000))

    # The invariant worth asserting is the SHAPE of the difference, not
    # an absolute ceiling on idle CPU. An idle block is never exactly
    # zero -- the monitor's own sampler thread costs system time, usually
    # ~0.015 s over 0.3 s but observed as high as 0.25 s when the machine
    # is loaded, which made an absolute threshold flaky. What is always
    # true: a busy loop burns USER time and a sleep does not.
    assert idle.usage().cpu_seconds_user < 0.05, "sleeping should burn no user CPU"
    assert busy.usage().cpu_seconds_user > 0.15, "a busy loop should burn user CPU"
    assert busy.usage().cpu_seconds_user > idle.usage().cpu_seconds_user * 3 + 0.1


def test_peak_memory_sees_a_known_allocation_that_was_already_freed():
    """The point of a high-water mark: it survives deallocation.

    tracemalloc is deliberately not used here — it only sees Python-level
    allocations and undercounts numpy/BLAS badly (measured 61 MB against
    the OS's 99 MB for the same array).
    """
    with ResourceMonitor(sample_interval=0.01) as monitor:
        before = monitor._process.memory_info().rss
        big = np.ones((20_000_000,), dtype=np.float64)  # ~153 MB
        _ = float(big.sum())
        del big
    usage = monitor.usage()
    # The allocation must be visible in the peak even though it is gone.
    assert usage.peak_rss_mb >= (before / MB) + 100, usage.peak_rss_mb
    # And rss must have come back down, proving peak != current.
    assert usage.rss_end_mb < usage.peak_rss_mb


def test_the_peak_memory_method_is_always_recorded():
    """Figures from different methods must never be compared blindly."""
    with ResourceMonitor(sample_interval=0.01) as monitor:
        time.sleep(0.05)
    method = monitor.usage().peak_rss_method
    assert method in ("windows_peak_wset", "posix_ru_maxrss", "sampled_rss_approximate")
    if platform.system() == "Windows":
        assert method == "windows_peak_wset", "Windows exposes a true OS peak"


def test_the_peak_method_names_the_source_not_whether_memory_grew():
    """The same machine must always report the same method.

    An earlier version fell back to "sampled" whenever memory did not
    grow during the block, so an idle run and a heavy run reported
    different methods on identical hardware -- making the figures look
    incomparable when they were not.
    """
    import numpy as np

    with ResourceMonitor(sample_interval=0.01) as idle:
        time.sleep(0.05)
    with ResourceMonitor(sample_interval=0.01) as heavy:
        block = np.ones((20_000_000,), dtype=np.float64)
        del block
    assert idle.usage().peak_rss_method == heavy.usage().peak_rss_method


def test_thread_count_and_samples_are_recorded():
    with ResourceMonitor(sample_interval=0.01) as monitor:
        time.sleep(0.20)
    usage = monitor.usage()
    assert usage.n_threads_peak >= 1
    assert usage.n_samples >= 1, "the sampler should have run at least once"


def test_parallel_efficiency_identifies_cpu_bound_work():
    """cpu_seconds/wall_seconds ~1 for single-threaded CPU-bound work.

    A low value on a long run means the process was blocked rather than
    computing — the signature of the paging this machine has shown — so
    this is worth having in the trend data.
    """
    with ResourceMonitor(sample_interval=0.01) as monitor:
        end = time.perf_counter() + 0.30
        while time.perf_counter() < end:
            sum(i * i for i in range(5000))
    efficiency = monitor.usage().parallel_efficiency
    assert efficiency is not None
    assert 0.5 <= efficiency <= 2.0, efficiency


def test_sleeping_shows_low_parallel_efficiency():
    """An idle block reads well below 1.0, but not 0.0.

    The floor is the monitor's own sampler thread. Measured 0.363 at a
    0.01 s interval; a coarser interval lowers it. The useful property is
    that it is far below the ~1.0 of CPU-bound work, which is what
    distinguishes "blocked on IO/paging" from "computing".
    """
    with ResourceMonitor(sample_interval=0.05) as monitor:
        time.sleep(0.30)
    efficiency = monitor.usage().parallel_efficiency
    assert efficiency < 0.5, efficiency


def test_disk_counters_are_numbers_or_honestly_none():
    with ResourceMonitor(sample_interval=0.01) as monitor:
        time.sleep(0.05)
    usage = monitor.usage()
    for value in (usage.disk_read_mb, usage.disk_write_mb):
        assert value is None or value >= 0


# --- contract -------------------------------------------------------------

def test_usage_before_the_block_finishes_raises_rather_than_half_reporting():
    monitor = ResourceMonitor()
    with pytest.raises(RuntimeError, match="before the block finished"):
        monitor.usage()


def test_an_exception_inside_the_block_still_propagates():
    """The monitor must never swallow a failure to report a tidy number."""
    with pytest.raises(ValueError, match="boom"):
        with ResourceMonitor(sample_interval=0.01):
            raise ValueError("boom")


def test_usage_serialises_flat_for_the_run_record():
    with ResourceMonitor(sample_interval=0.01) as monitor:
        time.sleep(0.05)
    record = monitor.usage().as_dict()
    assert isinstance(record, dict)
    for key in ("wall_seconds", "cpu_seconds_user", "peak_rss_mb", "peak_rss_method"):
        assert key in record
    # Every value must be a plain type SQLite can store.
    for key, value in record.items():
        assert value is None or isinstance(value, (int, float, str)), (key, type(value))


def test_usage_is_a_dataclass_with_the_documented_fields():
    fields = set(ResourceUsage.__dataclass_fields__)
    assert {"wall_seconds", "cpu_seconds_user", "cpu_seconds_system",
            "peak_rss_mb", "peak_rss_method", "n_threads_peak",
            "disk_read_mb", "disk_write_mb"} <= fields
