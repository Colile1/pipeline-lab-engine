# environment.py: capture the machine and software a run happened on.
#
# Resource measurements are only comparable WITHIN a machine, so the
# machine identity goes on every run row rather than being assumed from
# context. Library versions are recorded because a scikit-learn upgrade
# can change a default and therefore a result.
from __future__ import annotations

import platform
import subprocess
import sys
from pathlib import Path

import psutil

# Libraries whose version can change a result or a resource figure.
TRACKED_LIBRARIES = ("scikit-learn", "numpy", "pandas", "scipy", "optuna", "psutil")


def capture_environment() -> dict:
    """Describe the machine and software stack for the run record.

    Inputs: none.
    Output: a flat dict of typed, groupable fields (PLAN.md section 4's
    "Machine & environment" block). Every value is read from the running
    system; nothing is defaulted or inferred.
    """
    virtual = psutil.virtual_memory()
    freq = _cpu_frequency_mhz()
    return {
        "hostname": platform.node(),
        "cpu_model": platform.processor(),
        "cpu_arch": platform.machine(),
        "cpu_cores_physical": psutil.cpu_count(logical=False),
        "cpu_cores_logical": psutil.cpu_count(logical=True),
        "cpu_freq_max_mhz": freq,
        "ram_total_gb": round(virtual.total / 1024**3, 2),
        "os_name": platform.system(),
        "os_version": platform.version(),
        "os_release": platform.release(),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        **_library_versions(),
        "engine_git_sha": git_sha(),
    }


def _cpu_frequency_mhz() -> float | None:
    """Return the CPU's max clock in MHz, or None where unavailable.

    psutil cannot read this on every platform or VM, so None is recorded
    rather than a fabricated figure.
    """
    try:
        freq = psutil.cpu_freq()
    except (NotImplementedError, AttributeError):
        return None
    if freq is None:
        return None
    return float(freq.max) or float(freq.current) or None


def _library_versions() -> dict:
    """Return the installed version of each tracked library.

    Inputs: none.
    Output: dict keyed `version_<library>`, with None for any library
    that is not installed (optuna is optional: only Bayesian tuning
    needs it).
    """
    from importlib.metadata import PackageNotFoundError, version

    out = {}
    for name in TRACKED_LIBRARIES:
        key = f"version_{name.replace('-', '_')}"
        try:
            out[key] = version(name)
        except PackageNotFoundError:
            out[key] = None
    return out


def git_sha(repo_root: Path | None = None) -> str | None:
    """Return this engine's short git commit, or None outside a repo.

    Recorded so a result can be traced to the exact code that produced
    it. None is returned rather than a placeholder when the engine is not
    under version control, so the record never implies a provenance it
    does not have.
    Inputs: repo_root - directory to ask git about; defaults to the
    engine package's own root.
    Output: the abbreviated SHA, or None.
    """
    root = repo_root or Path(__file__).resolve().parents[3]
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10, check=True,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return None
    sha = result.stdout.strip()
    return sha or None


def machine_fingerprint(env: dict | None = None) -> str:
    """Return a short key identifying comparable machines.

    Two runs may only have their wall-time or memory compared when this
    matches. The Trends tab groups by it rather than pooling figures from
    different hardware.
    Inputs: env - a captured environment dict, or None to capture one.
    Output: a stable string, e.g. "Colile-10|AMD64|8c/12t|15.75GB".
    """
    env = env or capture_environment()
    return (
        f"{env['hostname']}|{env['cpu_arch']}"
        f"|{env['cpu_cores_physical']}c/{env['cpu_cores_logical']}t"
        f"|{env['ram_total_gb']}GB"
    )
