"""Config loading and run metadata."""

import datetime
import json
import pathlib
import platform
import subprocess
from typing import Any

import yaml


def load_config(path: str | pathlib.Path) -> dict[str, Any]:
    with open(path) as f:
        config: dict[str, Any] = yaml.safe_load(f)
    return config


def git_commit() -> str | None:
    """Returns HEAD's hash, or None outside a git repo."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return out.stdout.strip()


def write_run_info(
    out_dir: pathlib.Path,
    config: dict[str, Any],
    seed: int | None = None,
    device: str | None = None,
) -> None:
    """Writes out_dir/run_info.json: config, commit, seed, device, host, start time."""
    out_dir.mkdir(parents=True, exist_ok=True)
    info = {
        "config": config,
        "git_commit": git_commit(),
        "seed": seed,
        "device": device,
        "host": platform.node(),
        "started": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    with open(out_dir / "run_info.json", "w") as f:
        json.dump(info, f, indent=2)
