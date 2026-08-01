"""Deprecated convenience wrapper for the Conda development environment."""

from __future__ import annotations

import shutil
import subprocess
import sys
import warnings


def _run(*command: str) -> None:
    subprocess.run(command, check=True)


def main() -> None:
    """Create/update ``crienet_env`` and install the local checkout.

    This wrapper is retained for the 0.2 development cycle only.  Prefer the
    commands documented in README.md.
    """
    warnings.warn(
        "setup_env.py is deprecated; use `conda env update -f environment.yml "
        "--prune` followed by `conda run -n crienet_env python -m pip install "
        "-e .`.",
        FutureWarning,
        stacklevel=2,
    )
    conda = shutil.which("conda")
    if conda is None:
        raise RuntimeError("Conda is not available on PATH")

    _run(conda, "env", "update", "--file", "environment.yml", "--prune")
    _run(
        conda,
        "run",
        "--name",
        "crienet_env",
        "python",
        "-m",
        "pip",
        "install",
        "-e",
        ".",
    )
    _run(
        conda,
        "run",
        "--name",
        "crienet_env",
        "python",
        "-m",
        "ipykernel",
        "install",
        "--user",
        "--name=crienet_env",
        "--display-name=Python (crienet_env)",
    )


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, subprocess.CalledProcessError) as error:
        print(f"Environment setup failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
