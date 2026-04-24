from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BLOCKED_DIRS = ("data",)


class DataBoundaryError(RuntimeError):
    """Raised when private data is configured inside the repository."""


def get_data_root() -> Path:
    """Return the external data root defined by HTS_DATA_ROOT.

    The returned path must exist outside the repository workspace. This is the
    main privacy boundary for working with NDA-protected cafeteria data.
    """

    raw_path = os.getenv("HTS_DATA_ROOT")
    if not raw_path:
        raise DataBoundaryError(
            "HTS_DATA_ROOT is not set. Point it to a private directory outside the repository."
        )

    data_root = Path(raw_path).expanduser().resolve()

    if not data_root.exists():
        raise DataBoundaryError(f"Configured data root does not exist: {data_root}")

    try:
        data_root.relative_to(REPO_ROOT)
    except ValueError:
        pass
    else:
        raise DataBoundaryError(
            f"Configured data root must be outside the repository: {data_root}"
        )

    return data_root


def assert_repo_contains_no_data() -> None:
    """Fail fast if someone places data files inside blocked repository folders."""

    blocked_extensions = {
        ".csv",
        ".tsv",
        ".xlsx",
        ".xls",
        ".parquet",
        ".feather",
        ".jsonl",
    }

    for folder_name in DEFAULT_BLOCKED_DIRS:
        blocked_dir = REPO_ROOT / folder_name
        if not blocked_dir.exists():
            continue

        for path in blocked_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in blocked_extensions:
                raise DataBoundaryError(
                    f"Private data file found inside repository: {path}"
                )
