from __future__ import annotations

from pathlib import Path

from hts.config import assert_repo_contains_no_data, get_data_root


def resolve_private_dataset(*parts: str) -> Path:
    """Build a path under the external NDA data directory."""

    assert_repo_contains_no_data()
    return get_data_root().joinpath(*parts)
