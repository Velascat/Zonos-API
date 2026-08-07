"""Lightweight integrity checks for cached/deployed model files.

Cheap, load-free structural validation so a sync-conflict, aborted download, or
zeroed file is rejected before it's deployed into the synced ``models/`` tree
(the full Dac/backbone load is the final backstop).
"""
from __future__ import annotations

import json
from pathlib import Path


def validate_cached_file(path: Path) -> bool:
    """True if ``path`` looks structurally intact.

    - any file: must exist and be non-empty
    - ``.json``: must parse as JSON
    - ``.safetensors``: must carry a sane little-endian header-length prefix
      (first 8 bytes) that fits within the file
    """
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        return False
    suffix = path.suffix
    if suffix == ".json":
        try:
            with path.open("r", encoding="utf-8") as handle:
                json.load(handle)
            return True
        except (OSError, ValueError):
            return False
    if suffix == ".safetensors":
        try:
            size = path.stat().st_size
            with path.open("rb") as handle:
                head = handle.read(8)
            if len(head) < 8:
                return False
            header_len = int.from_bytes(head, "little")
            return 0 < header_len <= size - 8
        except OSError:
            return False
    return True


def sweep_stale_partials(
    cache_root: Path | str | None = None,
    *,
    max_age_s: float = 3600.0,
) -> list[str]:
    """Delete download partials old enough to be wreckage, not work.

    ``hf_hub_download`` stages into ``<blob>.incomplete`` and RESUMES an
    existing partial on retry — so a corrupt partial can wedge every retry
    forever. That is the 2026-08-03/04 VideoFoundry outage: the Zonos
    download sat dead for a day until a human deleted the ``.incomplete``
    by hand. This automates exactly that manual step at service startup.

    The age guard keeps a live concurrent download safe: anything younger
    than ``max_age_s`` is presumed active and left alone. Lock files under
    ``.locks/`` for a removed partial's repo are cleared so the next
    attempt starts unwedged. Never raises; returns the removed paths.
    """
    import os
    import time

    removed: list[str] = []
    try:
        root = Path(
            cache_root
            or os.environ.get("HF_HUB_CACHE")
            or os.environ.get("HUGGINGFACE_HUB_CACHE")
            or (Path.home() / ".cache" / "huggingface" / "hub")
        )
        if not root.is_dir():
            return removed
        now = time.time()
        for partial in root.rglob("*.incomplete"):
            try:
                if now - partial.stat().st_mtime < max_age_s:
                    continue
                partial.unlink()
                removed.append(str(partial))
                # blobs/<sha>.incomplete → models--org--name is 2 levels up;
                # clear its locks so the retry does not wait on a dead lock.
                repo_dir = partial.parent.parent
                lock_dir = root / ".locks" / repo_dir.name
                if lock_dir.is_dir():
                    for lock in lock_dir.glob("*.lock"):
                        try:
                            lock.unlink()
                        except OSError:
                            pass
            except OSError:
                continue
    except Exception:
        pass
    return removed
