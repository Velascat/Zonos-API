"""Tests for the lightweight cached-file integrity checks."""

from __future__ import annotations

from zonos.integrity import validate_cached_file


def _safetensors(path, header_len: int, total: int):
    """Write a fake .safetensors: 8-byte LE header length + padding to `total`."""
    data = header_len.to_bytes(8, "little") + b"x" * max(0, total - 8)
    path.write_bytes(data)
    return path


def test_empty_file_is_invalid(tmp_path):
    p = tmp_path / "model.safetensors"
    p.write_bytes(b"")
    assert validate_cached_file(p) is False


def test_missing_file_is_invalid(tmp_path):
    assert validate_cached_file(tmp_path / "nope.json") is False


def test_valid_json(tmp_path):
    p = tmp_path / "config.json"
    p.write_text('{"a": 1}')
    assert validate_cached_file(p) is True


def test_corrupt_json(tmp_path):
    p = tmp_path / "config.json"
    p.write_text("{not json")
    assert validate_cached_file(p) is False


def test_valid_safetensors_header(tmp_path):
    # header_len=2 ('{}'), total 10 → 0 < 2 <= 10-8
    p = _safetensors(tmp_path / "model.safetensors", header_len=2, total=10)
    assert validate_cached_file(p) is True


def test_truncated_safetensors_header_overflows_file(tmp_path):
    # header claims 9000 bytes but file is tiny → invalid
    p = _safetensors(tmp_path / "model.safetensors", header_len=9000, total=32)
    assert validate_cached_file(p) is False


def test_zeroed_safetensors_header(tmp_path):
    p = _safetensors(tmp_path / "model.safetensors", header_len=0, total=32)
    assert validate_cached_file(p) is False


def test_other_nonempty_file_ok(tmp_path):
    p = tmp_path / "vocab.txt"
    p.write_text("hello")
    assert validate_cached_file(p) is True


class TestSweepStalePartials:
    def _cache(self, tmp_path):
        repo = tmp_path / "models--Zyphra--Zonos-v0.1-transformer"
        blobs = repo / "blobs"
        blobs.mkdir(parents=True)
        locks = tmp_path / ".locks" / repo.name
        locks.mkdir(parents=True)
        return blobs, locks

    def test_old_partial_removed_with_its_locks(self, tmp_path):
        import os

        from zonos.integrity import sweep_stale_partials

        blobs, locks = self._cache(tmp_path)
        stale = blobs / "abc123.incomplete"
        stale.write_bytes(b"partial")
        old = 1_000_000_000.0
        os.utime(stale, (old, old))
        lock = locks / "model.safetensors.lock"
        lock.write_bytes(b"")

        removed = sweep_stale_partials(tmp_path, max_age_s=3600)

        assert [str(stale)] == removed
        assert not stale.exists()
        assert not lock.exists()

    def test_fresh_partial_is_presumed_active_and_kept(self, tmp_path):
        from zonos.integrity import sweep_stale_partials

        blobs, _ = self._cache(tmp_path)
        fresh = blobs / "abc123.incomplete"
        fresh.write_bytes(b"partial")  # mtime = now

        assert sweep_stale_partials(tmp_path, max_age_s=3600) == []
        assert fresh.exists()

    def test_intact_blobs_untouched(self, tmp_path):
        import os

        from zonos.integrity import sweep_stale_partials

        blobs, _ = self._cache(tmp_path)
        blob = blobs / "def456"
        blob.write_bytes(b"real weights")
        old = 1_000_000_000.0
        os.utime(blob, (old, old))

        sweep_stale_partials(tmp_path, max_age_s=3600)
        assert blob.exists()

    def test_missing_cache_root_is_a_noop(self, tmp_path):
        from zonos.integrity import sweep_stale_partials

        assert sweep_stale_partials(tmp_path / "nope") == []
