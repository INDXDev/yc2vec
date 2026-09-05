"""Small shared helpers: hashing, stable ids, atomic IO, slugs, logging."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import subprocess
import unicodedata
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import orjson

_LOG_CONFIGURED = False


def setup_logging(verbose: bool = False) -> None:
    global _LOG_CONFIGURED
    if _LOG_CONFIGURED:
        return
    from rich.logging import RichHandler

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(message)s",
        datefmt="%H:%M:%S",
        handlers=[RichHandler(rich_tracebacks=True, show_path=verbose, markup=False)],
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    _LOG_CONFIGURED = True


def log(name: str) -> logging.Logger:
    return logging.getLogger(name)


def now() -> datetime:
    return datetime.now(UTC)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_hash(obj: Any) -> str:
    """Hash of a JSON-serialisable object with deterministic key ordering."""
    return sha256_bytes(orjson.dumps(obj, option=orjson.OPT_SORT_KEYS | orjson.OPT_NON_STR_KEYS))


_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(value: str, *, max_len: int = 64) -> str:
    """ASCII slug used for stable tag and term ids.

    Ids must never change when a display name is edited, so slugs are minted
    once at creation time and then stored, not recomputed.
    """
    norm = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    slug = _SLUG_RE.sub("-", norm.lower()).strip("-")
    return slug[:max_len].strip("-") or "unnamed"


def normalize_name(value: str) -> str:
    """Aggressive normalisation used for alias matching and dedup.

    Collapses case, punctuation, whitespace and a small set of English
    plural/joiner forms so that ``AI Agents``, ``ai-agent`` and ``AI agent``
    all collide.
    """
    norm = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    norm = norm.lower().replace("&", " and ")
    norm = re.sub(r"[^a-z0-9]+", " ", norm).strip()
    words = [
        w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w
        for w in norm.split()
    ]
    return " ".join(w for w in words if w not in {"the", "a", "an", "of", "for"})


def git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5, check=False
        )
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:
        return None


@contextmanager
def atomic_write(path: Path, mode: str = "wb") -> Iterator[Any]:
    """Write via a temp file and rename, so an interrupt never truncates a release."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    try:
        with tmp.open(mode) as fh:
            yield fh
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def write_json(path: Path, obj: Any, *, pretty: bool = False) -> None:
    """Deterministic JSON: sorted keys, no incidental whitespace variation."""
    opts = orjson.OPT_SORT_KEYS | orjson.OPT_NON_STR_KEYS
    if pretty:
        opts |= orjson.OPT_INDENT_2
    with atomic_write(path) as fh:
        fh.write(orjson.dumps(obj, option=opts, default=_json_default))


def _json_default(o: Any) -> Any:
    if isinstance(o, datetime):
        return o.isoformat()
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, set | frozenset):
        return sorted(o)
    raise TypeError(f"cannot serialise {type(o)!r}")


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return orjson.loads(path.read_bytes())


def write_jsonl(path: Path, rows: Iterable[Any]) -> int:
    n = 0
    with atomic_write(path) as fh:
        for row in rows:
            payload = row.model_dump(mode="json") if hasattr(row, "model_dump") else row
            fh.write(orjson.dumps(payload, option=orjson.OPT_SORT_KEYS, default=_json_default))
            fh.write(b"\n")
            n += 1
    return n


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if not path.exists():
        return
    with path.open("rb") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield orjson.loads(line)


def append_jsonl(path: Path, rows: Iterable[Any]) -> int:
    """Checkpoint helper: append-only so an interrupted stage keeps its progress."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("ab") as fh:
        for row in rows:
            payload = row.model_dump(mode="json") if hasattr(row, "model_dump") else row
            fh.write(orjson.dumps(payload, option=orjson.OPT_SORT_KEYS, default=_json_default))
            fh.write(b"\n")
            n += 1
        fh.flush()
        os.fsync(fh.fileno())
    return n


def chunked[T](items: list[T], size: int) -> Iterator[list[T]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def truncate(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def json_compact(obj: Any) -> str:
    return json.dumps(obj, separators=(",", ":"), sort_keys=True, default=str)
