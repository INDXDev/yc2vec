"""Content-addressed artifact store with a manifest/DAG.

Each stage declares the inputs it depends on (upstream artifact hashes, the
relevant config slice, prompt text and model identifiers). The store hashes
that declaration into a *stage key*. If the key matches the recorded key for a
completed stage, the stage is skipped. Any change -- code version, prompt,
model, config or upstream data -- produces a different key and reruns the
stage. Writes are atomic, so the previous release survives an interrupt.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from pipeline import PIPELINE_VERSION
from pipeline.util import atomic_write, log, now, read_json, sha256_file, stable_hash, write_json

LOG = log(__name__)


@dataclass
class StageRecord:
    stage: str
    key: str
    finished_at: str
    outputs: dict[str, str]
    counts: dict[str, int]


class Store:
    """Filesystem artifact store rooted at ``data/``."""

    def __init__(self, data_dir: Path) -> None:
        self.root = Path(data_dir)
        self.manifest_path = self.root / "manifest.json"
        for sub in ("raw", "normalized", "inferred", "cache", "public", "export", "runs"):
            (self.root / sub).mkdir(parents=True, exist_ok=True)
        self._manifest: dict[str, Any] = read_json(self.manifest_path, default=None) or {
            "pipeline_version": PIPELINE_VERSION,
            "stages": {},
        }

    # -- stage keys -----------------------------------------------------
    def stage_key(self, stage: str, inputs: dict[str, Any]) -> str:
        return stable_hash({"stage": stage, "pipeline_version": PIPELINE_VERSION, "inputs": inputs})

    def is_fresh(self, stage: str, key: str) -> bool:
        rec = self._manifest["stages"].get(stage)
        if not rec or rec.get("key") != key:
            return False
        # A recorded output that vanished from disk invalidates the stage.
        for rel in rec.get("outputs", {}):
            if not (self.root / rel).exists():
                LOG.debug("stage %s output %s missing; rerunning", stage, rel)
                return False
        return True

    def record(
        self,
        stage: str,
        key: str,
        outputs: list[Path] | None = None,
        counts: dict[str, int] | None = None,
    ) -> None:
        checksums: dict[str, str] = {}
        for p in outputs or []:
            if p.exists() and p.is_file():
                checksums[str(p.relative_to(self.root))] = sha256_file(p)
        self._manifest["stages"][stage] = {
            "key": key,
            "finished_at": now().isoformat(),
            "outputs": checksums,
            "counts": counts or {},
        }
        self.flush()

    def stage_output_hashes(self, stage: str) -> dict[str, str]:
        return dict(self._manifest["stages"].get(stage, {}).get("outputs", {}))

    def counts(self, stage: str) -> dict[str, int]:
        return dict(self._manifest["stages"].get(stage, {}).get("counts", {}))

    def invalidate(self, stages: list[str]) -> None:
        for s in stages:
            self._manifest["stages"].pop(s, None)
        self.flush()

    def flush(self) -> None:
        write_json(self.manifest_path, self._manifest, pretty=True)

    # -- paths ----------------------------------------------------------
    def path(self, *parts: str) -> Path:
        p = self.root.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    # -- generic blob cache ---------------------------------------------
    def cache_path(self, namespace: str, key: str, suffix: str = ".json") -> Path:
        return self.path("cache", namespace, key[:2], key + suffix)

    def cache_get(self, namespace: str, key: str) -> Any | None:
        return read_json(self.cache_path(namespace, key), default=None)

    def cache_put(self, namespace: str, key: str, value: Any) -> None:
        write_json(self.cache_path(namespace, key), value)

    def cache_put_bytes(self, namespace: str, key: str, data: bytes, suffix: str = ".bin") -> Path:
        p = self.cache_path(namespace, key, suffix)
        with atomic_write(p) as fh:
            fh.write(data)
        return p

    # -- run records ------------------------------------------------------
    def write_run(self, run: Any) -> Path:
        started: datetime | str = getattr(run, "started_at", "")
        stamp = started.strftime("%Y%m%dT%H%M%S") if isinstance(started, datetime) else "run"
        p = self.path("runs", f"{stamp}-{run.stage}-{run.run_id[:8]}.json")
        write_json(p, run.model_dump(mode="json"), pretty=True)
        return p
