from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from pipeline.config import Config, load_config
from pipeline.models import CompanyRaw
from pipeline.store import Store
from pipeline.util import stable_hash

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_payloads() -> list[dict]:
    return json.loads((FIXTURES / "companies_sample.json").read_text())


@pytest.fixture
def sample_raws(sample_payloads: list[dict]) -> list[CompanyRaw]:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    return [
        CompanyRaw(
            company_id=f"ycoss:{p['id']}",
            source_url="https://yc-oss.github.io/api/companies/all.json",
            retrieved_at=now,
            source_last_updated="2026-01-01T00:00:00.000Z",
            payload=p,
            content_hash=stable_hash(p),
        )
        for p in sample_payloads
    ]


@pytest.fixture
def config(tmp_path: Path) -> Config:
    return load_config("fixture", data_dir=tmp_path / "data")


@pytest.fixture
def store(config: Config) -> Store:
    return Store(config.data_dir)
