from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class AdapterResult:
    name: str
    records: list[Any] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


class Adapter(Protocol):
    name: str
    enabled_by_default: bool

    async def fetch(self, *args: Any, **kwargs: Any) -> AdapterResult: ...
