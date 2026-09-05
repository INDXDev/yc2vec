"""Ollama client.

Design notes:

* All generation is *structured*: we pass a JSON Schema through Ollama's
  ``format`` field and validate the response ourselves. Malformed output gets a
  bounded number of repair attempts and then fails loudly rather than being
  guessed at.
* Every call is content-addressed and cached, keyed by model, digest, prompt,
  schema, options and prompt version. Reruns are therefore free and the whole
  pipeline is resumable after an interrupt.
* The client never substitutes a model. If the configured model is not
  installed it raises and lists what *is* installed.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

import httpx
from jsonschema import Draft202012Validator

from pipeline.config import Config
from pipeline.fixture_model import FixtureBackend
from pipeline.store import Store
from pipeline.util import log, stable_hash

LOG = log(__name__)


class OllamaError(RuntimeError):
    pass


class ModelNotInstalled(OllamaError):
    def __init__(self, model: str, installed: list[str]) -> None:
        self.model = model
        self.installed = installed
        super().__init__(
            f"model {model!r} is not installed on this Ollama host.\n"
            f"Installed models: {', '.join(installed) or '(none)'}\n"
            f"Run `ollama pull {model}` or select an installed model explicitly with "
            f"--chat-model / --embedding-model (or YC2VEC_CHAT_MODEL / YC2VEC_EMBEDDING_MODEL). "
            f"YC2Vec never substitutes a different model on your behalf."
        )


@dataclass
class ModelInfo:
    name: str
    digest: str
    size_bytes: int
    parameter_size: str | None
    quantization: str | None
    context_length: int | None
    family: str | None


class OllamaClient:
    """Async client with caching, retries and schema validation."""

    def __init__(
        self, config: Config, store: Store | None = None, *, offline: bool = False
    ) -> None:
        self.config = config
        self.store = store
        #: In ``offline`` mode only cached responses are served; a cache miss is
        #: an error instead of a network call.
        self.offline = offline
        #: The ``fixture`` profile swaps in a deterministic backend so CI can run
        #: the whole pipeline with no model download and no network.
        self.fixture: FixtureBackend | None = (
            FixtureBackend() if config.profile == "fixture" else None
        )
        self._client: httpx.AsyncClient | None = None
        self._digests: dict[str, str] = {}

    async def __aenter__(self) -> OllamaClient:
        if self.fixture is not None:
            return self
        self._client = httpx.AsyncClient(
            base_url=self.config.ollama_host,
            timeout=httpx.Timeout(self.config.models.request_timeout_s),
        )
        return self

    async def __aexit__(self, *exc: Any) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    @property
    def http(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("OllamaClient must be used as an async context manager")
        return self._client

    # -- introspection ---------------------------------------------------
    async def list_models(self) -> list[ModelInfo]:
        r = await self.http.get("/api/tags")
        r.raise_for_status()
        out = []
        for m in r.json().get("models", []):
            d = m.get("details") or {}
            out.append(
                ModelInfo(
                    name=m["name"],
                    digest=m.get("digest", ""),
                    size_bytes=m.get("size", 0),
                    parameter_size=d.get("parameter_size"),
                    quantization=d.get("quantization_level"),
                    context_length=d.get("context_length"),
                    family=d.get("family"),
                )
            )
        return out

    async def resolve(self, model: str) -> str:
        """Return the model's digest, or raise listing the installed models."""
        if self.fixture is not None:
            return "fixture-" + stable_hash({"model": model})[:12]
        if model in self._digests:
            return self._digests[model]
        installed = await self.list_models()
        for m in installed:
            if m.name == model:
                self._digests[model] = m.digest
                return m.digest
        raise ModelNotInstalled(model, [m.name for m in installed])

    async def ping(self) -> bool:
        if self.fixture is not None:
            return True
        try:
            r = await self.http.get("/api/version", timeout=5.0)
            return r.status_code == 200
        except Exception:
            return False

    # -- generation ------------------------------------------------------
    async def generate_json(
        self,
        *,
        system: str,
        prompt: str,
        schema: dict[str, Any],
        prompt_version: str,
        model: str | None = None,
        num_predict: int | None = None,
        cache_namespace: str = "generate",
    ) -> dict[str, Any]:
        """Generate a schema-valid JSON object, using the cache when possible."""
        mc = self.config.models
        model = model or mc.chat_model
        options = {
            "temperature": mc.temperature,
            "seed": mc.seed,
            "num_ctx": mc.num_ctx,
            "num_predict": num_predict or mc.num_predict,
        }
        key = stable_hash(
            {
                "model": model,
                "system": system,
                "prompt": prompt,
                "schema": schema,
                "options": options,
                "think": mc.think,
                "prompt_version": prompt_version,
            }
        )
        if self.store is not None:
            cached = self.store.cache_get(cache_namespace, key)
            if cached is not None:
                return cached
        if self.fixture is not None:
            # Validate the fixture's output too: a fixture that drifts out of
            # schema should fail here rather than corrupt the dataset.
            parsed = self.fixture.generate(
                system=system, prompt=prompt, schema=schema, namespace=cache_namespace
            )
            Draft202012Validator(schema).validate(parsed)
            if self.store is not None:
                self.store.cache_put(cache_namespace, key, parsed)
            return parsed
        if self.offline:
            raise OllamaError(
                f"offline mode: no cached response for {cache_namespace}/{key[:12]}. "
                "Run with a live Ollama host, or use the committed fixture cache."
            )

        validator = Draft202012Validator(schema)
        last_error = ""
        attempt_prompt = prompt
        for attempt in range(mc.max_retries):
            body = {
                "model": model,
                "system": system,
                "prompt": attempt_prompt,
                "stream": False,
                "think": mc.think,
                "format": schema,
                "options": options,
            }
            try:
                r = await self.http.post("/api/generate", json=body)
                r.raise_for_status()
            except httpx.HTTPError as exc:
                last_error = f"transport error: {exc}"
                await asyncio.sleep(min(2**attempt, 8))
                continue

            text = (r.json().get("response") or "").strip()
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError as exc:
                last_error = f"invalid JSON ({exc})"
            else:
                errors = sorted(validator.iter_errors(parsed), key=lambda e: list(e.path))
                if not errors:
                    if self.store is not None:
                        self.store.cache_put(cache_namespace, key, parsed)
                    return parsed
                last_error = "; ".join(
                    f"{'/'.join(map(str, e.path)) or '<root>'}: {e.message}" for e in errors[:4]
                )

            LOG.debug("repair attempt %d for %s: %s", attempt + 1, cache_namespace, last_error)
            attempt_prompt = (
                f"{prompt}\n\n"
                f"Your previous reply was rejected: {last_error}.\n"
                f"Reply again with a single JSON object that satisfies the schema exactly. "
                f"Output JSON only."
            )
        raise OllamaError(
            f"{cache_namespace}: model failed to produce schema-valid JSON: {last_error}"
        )

    # -- embeddings ------------------------------------------------------
    async def embed(self, texts: list[str], *, model: str | None = None) -> list[list[float]]:
        """Embed a batch of documents. Cached per (model, text)."""
        model = model or self.config.models.embedding_model
        results: list[list[float] | None] = [None] * len(texts)
        pending: list[tuple[int, str, str]] = []
        for i, t in enumerate(texts):
            key = stable_hash({"model": model, "text": t})
            if self.store is not None:
                hit = self.store.cache_get("embed", key)
                if hit is not None:
                    results[i] = hit
                    continue
            pending.append((i, t, key))

        if pending:
            if self.fixture is not None:
                for (i, _text, key), vec in zip(
                    pending, self.fixture.embed([t for _, t, _ in pending]), strict=True
                ):
                    results[i] = vec
                    if self.store is not None:
                        self.store.cache_put("embed", key, vec)
                return [v for v in results if v is not None]
            if self.offline:
                raise OllamaError(
                    f"offline mode: {len(pending)} embedding(s) missing from cache for model {model!r}."
                )
            r = await self.http.post(
                "/api/embed",
                json={"model": model, "input": [t for _, t, _ in pending], "truncate": True},
            )
            r.raise_for_status()
            vectors = r.json().get("embeddings") or []
            if len(vectors) != len(pending):
                raise OllamaError(f"expected {len(pending)} embeddings, got {len(vectors)}")
            for (i, _, key), vec in zip(pending, vectors, strict=True):
                results[i] = vec
                if self.store is not None:
                    self.store.cache_put("embed", key, vec)

        return [v for v in results if v is not None]
