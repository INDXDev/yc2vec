"""Ollama protocol handling: schema validation, bounded repair, caching.

CI never talks to a real model. The HTTP layer is mocked so the client's
contract — validate, repair a bounded number of times, then fail loudly — is
tested deterministically.
"""

from __future__ import annotations

import json

import httpx
import pytest

from pipeline.config import load_config
from pipeline.ollama import ModelNotInstalled, OllamaClient, OllamaError
from pipeline.store import Store

SCHEMA = {
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": ["yes", "no"]},
        "confidence": {"type": "number"},
    },
    "required": ["decision", "confidence"],
}


class FakeTransport(httpx.AsyncBaseTransport):
    """Serves a scripted sequence of `/api/generate` responses."""

    def __init__(self, responses: list[str], tags: list[dict] | None = None) -> None:
        self.responses = list(responses)
        self.tags = tags if tags is not None else [{"name": "test-chat", "digest": "abc123"}]
        self.calls: list[dict] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": self.tags})
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "test"})
        if request.url.path == "/api/embed":
            body = json.loads(request.content)
            n = len(body["input"])
            return httpx.Response(200, json={"embeddings": [[0.1, 0.2, 0.3] for _ in range(n)]})
        self.calls.append(json.loads(request.content))
        if not self.responses:
            return httpx.Response(500, text="exhausted")
        return httpx.Response(200, json={"response": self.responses.pop(0)})


async def client_with(transport: FakeTransport, store: Store) -> OllamaClient:
    config = load_config("balanced", data_dir=store.root, chat_model="test-chat")
    c = OllamaClient(config, store)
    await c.__aenter__()
    c._client = httpx.AsyncClient(base_url="http://test", transport=transport)
    return c


@pytest.mark.asyncio
async def test_valid_json_is_returned_and_cached(store):
    transport = FakeTransport(['{"decision":"yes","confidence":0.9}'])
    c = await client_with(transport, store)
    kwargs = dict(system="s", prompt="p", schema=SCHEMA, prompt_version="v1")
    assert await c.generate_json(**kwargs) == {"decision": "yes", "confidence": 0.9}
    # A second identical call is served from the cache: no further HTTP request.
    assert await c.generate_json(**kwargs) == {"decision": "yes", "confidence": 0.9}
    assert len(transport.calls) == 1
    await c.__aexit__()


@pytest.mark.asyncio
async def test_malformed_json_triggers_bounded_repair(store):
    transport = FakeTransport(
        [
            "not json at all",
            '{"decision":"maybe","confidence":2}',
            '{"decision":"no","confidence":0.2}',
        ]
    )
    c = await client_with(transport, store)
    result = await c.generate_json(system="s", prompt="p", schema=SCHEMA, prompt_version="v1")
    assert result == {"decision": "no", "confidence": 0.2}
    assert len(transport.calls) == 3
    # The repair prompt tells the model what was wrong rather than just retrying.
    assert "rejected" in transport.calls[1]["prompt"]
    assert "rejected" in transport.calls[2]["prompt"]
    await c.__aexit__()


@pytest.mark.asyncio
async def test_repair_is_bounded_then_fails_loudly(store):
    transport = FakeTransport(["nope"] * 10)
    c = await client_with(transport, store)
    with pytest.raises(OllamaError, match="schema-valid JSON"):
        await c.generate_json(system="s", prompt="p", schema=SCHEMA, prompt_version="v1")
    assert len(transport.calls) == 3  # max_retries, not unbounded
    await c.__aexit__()


@pytest.mark.asyncio
async def test_schema_violation_is_not_accepted(store):
    """A structurally valid object that breaks the schema must be rejected."""
    transport = FakeTransport(
        ['{"decision":"probably","confidence":0.5}', '{"decision":"yes","confidence":0.5}']
    )
    c = await client_with(transport, store)
    result = await c.generate_json(system="s", prompt="p", schema=SCHEMA, prompt_version="v1")
    assert result["decision"] == "yes"
    await c.__aexit__()


@pytest.mark.asyncio
async def test_missing_model_is_reported_never_substituted(store):
    transport = FakeTransport([], tags=[{"name": "some-other-model", "digest": "z"}])
    c = await client_with(transport, store)
    with pytest.raises(ModelNotInstalled) as exc:
        await c.resolve("test-chat")
    assert "some-other-model" in str(exc.value)
    assert "never substitutes" in str(exc.value)
    await c.__aexit__()


@pytest.mark.asyncio
async def test_embeddings_are_cached_per_text(store):
    transport = FakeTransport([])
    c = await client_with(transport, store)
    first = await c.embed(["alpha", "beta"])
    assert len(first) == 2 and len(first[0]) == 3
    # "alpha" is cached; only "gamma" needs the model.
    again = await c.embed(["alpha", "gamma"])
    assert len(again) == 2
    await c.__aexit__()


@pytest.mark.asyncio
async def test_offline_mode_refuses_to_call_the_model(store):
    config = load_config("balanced", data_dir=store.root)
    c = OllamaClient(config, store, offline=True)
    await c.__aenter__()
    with pytest.raises(OllamaError, match="offline"):
        await c.generate_json(system="s", prompt="p", schema=SCHEMA, prompt_version="v1")
    await c.__aexit__()


@pytest.mark.asyncio
async def test_prompt_version_participates_in_the_cache_key(store):
    transport = FakeTransport(
        ['{"decision":"yes","confidence":1}', '{"decision":"no","confidence":0}']
    )
    c = await client_with(transport, store)
    a = await c.generate_json(system="s", prompt="p", schema=SCHEMA, prompt_version="v1")
    b = await c.generate_json(system="s", prompt="p", schema=SCHEMA, prompt_version="v2")
    assert a != b  # a prompt change invalidates the cached answer
    await c.__aexit__()
