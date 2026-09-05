"""Optional enrichment from a company's own public homepage.

Disabled by default. When enabled it is deliberately timid: robots.txt is
honoured, the crawler identifies itself with a contact URL, concurrency is
per-domain, responses are size-capped and content-type-checked, redirects are
re-validated against the SSRF rules, and page JavaScript is never executed.

Only extracted main text is stored, and only in the local cache -- raw page
bodies are never committed, because their redistribution rights are unknown.
"""

from __future__ import annotations

import asyncio
import time
import urllib.robotparser
from collections import defaultdict
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse

import httpx
from selectolax.parser import HTMLParser

from pipeline.adapters.base import AdapterResult
from pipeline.adapters.url_safety import UnsafeUrl, check_url
from pipeline.config import Config
from pipeline.models import WebSource
from pipeline.store import Store
from pipeline.util import log, now, sha256_text, stable_hash
from pipeline.versions import EXTRACTION_VERSION

LOG = log(__name__)

#: Structural noise removed before main-text extraction.
_STRIP_TAGS = (
    "script",
    "style",
    "noscript",
    "template",
    "svg",
    "iframe",
    "form",
    "nav",
    "header",
    "footer",
    "aside",
)
_NOISE_HINTS = ("cookie", "consent", "banner", "newsletter", "subscribe", "breadcrumb", "sr-only")


@dataclass
class PageText:
    url: str
    title: str
    text: str


class CompanyWebsiteAdapter:
    name = "company_website"
    enabled_by_default = False

    def __init__(self, config: Config, store: Store) -> None:
        self.config = config
        self.store = store
        self._domain_locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self._robots: dict[str, urllib.robotparser.RobotFileParser | None] = {}
        self._last_request: dict[str, float] = {}

    async def fetch(self, companies: list[tuple[str, str]]) -> AdapterResult:
        """``companies`` is a list of ``(company_id, website_url)`` pairs."""
        cfg = self.config.crawl
        if not cfg.enabled:
            return AdapterResult(self.name, meta={"skipped": "crawl disabled"})

        sem = asyncio.Semaphore(cfg.global_concurrency)
        headers = {
            "User-Agent": self.config.user_agent,
            "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9",
            "Accept-Language": "en",
        }
        results: list[WebSource] = []
        texts: dict[str, list[PageText]] = {}

        async with httpx.AsyncClient(
            timeout=cfg.timeout_s,
            headers=headers,
            follow_redirects=False,
            max_redirects=0,
        ) as client:

            async def one(company_id: str, url: str) -> None:
                async with sem:
                    sources, pages = await self._fetch_company(client, company_id, url)
                    results.extend(sources)
                    if pages:
                        texts[company_id] = pages

            await asyncio.gather(
                *(one(cid, url) for cid, url in companies), return_exceptions=False
            )

        return AdapterResult(self.name, records=results, meta={"texts": texts})

    # -- per company ------------------------------------------------------
    async def _fetch_company(
        self, client: httpx.AsyncClient, company_id: str, url: str
    ) -> tuple[list[WebSource], list[PageText]]:
        cfg = self.config.crawl
        sources: list[WebSource] = []
        pages: list[PageText] = []
        try:
            root = check_url(url, denylist_domains=cfg.denylist_domains)
        except UnsafeUrl as exc:
            return [self._failed(company_id, url, f"unsafe url: {exc}")], []

        allowed = await self._robots_allows(client, url)
        if not allowed:
            LOG.info("robots.txt disallows %s", url)
            return [self._failed(company_id, url, "robots.txt disallow", robots_allowed=False)], []

        targets = [url] + [
            urljoin(url, p) for p in cfg.candidate_paths[: max(0, cfg.max_pages_per_company - 1)]
        ]
        for target in targets[: cfg.max_pages_per_company]:
            source, page = await self._fetch_page(client, company_id, target, root.host)
            sources.append(source)
            if page is not None:
                pages.append(page)
        return sources, pages

    async def _fetch_page(
        self, client: httpx.AsyncClient, company_id: str, url: str, origin_host: str
    ) -> tuple[WebSource, PageText | None]:
        cfg = self.config.crawl
        cache_key = stable_hash(
            {"url": url, "extraction": EXTRACTION_VERSION, "ua": self.config.user_agent}
        )
        cached = self.store.cache_get("web", cache_key)
        if cached is not None:
            age_h = (now().timestamp() - cached.get("fetched_ts", 0)) / 3600
            if age_h < cfg.cache_ttl_hours:
                src = WebSource(**cached["source"])
                page = PageText(**cached["page"]) if cached.get("page") else None
                return src, page

        current = url
        for _ in range(cfg.max_redirects + 1):
            try:
                check = check_url(current, denylist_domains=cfg.denylist_domains)
            except UnsafeUrl as exc:
                return self._failed(company_id, url, f"unsafe redirect: {exc}"), None
            # Redirects must stay on the original site; an off-origin hop is a
            # different publisher whose terms we have not reviewed.
            if check.host != origin_host and not check.host.endswith("." + origin_host):
                return self._failed(company_id, url, f"off-origin redirect to {check.host}"), None

            await self._throttle(check.host)
            try:
                resp = await client.get(current)
            except httpx.HTTPError as exc:
                return self._failed(company_id, url, f"transport error: {exc}"), None

            if resp.status_code in (301, 302, 303, 307, 308):
                loc = resp.headers.get("location")
                if not loc:
                    return self._failed(company_id, url, "redirect without location"), None
                current = urljoin(current, loc)
                continue
            break
        else:
            return self._failed(company_id, url, "too many redirects"), None

        ctype = (resp.headers.get("content-type") or "").split(";")[0].strip().lower()
        if resp.status_code != 200 or ctype not in cfg.allowed_content_types:
            return (
                self._failed(company_id, url, f"status {resp.status_code} content-type {ctype!r}"),
                None,
            )

        body = resp.content[: cfg.max_bytes]
        html = body.decode(resp.encoding or "utf-8", errors="replace")
        extracted = extract_main_text(html, current)

        source = WebSource(
            company_id=company_id,
            url=url,
            final_url=str(resp.url),
            fetched_at=now(),
            status=resp.status_code,
            content_type=ctype,
            content_hash=sha256_text(extracted.text),
            byte_size=len(body),
            extraction_version=EXTRACTION_VERSION,
            ok=bool(extracted.text.strip()),
            robots_allowed=True,
        )
        page = extracted if extracted.text.strip() else None
        self.store.cache_put(
            "web",
            cache_key,
            {
                "fetched_ts": now().timestamp(),
                "source": source.model_dump(mode="json"),
                "page": page.__dict__ if page else None,
            },
        )
        return source, page

    def _failed(
        self, company_id: str, url: str, error: str, *, robots_allowed: bool = True
    ) -> WebSource:
        # A transient failure must not erase a previous good result; callers
        # keep the last successful WebSource for a company when ok is False.
        return WebSource(
            company_id=company_id,
            url=url,
            final_url=url,
            fetched_at=now(),
            status=0,
            extraction_version=EXTRACTION_VERSION,
            ok=False,
            error=error,
            robots_allowed=robots_allowed,
        )

    async def _throttle(self, host: str) -> None:
        async with self._domain_locks[host]:
            delay = self.config.crawl.request_delay_s
            elapsed = time.monotonic() - self._last_request.get(host, 0.0)
            if elapsed < delay:
                await asyncio.sleep(delay - elapsed)
            self._last_request[host] = time.monotonic()

    async def _robots_allows(self, client: httpx.AsyncClient, url: str) -> bool:
        if not self.config.crawl.respect_robots:
            return True
        parsed = urlparse(url)
        origin = f"{parsed.scheme}://{parsed.netloc}"
        if origin not in self._robots:
            rp: urllib.robotparser.RobotFileParser | None = urllib.robotparser.RobotFileParser()
            try:
                await self._throttle(parsed.hostname or origin)
                r = await client.get(urljoin(origin, "/robots.txt"))
                if r.status_code == 200:
                    rp.parse(r.text.splitlines())  # type: ignore[union-attr]
                else:
                    # No robots.txt means no stated restriction.
                    rp = None
            except httpx.HTTPError:
                rp = None
            self._robots[origin] = rp
        rp = self._robots[origin]
        if rp is None:
            return True
        return rp.can_fetch(self.config.user_agent, url)


def extract_main_text(html: str, url: str, *, max_chars: int = 6000) -> PageText:
    """Strip chrome and return readable main text. Never executes JavaScript."""
    tree = HTMLParser(html)
    title = ""
    if tree.head is not None:
        node = tree.head.css_first("title")
        if node is not None:
            title = " ".join((node.text() or "").split())[:200]

    for tag in _STRIP_TAGS:
        for node in tree.css(tag):
            node.decompose()
    for node in tree.css("[class],[id]"):
        attrs = f"{node.attributes.get('class') or ''} {node.attributes.get('id') or ''}".lower()
        if any(hint in attrs for hint in _NOISE_HINTS):
            node.decompose()

    root = tree.css_first("main") or tree.css_first("article") or tree.body
    raw = root.text(separator="\n") if root is not None else ""

    lines: list[str] = []
    seen: set[str] = set()
    for line in raw.splitlines():
        line = " ".join(line.split())
        # Single words and repeated nav labels carry no signal.
        if len(line) < 3 or line.lower() in seen:
            continue
        seen.add(line.lower())
        lines.append(line)
    text = "\n".join(lines)[:max_chars]
    return PageText(url=url, title=title, text=text)
