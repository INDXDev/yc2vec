"""SSRF and crawler-limit behaviour.

The crawler follows URLs supplied by an upstream dataset, so these are security
tests, not hygiene tests.
"""

from __future__ import annotations

import pytest

from pipeline.adapters.company_website import extract_main_text
from pipeline.adapters.url_safety import UnsafeUrl, check_url


def fake_resolver(mapping: dict[str, tuple[str, ...]]):
    def resolve(host: str) -> tuple[str, ...]:
        if host not in mapping:
            raise OSError("NXDOMAIN")
        return mapping[host]

    return resolve


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost/",
        "http://127.0.0.1/",
        "https://127.0.0.1:443/",
        "http://[::1]/",
        "http://10.0.0.5/",
        "http://192.168.1.1/",
        "http://172.16.0.1/",
        "http://169.254.169.254/latest/meta-data/",  # AWS instance metadata
        "http://metadata.google.internal/",
        "file:///etc/passwd",
        "gopher://example.com/",
        "ftp://example.com/",
        "http://example.com:22/",  # non-web port
        "http://0.0.0.0/",
    ],
)
def test_rejects_unsafe_urls(url):
    with pytest.raises(UnsafeUrl):
        check_url(url, resolver=fake_resolver({"example.com": ("93.184.216.34",)}))


def test_rejects_public_name_resolving_to_private_address():
    """A DNS-rebinding style record must be caught by address inspection."""
    with pytest.raises(UnsafeUrl, match="non-public address"):
        check_url("https://evil.example/", resolver=fake_resolver({"evil.example": ("10.1.2.3",)}))


def test_rejects_when_any_answer_is_private():
    """Split-horizon records must not slip through on the first good answer."""
    with pytest.raises(UnsafeUrl):
        check_url(
            "https://mixed.example/",
            resolver=fake_resolver({"mixed.example": ("93.184.216.34", "127.0.0.1")}),
        )


def test_denylist_blocks_domain_and_subdomains():
    resolver = fake_resolver(
        {"blocked.example": ("93.184.216.34",), "www.blocked.example": ("93.184.216.34",)}
    )
    for url in ("https://blocked.example/", "https://www.blocked.example/x"):
        with pytest.raises(UnsafeUrl, match="denylist"):
            check_url(url, denylist_domains=("blocked.example",), resolver=resolver)


def test_accepts_ordinary_public_url():
    check = check_url(
        "https://example.com/about", resolver=fake_resolver({"example.com": ("93.184.216.34",)})
    )
    assert check.host == "example.com"
    assert check.addresses == ("93.184.216.34",)


def test_extraction_strips_scripts_and_chrome():
    html = """
    <html><head><title>Acme — Widgets</title></head>
      <body>
        <nav>Home Products Pricing</nav>
        <div class="cookie-banner">We use cookies</div>
        <script>window.__DATA__ = 'secret'; fetch('/evil')</script>
        <style>.a{color:red}</style>
        <main><p>Acme builds industrial widgets for factory automation.</p></main>
        <footer>© Acme</footer>
      </body></html>
    """
    page = extract_main_text(html, "https://acme.example/")
    assert page.title == "Acme — Widgets"
    assert "industrial widgets" in page.text
    for leaked in (
        "secret",
        "fetch(",
        "color:red",
        "Home Products Pricing",
        "We use cookies",
        "© Acme",
    ):
        assert leaked not in page.text


def test_extraction_caps_length():
    html = "<html><body><main>" + "<p>filler sentence here.</p>" * 5000 + "</main></body></html>"
    page = extract_main_text(html, "https://acme.example/", max_chars=1200)
    assert len(page.text) <= 1200
