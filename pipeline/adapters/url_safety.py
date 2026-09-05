"""URL safety checks for the optional website crawler.

The crawler follows URLs supplied by an upstream dataset, so it is an SSRF
surface. Every URL -- including every redirect target -- passes through
:func:`check_url` before a socket is opened. Failures are hard errors, never
warnings.
"""

from __future__ import annotations

import ipaddress
import socket
from dataclasses import dataclass
from urllib.parse import urlparse

#: Hostnames that must never be fetched regardless of DNS.
BLOCKED_HOSTNAMES = {
    "localhost",
    "localhost.localdomain",
    "metadata.google.internal",
    "metadata.goog",
    "instance-data",
}

#: Cloud instance-metadata endpoints, in addition to the private ranges below.
BLOCKED_NETWORKS = [
    ipaddress.ip_network("169.254.0.0/16"),  # link-local, incl. 169.254.169.254
    ipaddress.ip_network("fd00::/8"),  # unique local
    ipaddress.ip_network("fe80::/10"),  # link-local v6
]

ALLOWED_SCHEMES = {"http", "https"}


class UnsafeUrl(ValueError):
    pass


@dataclass(frozen=True)
class UrlCheck:
    url: str
    host: str
    addresses: tuple[str, ...]


def _address_is_safe(addr: str) -> bool:
    ip = ipaddress.ip_address(addr)
    if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast:
        return False
    if ip.is_reserved or ip.is_unspecified:
        return False
    return all(ip not in net for net in BLOCKED_NETWORKS)


def resolve_addresses(host: str) -> tuple[str, ...]:
    infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    return tuple(sorted({str(info[4][0]) for info in infos}))


def check_url(
    url: str,
    *,
    denylist_domains: tuple[str, ...] = (),
    resolver=resolve_addresses,
) -> UrlCheck:
    """Validate a URL, or raise :class:`UnsafeUrl`.

    ``resolver`` is injectable so tests can exercise DNS-rebinding style cases
    without touching the network.
    """
    parsed = urlparse(url)
    if parsed.scheme.lower() not in ALLOWED_SCHEMES:
        raise UnsafeUrl(f"scheme {parsed.scheme!r} is not allowed")
    host = (parsed.hostname or "").lower().rstrip(".")
    if not host:
        raise UnsafeUrl("missing hostname")
    if host in BLOCKED_HOSTNAMES:
        raise UnsafeUrl(f"hostname {host!r} is blocked")
    if parsed.port is not None and parsed.port not in (80, 443):
        raise UnsafeUrl(f"port {parsed.port} is not allowed")
    for domain in denylist_domains:
        d = domain.lower().lstrip(".")
        if host == d or host.endswith("." + d):
            raise UnsafeUrl(f"host {host!r} is on the project denylist")

    # A bare IP literal is checked directly; a name is resolved and *every*
    # answer must be safe, so a split-horizon record cannot slip through.
    try:
        ipaddress.ip_address(host)
        addresses: tuple[str, ...] = (host,)
    except ValueError:
        try:
            addresses = resolver(host)
        except OSError as exc:
            raise UnsafeUrl(f"cannot resolve {host!r}: {exc}") from exc
    if not addresses:
        raise UnsafeUrl(f"{host!r} resolved to no addresses")
    for addr in addresses:
        if not _address_is_safe(addr):
            raise UnsafeUrl(f"{host!r} resolves to non-public address {addr}")
    return UrlCheck(url=url, host=host, addresses=addresses)
