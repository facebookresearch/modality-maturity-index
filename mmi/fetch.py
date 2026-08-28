# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree
"""Guarded remote fetching for URL-delivered artifacts.

**Off by default.** Nothing in the harness fetches a remote URL unless the run
explicitly enables it. Downloading is a convenience for inspecting what a model
pointed at; it is never part of the metric. A fetched artifact stays
URL-delivered, so enabling this can never turn URL evidence into native
evidence.

The threat model is that the URL comes from a language model, i.e. from an
untrusted source, and the harness may be running inside a network where
loopback and link-local addresses are interesting targets. Hence: HTTPS only,
every hop's resolved address checked, size and time bounded, and signed-URL
parameters redacted from anything persisted. Environment proxy configuration is
honored so retrieval works in networks that require an outbound proxy.
"""

from __future__ import annotations

import ipaddress
import socket
from dataclasses import dataclass
from urllib.parse import urlparse, urlunparse

import httpx

DEFAULT_MAX_BYTES = 1024 * 1024 * 1024
DEFAULT_TIMEOUT = 1200.0
DEFAULT_MAX_REDIRECTS = 5

#: Query parameters that carry signed-URL credentials. Stripped before any URL
#: is persisted or logged.
_CREDENTIAL_PARAMS = frozenset(
    {
        "access_token",
        "signature",
        "sig",
        "token",
        "key",
        "apikey",
        "api_key",
        "x-amz-signature",
        "x-amz-credential",
        "x-amz-security-token",
        "x-goog-signature",
        "x-goog-credential",
        "se",
        "sp",
        "sv",
        "srt",
        "st",
        "spr",
        "sr",
        "skoid",
    }
)


class FetchRejected(Exception):
    """The URL was refused before or during transfer."""


@dataclass(frozen=True)
class FetchResult:
    url: str
    final_url: str
    status_code: int
    mime_type: str
    data: bytes


def redact_url(url: str) -> str:
    """Strip credential-bearing query parameters and any userinfo."""
    try:
        parsed = urlparse(url)
    except ValueError:
        return "<unparseable-url>"

    netloc = parsed.netloc.rsplit("@", 1)[-1] if "@" in parsed.netloc else parsed.netloc

    kept = []
    redacted = False
    for pair in parsed.query.split("&"):
        if not pair:
            continue
        name = pair.split("=", 1)[0]
        if name.lower() in _CREDENTIAL_PARAMS:
            kept.append(f"{name}=REDACTED")
            redacted = True
        else:
            kept.append(pair)
    query = "&".join(kept) if (kept and redacted) else parsed.query

    return urlunparse((parsed.scheme, netloc, parsed.path, parsed.params, query, ""))


def _address_is_public(address: str) -> bool:
    ip = ipaddress.ip_address(address)
    return not (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def assert_url_is_fetchable(url: str) -> None:
    """Reject a URL before any connection is made.

    Resolution happens here and is re-checked on every redirect hop, which is
    what closes the DNS-rebinding window: a name that resolved public once is
    not trusted to stay public across a redirect.
    """
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise FetchRejected(f"only https URLs may be fetched, got {parsed.scheme!r}")
    host = parsed.hostname
    if not host:
        raise FetchRejected("URL has no host")

    try:
        infos = socket.getaddrinfo(host, parsed.port or 443, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise FetchRejected(f"could not resolve host: {exc}") from exc

    addresses = {info[4][0] for info in infos}
    if not addresses:
        raise FetchRejected("host resolved to no addresses")
    for address in addresses:
        if not _address_is_public(address):
            raise FetchRejected(f"host resolves to a non-public address ({address})")


async def fetch_url(
    url: str,
    *,
    max_bytes: int = DEFAULT_MAX_BYTES,
    timeout: float = DEFAULT_TIMEOUT,
    max_redirects: int = DEFAULT_MAX_REDIRECTS,
    allowed_mime_prefixes: tuple[str, ...] = (
        "image/",
        "audio/",
        "video/",
        "application/",
        "text/",
    ),
) -> FetchResult:
    """Fetch a remote artifact under strict limits.

    Redirects are followed manually so each hop can be revalidated; httpx's own
    redirect following would bypass the address check. Environment proxy
    configuration is honored to support networks that require outbound proxies.
    """
    current = url
    async with httpx.AsyncClient(
        timeout=timeout,
        follow_redirects=False,
        trust_env=True,
    ) as client:
        for _ in range(max_redirects + 1):
            assert_url_is_fetchable(current)
            async with client.stream("GET", current) as response:
                if response.is_redirect:
                    location = response.headers.get("location")
                    if not location:
                        raise FetchRejected("redirect without a Location header")
                    current = str(response.url.join(location))
                    continue

                response.raise_for_status()
                mime_type = (
                    response.headers.get("content-type", "").split(";", 1)[0].strip()
                )
                if mime_type and not mime_type.startswith(allowed_mime_prefixes):
                    raise FetchRejected(f"disallowed content type {mime_type!r}")

                declared = response.headers.get("content-length")
                if declared and declared.isdigit() and int(declared) > max_bytes:
                    raise FetchRejected(
                        f"declared size {declared} exceeds {max_bytes} byte limit"
                    )

                chunks: list[bytes] = []
                total = 0
                async for chunk in response.aiter_bytes():
                    total += len(chunk)
                    if total > max_bytes:
                        raise FetchRejected(f"asset exceeds {max_bytes} byte limit")
                    chunks.append(chunk)

                return FetchResult(
                    url=redact_url(url),
                    final_url=redact_url(str(response.url)),
                    status_code=response.status_code,
                    mime_type=mime_type,
                    data=b"".join(chunks),
                )

    raise FetchRejected(f"too many redirects (>{max_redirects})")
