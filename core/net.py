"""Outbound HTTP for fetching papers and metadata.

Only http(s) URLs that resolve to public addresses are fetched, redirects are followed manually so each
hop is checked, and every request has a byte limit and an overall deadline.

Known gap: the address is checked when the URL is resolved, and requests resolves it again when it
connects, so a DNS server that answers differently the second time (DNS rebinding) is not stopped.
"""
import ipaddress
import json
import socket
import time
from dataclasses import dataclass
from urllib.parse import urljoin, urlsplit

import requests

MAX_REDIRECTS = 5
CONTACT_HOSTS = {"api.crossref.org", "export.arxiv.org", "arxiv.org", "api.semanticscholar.org"}


class FetchError(Exception):
    def __init__(self, message, status=None):
        super().__init__(message)
        self.status = status


@dataclass
class FetchResult:
    url: str
    status: int
    content_type: str
    content: bytes

    def json(self):
        return json.loads(self.content)


def _is_public(address):
    ip = ipaddress.ip_address(address.split("%")[0])
    if ip.version == 6 and ip.ipv4_mapped:
        ip = ip.ipv4_mapped
    return ip.is_global and not ip.is_multicast


def check_url(url):
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https"):
        raise FetchError(f"Unsupported URL scheme: {parts.scheme or '(none)'}")
    if parts.username or parts.password:
        raise FetchError("URLs with embedded credentials are not allowed")
    host = parts.hostname
    if not host:
        raise FetchError("URL has no host")
    try:
        port = parts.port or (443 if parts.scheme == "https" else 80)
    except ValueError as e:
        raise FetchError("URL has an invalid port") from e
    try:
        infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror as e:
        raise FetchError(f"Could not resolve {host}") from e
    blocked = [info[4][0] for info in infos if not _is_public(info[4][0])]
    if blocked or not infos:
        raise FetchError(f"{host} resolves to a non-public address; not fetching it")


def fetch(url, *, max_bytes, timeout=30, headers=None, params=None, session=requests):
    deadline = time.monotonic() + timeout
    original = urlsplit(url)
    for _ in range(MAX_REDIRECTS + 1):
        check_url(url)
        destination = urlsplit(url)
        same_origin = (destination.scheme, destination.hostname) == (original.scheme, original.hostname)
        if headers is not None:
            headers = dict(headers)
            for name in list(headers):
                if name.lower() == "x-api-key" and not (
                    same_origin and destination.scheme == "https"
                    and destination.hostname == "api.semanticscholar.org"
                ):
                    del headers[name]
                elif name.lower() == "user-agent" and not (
                    same_origin and destination.scheme == "https" and destination.hostname in CONTACT_HOSTS
                ):
                    headers[name] = "ResearchPaperSummarizer/1.0"
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise FetchError("The request took too long")
        try:
            response = session.get(url, headers=headers, params=params, stream=True,
                                   allow_redirects=False, timeout=(min(10, remaining), remaining))
        except requests.RequestException as e:
            raise FetchError(f"Request failed: {e}") from e

        with response:
            location = response.headers.get("Location")
            if response.status_code in (301, 302, 303, 307, 308) and location:
                url, params = urljoin(url, location), None
                continue
            if response.status_code >= 400:
                raise FetchError(f"The server returned HTTP {response.status_code}", status=response.status_code)

            declared = response.headers.get("Content-Length", "")
            if declared.isdigit() and int(declared) > max_bytes:
                raise FetchError(f"The file is larger than {max_bytes // (1024 * 1024)} MB")
            chunks, size = [], 0
            try:
                for chunk in response.iter_content(65536):
                    size += len(chunk)
                    if size > max_bytes:
                        raise FetchError(f"The file is larger than {max_bytes // (1024 * 1024)} MB")
                    if time.monotonic() > deadline:
                        raise FetchError("The download took too long")
                    chunks.append(chunk)
            except requests.RequestException as e:
                raise FetchError(f"Download interrupted: {e}") from e
            return FetchResult(url, response.status_code, response.headers.get("Content-Type", ""), b"".join(chunks))

    raise FetchError("Too many redirects")
