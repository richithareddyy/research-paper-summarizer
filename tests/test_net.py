import socket
from unittest import mock

import pytest
import requests

from core import net
from core.net import FetchError, check_url, fetch

PUBLIC = "93.184.216.34"


def resolve_to(*addresses):
    def fake(host, port, *args, **kwargs):
        return [(socket.AF_INET6 if ":" in a else socket.AF_INET, socket.SOCK_STREAM, 6, "", (a, port)) for a in addresses]
    return mock.patch.object(net.socket, "getaddrinfo", side_effect=fake)


class FakeResponse:
    def __init__(self, status=200, headers=None, chunks=(b"%PDF-1.7 data",), url=""):
        self.status_code, self.headers, self._chunks, self.url = status, headers or {}, chunks, url

    def iter_content(self, size):
        yield from self._chunks

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeSession:
    def __init__(self, *responses):
        self.responses, self.calls = list(responses), []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.responses.pop(0)


@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://example.org/a.pdf", "gopher://x", "example.org/a.pdf",
                                 "https://user:pw@example.org/a.pdf"])
def test_rejects_unsupported_urls(url):
    with resolve_to(PUBLIC), pytest.raises(FetchError):
        check_url(url)


@pytest.mark.parametrize("address", ["127.0.0.1", "10.0.0.5", "192.168.1.1", "172.16.0.1", "169.254.169.254",
                                     "100.64.0.1", "0.0.0.0", "::1", "fe80::1", "::ffff:127.0.0.1", "224.0.0.1"])
def test_rejects_non_public_destinations(address):
    with resolve_to(address), pytest.raises(FetchError, match="non-public"):
        check_url("https://papers.example.org/a.pdf")


def test_rejects_when_any_resolved_address_is_private():
    with resolve_to(PUBLIC, "10.1.2.3"), pytest.raises(FetchError):
        check_url("https://papers.example.org/a.pdf")


def test_follows_public_redirects_without_letting_requests_redirect():
    session = FakeSession(FakeResponse(302, {"Location": "/files/a.pdf"}), FakeResponse())
    with resolve_to(PUBLIC):
        result = fetch("https://papers.example.org/doi/1", max_bytes=1000, session=session)
    assert result.url == "https://papers.example.org/files/a.pdf"
    assert all(kwargs["allow_redirects"] is False for _, kwargs in session.calls)


def test_redirect_to_private_address_is_blocked():
    session = FakeSession(FakeResponse(302, {"Location": "http://metadata.internal/latest"}))

    def fake(host, port, *a, **k):
        address = "169.254.169.254" if host == "metadata.internal" else PUBLIC
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, port))]

    with mock.patch.object(net.socket, "getaddrinfo", side_effect=fake), pytest.raises(FetchError, match="non-public"):
        fetch("https://papers.example.org/a.pdf", max_bytes=1000, session=session)


def test_redirect_limit():
    session = FakeSession(*[FakeResponse(301, {"Location": f"/hop{i}"}) for i in range(10)])
    with resolve_to(PUBLIC), pytest.raises(FetchError, match="redirects"):
        fetch("https://papers.example.org/a", max_bytes=1000, session=session)


def test_declared_size_over_limit_is_rejected_before_reading():
    session = FakeSession(FakeResponse(headers={"Content-Length": "999999"}))
    with resolve_to(PUBLIC), pytest.raises(FetchError, match="larger"):
        fetch("https://papers.example.org/a.pdf", max_bytes=1000, session=session)


def test_streamed_size_over_limit_is_rejected():
    session = FakeSession(FakeResponse(chunks=[b"x" * 600, b"x" * 600]))
    with resolve_to(PUBLIC), pytest.raises(FetchError, match="larger"):
        fetch("https://papers.example.org/a.pdf", max_bytes=1000, session=session)


def test_overall_deadline_applies_to_slow_downloads():
    clock = iter([0, 0, 0, 100, 100])
    session = FakeSession(FakeResponse(chunks=[b"a", b"b"]))
    with resolve_to(PUBLIC), mock.patch.object(net.time, "monotonic", side_effect=lambda: next(clock)), \
            pytest.raises(FetchError, match="too long"):
        fetch("https://papers.example.org/a.pdf", max_bytes=1000, timeout=30, session=session)


def test_http_errors_carry_status():
    session = FakeSession(FakeResponse(404))
    with resolve_to(PUBLIC), pytest.raises(FetchError) as info:
        fetch("https://api.example.org/works/x", max_bytes=1000, session=session)
    assert info.value.status == 404


def test_connection_errors_become_fetch_errors():
    session = mock.Mock(get=mock.Mock(side_effect=requests.ConnectionError("offline")))
    with resolve_to(PUBLIC), pytest.raises(FetchError, match="offline"):
        fetch("https://papers.example.org/a.pdf", max_bytes=1000, session=session)


@pytest.mark.parametrize('destination,keep', [
    ('https://unrelated.example/paper', False),
    ('https://api.semanticscholar.org/next', True),
    ('http://api.semanticscholar.org/next', False),
])
def test_semantic_redirect_credentials(destination, keep):
    session = FakeSession(FakeResponse(302, {'Location': destination}), FakeResponse())
    headers = {'X-API-Key': 'placeholder-key',
               'User-Agent': 'ResearchPaperSummarizer/1.0 (mailto:placeholder@example.invalid)'}
    with resolve_to(PUBLIC):
        fetch('https://api.semanticscholar.org/start', max_bytes=1000,
              headers=headers, session=session)
    first, second = [kwargs['headers'] for _, kwargs in session.calls]
    assert first['X-API-Key'] == 'placeholder-key'
    assert ('X-API-Key' in second) is keep
    assert ('mailto:' in second['User-Agent']) is keep
    assert headers['X-API-Key'] == 'placeholder-key'


@pytest.mark.parametrize('host', ['api.crossref.org', 'export.arxiv.org', 'arxiv.org',
                                  'api.semanticscholar.org'])
def test_contact_redirect_to_unrelated_host(host):
    session = FakeSession(FakeResponse(302, {'Location': 'https://unrelated.example/pdf'}), FakeResponse())
    with resolve_to(PUBLIC):
        fetch(f'https://{host}/start', max_bytes=1000, session=session,
              headers={'User-Agent': 'ResearchPaperSummarizer/1.0 (mailto:placeholder@example.invalid)'})
    assert 'mailto:' in session.calls[0][1]['headers']['User-Agent']
    assert session.calls[1][1]['headers']['User-Agent'] == 'ResearchPaperSummarizer/1.0'


@pytest.mark.parametrize('url', ['https://unrelated.example/paper',
                                 'https://api.semanticscholar.org.evil.example/paper',
                                 'http://api.semanticscholar.org/paper'])
def test_unintended_initial_destination_has_no_credentials(url):
    session = FakeSession(FakeResponse())
    with resolve_to(PUBLIC):
        fetch(url, max_bytes=1000, session=session,
              headers={'x-api-key': 'placeholder-key',
                       'user-agent': 'ResearchPaperSummarizer/1.0 (mailto:placeholder@example.invalid)'})
    sent = session.calls[0][1]['headers']
    assert 'x-api-key' not in sent
    assert sent['user-agent'] == 'ResearchPaperSummarizer/1.0'
