from __future__ import annotations

import pytest
from obs_rig import API_KEY, BEARER

from ai_gateway.observability import schema
from ai_gateway.observability.redact import (
    REDACTED,
    is_sensitive_key,
    redact_mapping,
    scrub_text,
    scrub_url,
    strip_query,
)


@pytest.mark.parametrize(
    "text",
    [
        f"Authorization: Bearer {BEARER}",
        f"got header bearer {BEARER} from client",
        f"token {BEARER} rejected",
        f"x-api-key: {API_KEY}",
        f'{{"api_key": "{API_KEY}"}}',
        f"using key {API_KEY} for call",
        f"GET /v1/x?access_token={BEARER}&page=2",
        "password=hunter2-pass&user=bob",
    ],
)
def test_scrub_text_removes_secrets(text: str) -> None:
    out = scrub_text(text)
    assert BEARER not in out
    assert API_KEY not in out
    assert "hunter2-pass" not in out
    assert REDACTED in out


def test_scrub_text_keeps_innocent_text() -> None:
    msg = "request completed tokens=120 token_id=abc bearer of news status=200"
    assert scrub_text(msg) == msg


def test_scrub_url_drops_userinfo_and_query_credentials() -> None:
    out = scrub_url(f"https://user:pw@host/p?api_key={API_KEY}&q=hello&sig=abcd1234&access_token={BEARER}")
    assert "user:pw" not in out and API_KEY not in out and BEARER not in out and "abcd1234" not in out
    assert "q=hello" in out and out.startswith("https://[REDACTED]@host/p?")


def test_strip_query() -> None:
    assert strip_query("/a/b?token=x#frag") == "/a/b"


def test_key_based_redaction_covers_schema_keys_case_insensitively() -> None:
    for key in schema.REDACT_KEYS:
        assert is_sensitive_key(key)
        assert is_sensitive_key(key.upper())
    assert is_sensitive_key("X-Upstream-Token") and is_sensitive_key("db.password")
    assert not is_sensitive_key("token_id") and not is_sensitive_key(schema.A_LLM_TOKEN_TYPE)


def test_redact_mapping_nested() -> None:
    out = redact_mapping(
        {"headers": {"Authorization": f"Bearer {BEARER}", "accept": "x"}, "items": [{"api_key": "k"}], "n": 3}
    )
    assert out == {
        "headers": {"Authorization": REDACTED, "accept": "x"},
        "items": [{"api_key": REDACTED}],
        "n": 3,
    }
