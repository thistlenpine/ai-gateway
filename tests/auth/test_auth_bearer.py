import pytest

from ai_gateway.auth import extract_bearer
from ai_gateway.errors import AuthenticationError, AuthFailure


def test_ok_case_insensitive_scheme() -> None:
    assert extract_bearer("Bearer abc.def.ghi") == "abc.def.ghi"
    assert extract_bearer("bEaReR tok") == "tok"


@pytest.mark.parametrize("value", [None, ""])
def test_missing(value: str | None) -> None:
    with pytest.raises(AuthenticationError) as ei:
        extract_bearer(value)
    assert ei.value.failure is AuthFailure.MISSING_TOKEN
    assert ei.value.headers["WWW-Authenticate"] == "Bearer"


@pytest.mark.parametrize(
    "value", ["Bearer", "Bearer ", "Basic abc", "Bearer a b", "Bearer  a", "tokenonly", " Bearer a"]
)
def test_malformed(value: str) -> None:
    with pytest.raises(AuthenticationError) as ei:
        extract_bearer(value)
    assert ei.value.failure is AuthFailure.MALFORMED
    assert "abc" not in ei.value.message
