from widget import normalize


def test_normalize_none():
    assert normalize(None) == "unknown"

