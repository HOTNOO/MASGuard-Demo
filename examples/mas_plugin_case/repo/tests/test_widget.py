from widget import normalize


def test_normalize_none():
    assert normalize(None) == "unknown"


def test_normalize_text():
    assert normalize("  HeLLo  ") == "hello"


if __name__ == "__main__":
    test_normalize_none()
    test_normalize_text()
