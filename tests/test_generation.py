"""Self-check for per-claim citation verification (src/generation.py). Run directly: no framework, no Ollama call."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.generation import _split_claims, _verify_claims

CHUNKS = [
    {"id": "a", "text": "...", "source": "docs/a.md", "section": "Intro"},
    {"id": "b", "text": "...", "source": "docs/b.md", "section": "Setup"},
]


def test_split_claims_pairs_text_with_following_tag():
    claims, trailing = _split_claims(
        "First fact. [source: docs/a.md#Intro] Second fact. [source: docs/b.md#Setup]"
    )
    assert claims == [
        ("First fact.", "docs/a.md#Intro"),
        ("Second fact.", "docs/b.md#Setup"),
    ]
    assert trailing == ""


def test_split_claims_captures_trailing_uncited_text():
    _, trailing = _split_claims("Cited. [source: docs/a.md#Intro] Uncited closer.")
    assert trailing == "Uncited closer."


def test_verify_claims_keeps_valid_drops_hallucinated():
    answer = (
        "Real fact. [source: docs/a.md#Intro] "
        "Made-up fact. [source: docs/nonexistent.md#Nope]"
    )
    filtered, tags = _verify_claims(answer, CHUNKS)
    assert tags == ["docs/a.md#Intro"]
    assert "Real fact." in filtered
    assert "Made-up fact." not in filtered


def test_verify_claims_drops_uncited_trailing_text():
    filtered, tags = _verify_claims("Real fact. [source: docs/a.md#Intro] Unsupported add-on.", CHUNKS)
    assert filtered == "Real fact. [source: docs/a.md#Intro]"
    assert tags == ["docs/a.md#Intro"]


def test_verify_claims_all_hallucinated_yields_empty():
    filtered, tags = _verify_claims("Nope. [source: docs/nonexistent.md#Nope]", CHUNKS)
    assert filtered == ""
    assert tags == []


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} passed")
