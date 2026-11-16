import pytest

from app.segmentation import segment_script


def test_empty_script_has_no_segments():
    assert segment_script("   ") == []


def test_groups_short_sentences_without_losing_punctuation():
    assert segment_script("Hello there. How are you?") == ["Hello there. How are you?"]


def test_splits_at_sentence_boundaries_under_character_limit():
    text = "First sentence. Second sentence. Third sentence."
    segments = segment_script(text, max_characters=20)

    assert segments == ["First sentence.", "Second sentence.", "Third sentence."]
    assert all(len(segment) <= 20 for segment in segments)


def test_splits_long_sentence_to_respect_character_limit():
    segments = segment_script("abcdefghij", max_characters=4)

    assert segments == ["abcd", "efgh", "ij"]


def test_rejects_non_positive_character_limit():
    with pytest.raises(ValueError, match="greater than zero"):
        segment_script("hello", max_characters=0)
