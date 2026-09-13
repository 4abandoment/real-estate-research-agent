import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from research_agent.review_pipeline import (
    TOPIC_COLUMNS,
    clean_text,
    detect_language,
    enrich_rows,
    load_taxonomy,
    parse_vector,
    sentiment_label,
    topic_flags,
)


def test_clean_text_strips_html_and_entities() -> None:
    assert clean_text("<br/>Great stay &amp; lovely") == "Great stay & lovely"
    assert clean_text("line one<br/>line two") == "line one line two"
    assert clean_text(None) == ""
    assert clean_text("   ") == ""


def test_sentiment_label_thresholds() -> None:
    assert sentiment_label(0.6) == "positive"
    assert sentiment_label(-0.3) == "negative"
    assert sentiment_label(0.01) == "neutral"
    assert sentiment_label(-0.01) == "neutral"


def test_parse_vector_handles_formats() -> None:
    assert parse_vector(None) is None
    assert parse_vector("") is None
    assert parse_vector("[1.0,2.0,3.0]").tolist() == [1.0, 2.0, 3.0]
    assert parse_vector(np.array([1, 2], dtype=np.float32)).tolist() == [1.0, 2.0]


def test_taxonomy_matches_schema_columns() -> None:
    taxonomy = load_taxonomy()

    assert sorted(topic["id"] for topic in taxonomy["topics"]) == sorted(TOPIC_COLUMNS)
    assert all(topic["anchors"] for topic in taxonomy["topics"])


def test_topic_flags_fires_on_clear_match() -> None:
    anchors = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    topic_index = np.asarray([0, 1])

    flags = topic_flags(np.asarray([1.0, 0.0], dtype=np.float32), anchors, topic_index, 0.75)

    assert flags[0] is True
    assert flags[1] is False
    assert len(flags) == len(TOPIC_COLUMNS)


def test_topic_flags_below_threshold_and_zero_vector() -> None:
    anchors = np.asarray([[1.0, 0.0]], dtype=np.float32)
    topic_index = np.asarray([0])

    below = topic_flags(np.asarray([0.5, 0.5], dtype=np.float32), anchors, topic_index, 0.75)
    zero = topic_flags(np.asarray([0.0, 0.0], dtype=np.float32), anchors, topic_index, 0.1)

    assert below == [False] * len(TOPIC_COLUMNS)
    assert zero == [False] * len(TOPIC_COLUMNS)


def test_detect_language() -> None:
    assert detect_language("The flat was great and very clean, we loved it") == "en"
    assert detect_language("Die Wohnung war sehr schön und sauber, vielen Dank!") == "de"
    assert detect_language("L'appartement était très bien situé, je recommande") == "fr"
    assert detect_language("Muchas gracias por todo, estaba muy limpio") == "es"
    assert detect_language("Superbe séjour, très bel appartement, merci beaucoup") == "fr"
    # short texts default to English (corpus prior)
    assert detect_language("Great place!") == "en"
    assert detect_language("") == "en"


def test_enrich_empty_comment_is_neutral_without_topics() -> None:
    analyzer = SentimentIntensityAnalyzer()
    anchors = np.zeros((1, 2), dtype=np.float32)
    topic_index = np.asarray([0])

    results = enrich_rows([(1, "   ", None)], analyzer, anchors, topic_index, 0.5)

    row = results[0]
    assert row[1] is None
    assert row[2] == "neutral"
    assert row[3] == 0.0
    assert not any(row[7:])


def test_enrich_scores_real_review_text() -> None:
    analyzer = SentimentIntensityAnalyzer()
    anchors = np.zeros((1, 2), dtype=np.float32)
    topic_index = np.asarray([0])

    results = enrich_rows(
        [(1, "Spotlessly clean &amp; a great location!", None)],
        analyzer,
        anchors,
        topic_index,
        0.5,
    )

    assert results[0][1] == "en"
    assert results[0][2] == "positive"
    assert results[0][3] > 0.05


def test_enrich_excludes_non_english_rows() -> None:
    analyzer = SentimentIntensityAnalyzer()
    anchors = np.zeros((1, 2), dtype=np.float32)
    topic_index = np.asarray([0])

    results = enrich_rows(
        [(1, "Die Wohnung war sehr schön und sauber, vielen Dank", None)],
        analyzer,
        anchors,
        topic_index,
        0.5,
    )

    row = results[0]
    assert row[1] == "de"
    assert row[2] is None
    assert row[3] is None
    assert all(flag is None for flag in row[7:])
