"""Tests for ellipsis normalization (pixelle_video/services/prosody.py).

Google Chirp3-HD (the project's default voice) mispronounces a raw ellipsis instead of
pausing, so normalize_ellipsis() rewrites "...", "…" and "……" into ordinary punctuation
a TTS voice reads as a pause. It is applied both at the Google-TTS text layer
(plain_text / build_ssml) and in the Story Shortener output. See the plan in
.claude/plans/ch-c-n-ng-story-shortener-harmonic-pike.md

Covered here (pure logic — no network / LLM / audio):
- normalize_ellipsis: the smart-pause mapping (mid -> ", ", end -> ".", redundant/leading
  -> dropped) across ASCII, Unicode and Chinese ellipsis.
- plain_text / build_ssml: no raw ellipsis reaches the engine.
- _split_title_and_summary: summarizer output is cleaned per paragraph (boundaries kept).

NOTE: named *_test.py (not test_*.py) because .gitignore ignores test_*.py;
pytest's default discovery (python_files = test_*.py *_test.py) still finds it.
"""

from pixelle_video.services.prosody import build_ssml, normalize_ellipsis, plain_text
from web.utils.story_summarizer import _split_title_and_summary


# --- normalize_ellipsis: the smart-pause mapping ---------------------------

def test_mid_sentence_becomes_comma():
    assert normalize_ellipsis("Tôi ổn... nhưng rồi sụp đổ") == "Tôi ổn, nhưng rồi sụp đổ"


def test_trailing_becomes_period():
    assert normalize_ellipsis("Tôi đã mất tất cả...") == "Tôi đã mất tất cả."


def test_unicode_ellipsis_mid_sentence():
    assert normalize_ellipsis("Tôi ổn… rồi sao") == "Tôi ổn, rồi sao"


def test_chinese_double_ellipsis_trailing():
    assert normalize_ellipsis("Tôi ổn……") == "Tôi ổn."


def test_redundant_after_terminator_is_dropped():
    assert normalize_ellipsis("Thật sao?...") == "Thật sao?"


def test_redundant_after_comma_keeps_a_space():
    assert normalize_ellipsis("Tôi ổn,... rồi sao") == "Tôi ổn, rồi sao"


def test_leading_ellipsis_dropped():
    assert normalize_ellipsis("...rồi tôi đi") == "rồi tôi đi"


def test_two_dots_also_normalized():
    assert normalize_ellipsis("Chờ.. đã") == "Chờ, đã"


def test_no_ellipsis_unchanged():
    assert normalize_ellipsis("Tôi ổn, rồi sao?") == "Tôi ổn, rồi sao?"


def test_single_dots_are_not_ellipsis():
    # ordinary sentence terminators must survive untouched
    assert normalize_ellipsis("Tôi ổn. Rồi sao.") == "Tôi ổn. Rồi sao."


def test_empty_and_falsy_unchanged():
    assert normalize_ellipsis("") == ""
    assert normalize_ellipsis(None) is None


# --- plain_text (Chirp3-HD path) -------------------------------------------

def test_plain_text_strips_markers_and_normalizes_ellipsis():
    assert plain_text("Tôi *ổn*... nhé") == "Tôi ổn, nhé"


def test_plain_text_no_raw_ellipsis_reaches_engine():
    out = plain_text("Một… hai... ba……")
    assert "…" not in out
    assert ".." not in out


# --- build_ssml (WaveNet path) ---------------------------------------------

def test_build_ssml_no_raw_ellipsis():
    out = build_ssml("Tôi ổn… rồi sao?")
    assert "…" not in out
    assert ".." not in out


def test_build_ssml_trailing_ellipsis_is_a_clean_sentence():
    out = build_ssml("Tôi mất tất cả...")
    assert out.startswith("<speak>") and out.endswith("</speak>")
    assert "…" not in out and ".." not in out


# --- summarizer output (web/utils/story_summarizer.py) ---------------------

def test_summarizer_cleans_ellipsis_per_paragraph():
    raw = "TITLE: Cú sốc...\n\nPara một...\n\nPara hai... vẫn tiếp"
    story = _split_title_and_summary(raw)

    # Title cleaned (trailing ellipsis -> full stop).
    assert story.title == "Cú sốc."
    # No raw ellipsis anywhere in the spoken summary.
    assert "..." not in story.summary and "…" not in story.summary
    # Paragraph-final -> period, mid-sentence -> comma, the 16-paragraph layout kept.
    assert story.summary.split("\n\n") == ["Cú sốc.", "Para một.", "Para hai, vẫn tiếp"]
