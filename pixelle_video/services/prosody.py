# Copyright (C) 2025 AIDC-AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Prosody / rhythm layer — turn a flat narration line into expressive SSML.

A plain Vietnamese narration segment reads *đều đều* (monotone) on a raw neural
voice. This module wraps it in SSML so the Google Cloud WaveNet voice delivers it
with rise/fall, rhythm and punch — the "dopamine" delivery that keeps viewers
watching — while staying the SAME speaker every segment (variation here is by
DESIGN, not random, so it is consistent and reproducible).

Levers (all supported by WaveNet/Standard voices; Chirp3-HD/Journey do NOT take
SSML, so this layer must only be used with a WaveNet/Standard ``gcloud:`` voice):
- **Sentence-type pitch**: questions and exclamations get a pitch lift so they
  don't land flat.
- **Keyword emphasis**: the LLM marks impactful words with ``*asterisks*`` →
  ``<emphasis>`` (with a light heuristic fallback for numbers).
- **Dramatic breaks**: short ``<break>`` at clause (comma) boundaries, a medium
  one between sentences, and a long one AFTER the title hook so it "lands".
- **Energetic baseline rate** on body lines; the **title** gets a distinct,
  slightly slower + brighter delivery to grab attention in the first second.

Public API: :func:`build_ssml` (pure, testable), :func:`plain_text`,
:func:`normalize_ellipsis` and :func:`prosody_settings_from_config`.
"""

import re
from typing import Any, Dict, List, Optional, Tuple

# --------------------------------------------------------------------------- #
# Tunable defaults (override per-video via config.yaml -> comfyui.tts.local,
# keys prefixed ``prosody_*`` — see prosody_settings_from_config()).
# --------------------------------------------------------------------------- #
DEFAULTS: Dict[str, Any] = {
    # Body (non-title) lines
    "baseline_rate": 108,      # % of normal speed — a touch brisk = energetic
    "baseline_pitch": 0.0,     # semitones, relative to the voice default
    # Per-sentence-type pitch lift (semitones, added on top of baseline)
    "question_pitch": 2.0,     # questions lift so they sound curious, not flat
    "exclaim_pitch": 1.5,      # exclamations lift for energy
    # Emphasis
    "emphasis_level": "strong",   # SSML <emphasis level=...> for *marked* words
    "emphasize_numbers": True,    # heuristic: also emphasize bare numbers
    # Pauses (milliseconds)
    "comma_break_ms": 220,        # clause rhythm
    "sentence_break_ms": 380,     # between sentences within one segment
    # Title (hook) — the first segment; grab attention + let it land
    "title_rate": 98,             # slightly slower than body for gravitas/clarity
    "title_pitch": 1.0,           # a touch brighter/engaged
    "title_emphasis": False,      # wrap the WHOLE title in <emphasis> (usually rely
                                  #   on the LLM's *keyword* markers instead)
    "title_trailing_break_ms": 600,  # dramatic pause after the hook
}

_EMPHASIS_RE = re.compile(r"\*([^*]+)\*")
_NUMBER_RE = re.compile(r"^[0-9][0-9.,]*$")
# A run of ellipsis: 2+ ASCII dots, the Unicode ellipsis (…) or the Chinese double
# ellipsis (……). Only horizontal whitespace around it is consumed (not newlines) so a
# paragraph-final ellipsis never swallows the blank line separating paragraphs.
_ELLIPSIS_RE = re.compile(r"[^\S\r\n]*(?:\.{2,}|…+)[^\S\r\n]*")


def _fmt_st(semitones: float) -> str:
    """Format a semitone delta as an SSML relative pitch, e.g. ``+2st`` / ``-1st``."""
    return f"{semitones:+g}st"


def _escape(text: str) -> str:
    """XML-escape SSML text content (the markup tags are added separately)."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def normalize_ellipsis(text: str) -> str:
    """Map an ellipsis (``...``, ``…``, ``……``) to punctuation a TTS voice reads as a
    natural pause.

    Google Chirp3-HD mispronounces a raw ellipsis instead of pausing, so each run is
    rewritten by position:

    - ends the text → ``.`` (full stop / dramatic landing), unless it already follows a
      terminator;
    - mid-sentence → ``", "`` (a short pause that keeps the brisk flow);
    - directly after ``. ! ? , ; :`` → dropped (the existing mark stands);
    - leads the text → dropped.

    Operates on a single line/segment; callers with multi-paragraph text should normalize
    each paragraph (the regex never consumes the newlines between them).
    """
    if not text:
        return text

    def _repl(m: "re.Match[str]") -> str:
        before = text[:m.start()].rstrip()
        after = text[m.end():]
        if not after.strip():                       # ends the text -> full stop
            return "." if before and before[-1] not in ".!?" else ""
        if not before:                              # leads the text -> drop
            return ""
        if before[-1] in ".!?,;:":                  # already punctuated -> keep, just space
            return " "
        return ", "                                  # mid-sentence -> short pause

    return _ELLIPSIS_RE.sub(_repl, text)


def _split_sentences(text: str) -> List[Tuple[str, str]]:
    """Split ``text`` into ``[(body_without_terminator, terminator), ...]``.

    Terminator is one of ``. ! ?``; a trailing fragment with no terminator is
    treated as a statement (``.``). The terminator is kept (re-attached by the
    renderer) because it helps the engine's own intonation, esp. for ``?``.
    """
    out: List[Tuple[str, str]] = []
    tokens = re.split(r"([.!?]+)", text)
    i = 0
    while i < len(tokens):
        body = tokens[i].strip()
        term = tokens[i + 1] if i + 1 < len(tokens) else ""
        if body:
            out.append((body, term[-1] if term else "."))
        i += 2
    return out


def _render_clause(clause: str, s: Dict[str, Any]) -> str:
    """Escape a clause and convert ``*word*`` markers (and bare numbers) to emphasis."""
    esc = _escape(clause)
    level = s["emphasis_level"]
    esc = _EMPHASIS_RE.sub(lambda m: f'<emphasis level="{level}">{m.group(1)}</emphasis>', esc)
    # Drop any unbalanced leftover marker so it is never spoken as "sao".
    esc = esc.replace("*", "")
    if s.get("emphasize_numbers"):
        # Emphasize standalone numbers (already spelled out by the LLM usually, but
        # any literal number is a high-salience token worth punching).
        esc = " ".join(
            f'<emphasis level="{level}">{w}</emphasis>' if _NUMBER_RE.match(w) else w
            for w in esc.split(" ")
        )
    return esc


def _render_sentence(body: str, term: str, s: Dict[str, Any]) -> str:
    """Render one sentence: clause breaks for rhythm + a pitch lift for ? / !."""
    clauses = [c.strip() for c in body.split(",") if c.strip()]
    if not clauses:
        clauses = [body.strip()]
    joined = (f'<break time="{s["comma_break_ms"]}ms"/>').join(
        _render_clause(c, s) for c in clauses
    )
    inner = joined + _escape(term)
    if term == "?":
        inner = f'<prosody pitch="{_fmt_st(s["question_pitch"])}">{inner}</prosody>'
    elif term == "!":
        inner = f'<prosody pitch="{_fmt_st(s["exclaim_pitch"])}">{inner}</prosody>'
    return inner


def build_ssml(text: str, is_title: bool = False, settings: Optional[Dict[str, Any]] = None) -> str:
    """Convert a narration segment into expressive SSML for Google Cloud TTS.

    Args:
        text: The narration line. May contain ``*word*`` emphasis markers placed by
            the LLM; only the basic terminators ``. , ! ?`` are expected otherwise.
        is_title: True for the first segment (the hook/title) → distinct, attention-
            grabbing delivery with a trailing pause.
        settings: Optional overrides for :data:`DEFAULTS`.

    Returns:
        A ``<speak>…</speak>`` SSML string. Returns ``<speak></speak>`` for empty text.
    """
    s = {**DEFAULTS, **(settings or {})}
    # Normalize ellipsis first so _split_sentences sees ordinary terminators: a raw
    # "..." would otherwise collapse to a single "." and "…" would leak as literal text.
    text = normalize_ellipsis(" ".join((text or "").split()))
    if not text:
        return "<speak></speak>"

    sentences = _split_sentences(text)
    rendered = [_render_sentence(b, t, s) for b, t in sentences]
    body = (f'<break time="{s["sentence_break_ms"]}ms"/>').join(rendered)

    if is_title:
        body = f'<prosody rate="{s["title_rate"]}%" pitch="{_fmt_st(s["title_pitch"])}">{body}</prosody>'
        if s.get("title_emphasis"):
            body = f'<emphasis level="{s["emphasis_level"]}">{body}</emphasis>'
        body += f'<break time="{s["title_trailing_break_ms"]}ms"/>'
    else:
        body = f'<prosody rate="{s["baseline_rate"]}%" pitch="{_fmt_st(s["baseline_pitch"])}">{body}</prosody>'

    return f"<speak>{body}</speak>"


def plain_text(text: str) -> str:
    """Strip the LLM's markup so the line can be spoken by a non-SSML engine.

    Chirp3-HD voices ignore SSML and carry their own expressive delivery, so they take
    plain text. This removes the ``*word*`` emphasis markers the LLM adds (keeping the
    word itself, never the literal ``*`` — which would be read aloud as "sao") and
    collapses whitespace. Basic punctuation ``. , ! ?`` is kept on purpose: Chirp3-HD
    uses it for its own intonation. An ellipsis is normalized to a pause via
    :func:`normalize_ellipsis` (Chirp3-HD otherwise mispronounces a raw ``...``/``…``).
    """
    text = _EMPHASIS_RE.sub(lambda m: m.group(1), text or "")
    text = text.replace("*", "")  # drop any unbalanced leftover marker
    text = normalize_ellipsis(text)  # Chirp3-HD mispronounces a raw "..."/"…"
    return " ".join(text.split())


def prosody_settings_from_config(local_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Pull ``prosody_*`` overrides out of the ``comfyui.tts.local`` config dict.

    Any key in :data:`DEFAULTS` can be overridden in config.yaml by prefixing it
    with ``prosody_`` (e.g. ``prosody_baseline_rate: 112``). Missing keys keep
    their default. Requires TTSLocalConfig ``extra='allow'`` so the keys survive
    the config round-trip.
    """
    settings: Dict[str, Any] = {}
    for key in DEFAULTS:
        cfg_key = f"prosody_{key}"
        if cfg_key in local_cfg and local_cfg[cfg_key] is not None:
            settings[key] = local_cfg[cfg_key]
    return settings


if __name__ == "__main__":  # quick manual check: python -m pixelle_video.services.prosody
    print("TITLE :", build_ssml("Tôi đã mất tất cả chỉ trong *một đêm*!", is_title=True))
    print("BODY  :", build_ssml("Năm đó tôi 20 tuổi, ôm giấc mơ đổi đời. Liệu tôi có làm được không?"))
