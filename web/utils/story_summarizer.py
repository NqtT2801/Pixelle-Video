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
Story summarizer.

Condenses a first-person life narrative into a sensational clickbait hook title
plus 15 narrative paragraphs, using the project's configured LLM (OpenAI gpt-4o
by default) via the shared OpenAI-SDK LLMService. The hook is prepended as the
first paragraph, so the spoken output is exactly 16 paragraphs (hook + 15).
"""

import re
from typing import NamedTuple

from loguru import logger

from pixelle_video.services.prosody import normalize_ellipsis


class ShortenedStory(NamedTuple):
    """A shortened narrative: a clickbait hook title and the 16-paragraph summary.

    The hook is prepended as the first paragraph, so the summary is EXACTLY 16
    short paragraphs (hook + 15 narrative). Downstream, Quick Create turns each
    paragraph into one ~5s video segment (16 × ~5s ≈ 80s), the first of which
    speaks the hook. `title` holds the same hook for use as the on-screen title.
    """
    title: str
    summary: str


# Target words per paragraph. Each paragraph becomes one video segment whose
# duration equals the length of the Vietnamese TTS audio (Google Chirp3-HD) that
# reads it. For an energetic pace the paragraphs are deliberately DENSE and the
# voice reads them FAST: this knob is PAIRED with the TTS speed
# (comfyui.tts.local.speed, ~1.4x) so each segment still lands at ~5s while
# delivering more story per second. Tune them together — raise words AND speed to
# pack more in at the same length, lower both to slow down. Changing one alone
# shifts the ~5s-per-segment target.
_WORDS_PER_PARAGRAPH = "20 to 26"

# Instruction prefix for the LLM. The narrative is appended after this prompt.
# The title is returned on a leading "TITLE:" line so a single plain-text call
# yields both pieces.
#
# The output feeds a Vietnamese TTS engine (Google Chirp3-HD): each of the 16
# paragraphs is read aloud (at a brisk ~1.4x pace) as one ~5s segment, so the text
# must be dense-but-short-per-paragraph and free of glyphs/abbreviations a TTS
# would mangle.
_SUMMARIZE_PROMPT = f"""You will receive a person's first-person narrative recounting their life circumstances and story.

The result will be read aloud, paragraph by paragraph, by a Vietnamese text-to-speech voice, where each paragraph becomes one short (~5 second) video segment. Rewrite the narrative as a condensed version optimized for that.

Structure rules:
- Output EXACTLY 15 paragraphs, separated by a single blank line, with the story distributed evenly across all 15. (A separate clickbait title is requested below; do not count it here.)
- Each paragraph must be roughly {_WORDS_PER_PARAGRAPH} words (a dense single spoken line, read at a brisk ~1.4x pace so it is still about 5 seconds aloud); never exceed 28 words and never go below 18.
- Stay in the first person and preserve the narrator's original voice, tone and emotion — do not turn it into a neutral report.
- Each paragraph must be self-contained and natural to read aloud on its own, yet the 15 paragraphs together must flow as one coherent, easy-to-follow story.
- Keep the essential events, emotions and circumstances; remove only repetition and minor detail.
- Vary the rhythm so the spoken narration rises and falls with feeling, never flat or monotone: mix short punchy sentences with the occasional question or exclamation that fits the emotion of the moment.
- Write in the SAME language as the input narrative.

Natural-reading rules (so the TTS voice never stumbles) — apply throughout:
- Spell out EVERY number, year, date, time, unit and percentage as full words in the output language. Examples (Vietnamese): `2020` -> "hai nghìn không trăm hai mươi", `5km` -> "năm ki-lô-mét", `30%` -> "ba mươi phần trăm", `8h` -> "tám giờ".
- Expand or rewrite abbreviations and symbols into words. Examples: `TP.HCM` -> "Thành phố Hồ Chí Minh", `&` -> "và". Do not leave bare acronyms.
- Replace foreign (e.g. English) words with a natural equivalent in the output language, or transliterate them so the voice pronounces them correctly. Do not leave raw foreign spellings.
- Use only the basic punctuation marks `.` `,` `!` `?` to shape the rhythm. To stop the voice from droning, wrap ONE or at most TWO genuinely impactful words per paragraph in *asterisks* (e.g. "Tôi mất *tất cả* chỉ trong một đêm") — the voice engine reads asterisked words with stronger emphasis. Use asterisks ONLY for this; do NOT use parentheses, slashes, ellipses, dashes, quotation marks, emoji or any other special characters.
- Avoid hard-to-pronounce clusters and ambiguous shorthand; prefer common, plainly spoken phrasing.

Output format — follow it EXACTLY and output nothing else:
- The first line must be `TITLE: ` followed by a sensational, scroll-stopping clickbait hook that grabs the viewer in the first second — open a strong curiosity gap or strike a raw emotion (shock, outrage, heartbreak, disbelief). You MAY tease or exaggerate to maximize clicks. Keep it to one punchy spoken line (at most ~14 words), in the SAME language as the input, no quotes. This hook is SPOKEN as the very first segment, so it must obey the natural-reading rules above and read aloud cleanly. Mark the single most curiosity-piquing word in the hook with *asterisks*, and end the hook with a `!` or `?` for punch.
- Then one empty line.
- Then the condensed version: exactly 15 paragraphs separated by one empty line, with no heading, labels, quotes or numbering.

Here is the narrative to condense:
"""

_TITLE_PREFIX = "TITLE:"


def _split_title_and_summary(output: str) -> ShortenedStory:
    """Parse the `TITLE:` line and build the 16-paragraph summary.

    The clickbait title is prepended as the first paragraph of the summary, so the
    spoken output is exactly 16 paragraphs (the hook + the 15 narrative paragraphs)
    and the very first segment is the hook. `title` is still returned separately so
    the same hook can also be used as the on-screen video title.
    """
    text = (output or "").strip()
    title = ""
    body = text

    newline = text.find("\n")
    first_line = text if newline == -1 else text[:newline]
    if first_line.strip().upper().startswith(_TITLE_PREFIX):
        title = first_line.split(":", 1)[1].strip().strip('"').strip("'").strip()
        body = (text[newline + 1:].strip() if newline != -1 else "")

    if not body:
        raise ValueError("The LLM returned an empty summary")

    # The prompt forbids ellipsis, but the LLM can still emit "..."/"…" — and the Google
    # Chirp3-HD voice mispronounces it. Normalize each paragraph (so a paragraph-final
    # ellipsis becomes a full stop, not a comma) plus the title, before assembling.
    title = normalize_ellipsis(title)
    body = "\n\n".join(normalize_ellipsis(p) for p in re.split(r"\n\s*\n", body))

    # Prepend the hook as the first paragraph so the spoken output is exactly 16
    # paragraphs (hook + 15) and segment one is the hook. If the model omitted the
    # TITLE line, fall back to the body as-is.
    summary = f"{title}\n\n{body}" if title else body
    return ShortenedStory(title=title, summary=summary)


def summarize_story(narrative: str, timeout: int = 240) -> ShortenedStory:
    """
    Condense a first-person life narrative via the project's configured LLM.

    Uses the LLM set in Settings (OpenAI gpt-4o by default) through the shared
    LLMService, and returns a clickbait hook title plus a 16-paragraph summary
    (the hook prepended as the first paragraph + 15 narrative paragraphs) that
    preserves the narrator's voice and the input language. Each paragraph is dense
    and sized to read in ~5s at the configured ~1.4x pace by the Google Chirp3-HD
    voice (16 × ~5s ≈ 80s) and is sanitized for natural reading
    (numbers/abbreviations spelled out, no TTS-unfriendly glyphs).

    Args:
        narrative: The raw first-person narrative.
        timeout: Unused; kept for backward compatibility with the previous engine.

    Returns:
        A ShortenedStory(title, summary).

    Raises:
        RuntimeError: if the LLM call fails.
        ValueError: if the response is empty or unparseable.
    """
    # Import lazily so importing this module never pulls in heavy deps eagerly.
    from pixelle_video.services.llm_service import LLMService
    from web.utils.async_helpers import run_async

    full_prompt = f"{_SUMMARIZE_PROMPT}\n\n{narrative}"

    # LLMService reads api_key/base_url/model dynamically from config_manager,
    # so it always uses whatever is configured in Settings (OpenAI gpt-4o).
    service = LLMService({})

    try:
        # max_tokens generous: 16 short paragraphs (title + 15); Vietnamese is
        # token-heavy, so leave headroom to avoid truncation.
        text = run_async(service(full_prompt, temperature=0.7, max_tokens=3000))
    except Exception as e:
        logger.error(f"LLM summarization failed: {e}")
        raise RuntimeError(str(e))

    summary_text = (text or "").strip()
    if not summary_text:
        raise ValueError("The LLM returned an empty response")

    return _split_title_and_summary(summary_text)
