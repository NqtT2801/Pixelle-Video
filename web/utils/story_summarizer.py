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

Condenses a first-person life narrative into a title plus an 18-paragraph
summary using the project's configured LLM (OpenAI gpt-4o by default), via the
shared OpenAI-SDK LLMService — the same engine the rest of the app uses.
"""

from typing import NamedTuple

from loguru import logger


class ShortenedStory(NamedTuple):
    """A shortened narrative: an AI-suggested title and the 18-paragraph summary.

    The summary is split into EXACTLY 18 short paragraphs so that, downstream,
    Quick Create turns each paragraph into one ~5s video segment (18 × 5s ≈ 90s).
    """
    title: str
    summary: str


# Target words per paragraph. Each paragraph becomes one video segment whose
# duration equals the length of the VieNeu (Vietnamese TTS) audio that reads it.
# This number is the single knob to tune the ~5s-per-segment target: lower it if
# segments come out longer than 5s, raise it if they come out shorter.
_WORDS_PER_PARAGRAPH = "14 to 18"

# Instruction prefix for the LLM. The narrative is appended after this prompt.
# The title is returned on a leading "TITLE:" line so a single plain-text call
# yields both pieces.
#
# The output feeds a Vietnamese TTS engine (VieNeu): each of the 18 paragraphs is
# read aloud as one ~5s segment, so the text must be both short-per-paragraph and
# free of glyphs/abbreviations a TTS would mangle.
_SUMMARIZE_PROMPT = f"""You will receive a person's first-person narrative recounting their life circumstances and story.

The result will be read aloud, paragraph by paragraph, by a Vietnamese text-to-speech voice, where each paragraph becomes one short (~5 second) video segment. Rewrite the narrative as a condensed version optimized for that.

Structure rules:
- Output EXACTLY 18 paragraphs, separated by a single blank line, with the story distributed evenly across all 18.
- Each paragraph must be roughly {_WORDS_PER_PARAGRAPH} words (about a single spoken line, ~5 seconds aloud); never exceed 20 words and never go below 12.
- Stay in the first person and preserve the narrator's original voice, tone and emotion — do not turn it into a neutral report.
- Each paragraph must be self-contained and natural to read aloud on its own, yet the 18 paragraphs together must flow as one coherent, easy-to-follow story.
- Keep the essential events, emotions and circumstances; remove only repetition and minor detail.
- Write in the SAME language as the input narrative.

Natural-reading rules (so the TTS voice never stumbles) — apply throughout:
- Spell out EVERY number, year, date, time, unit and percentage as full words in the output language. Examples (Vietnamese): `2020` -> "hai nghìn không trăm hai mươi", `5km` -> "năm ki-lô-mét", `30%` -> "ba mươi phần trăm", `8h` -> "tám giờ".
- Expand or rewrite abbreviations and symbols into words. Examples: `TP.HCM` -> "Thành phố Hồ Chí Minh", `&` -> "và". Do not leave bare acronyms.
- Replace foreign (e.g. English) words with a natural equivalent in the output language, or transliterate them so the voice pronounces them correctly. Do not leave raw foreign spellings.
- Use only the basic punctuation marks `.` `,` `!` `?` to shape the rhythm. Do NOT use parentheses, slashes, ellipses, dashes, quotation marks, emoji or any other special characters.
- Avoid hard-to-pronounce clusters and ambiguous shorthand; prefer common, plainly spoken phrasing.

Output format — follow it EXACTLY and output nothing else:
- The first line must be `TITLE: ` followed by a short, evocative title (at most ~10 words, same language as the input, no quotes), itself following the natural-reading rules above.
- Then one empty line.
- Then the condensed version: exactly 18 paragraphs separated by one empty line, with no heading, labels, quotes or numbering.

Here is the narrative to condense:
"""

_TITLE_PREFIX = "TITLE:"


def _split_title_and_summary(output: str) -> ShortenedStory:
    """Parse the `TITLE:` first line; the remainder is the 18-paragraph summary."""
    text = (output or "").strip()
    title = ""
    summary = text

    newline = text.find("\n")
    first_line = text if newline == -1 else text[:newline]
    if first_line.strip().upper().startswith(_TITLE_PREFIX):
        title = first_line.split(":", 1)[1].strip().strip('"').strip("'").strip()
        summary = (text[newline + 1:].strip() if newline != -1 else "")

    if not summary:
        raise ValueError("The LLM returned an empty summary")
    return ShortenedStory(title=title, summary=summary)


def summarize_story(narrative: str, timeout: int = 240) -> ShortenedStory:
    """
    Condense a first-person life narrative via the project's configured LLM.

    Uses the LLM set in Settings (OpenAI gpt-4o by default) through the shared
    LLMService, and returns a title plus an 18-paragraph summary that preserves
    the narrator's voice and the input language. Each paragraph is sized to read
    in ~5s by the VieNeu TTS voice (18 × 5s ≈ 90s) and is sanitized for natural
    reading (numbers/abbreviations spelled out, no TTS-unfriendly glyphs).

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
        # max_tokens generous: 18 short paragraphs + title; Vietnamese is
        # token-heavy, so leave headroom to avoid truncation.
        text = run_async(service(full_prompt, temperature=0.7, max_tokens=3000))
    except Exception as e:
        logger.error(f"LLM summarization failed: {e}")
        raise RuntimeError(str(e))

    summary_text = (text or "").strip()
    if not summary_text:
        raise ValueError("The LLM returned an empty response")

    return _split_title_and_summary(summary_text)
