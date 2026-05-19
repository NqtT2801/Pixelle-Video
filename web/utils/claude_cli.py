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
Claude Code CLI helper.

Shells out to the locally installed, logged-in `claude` CLI in print mode so
features can use the user's Claude Code subscription directly, without an API
key and without adding a dependency.
"""

import os
import shutil
import subprocess
import tempfile
from typing import NamedTuple

from loguru import logger


class ClaudeCliError(Exception):
    """Raised when the `claude` CLI is missing, times out, or exits with an error."""
    pass


class ShortenedStory(NamedTuple):
    """A shortened narrative: an AI-suggested title and the 6-paragraph summary."""
    title: str
    summary: str


# Instruction passed to `claude -p`. The narrative itself is piped via stdin,
# so long stories are not constrained by command-line length limits. The title
# is returned on a leading "TITLE:" line so a single plain-text call yields
# both pieces (structured --json-schema output hangs together with --tools "").
_SUMMARIZE_PROMPT = """You will receive a person's first-person narrative recounting their life circumstances and story.

Rewrite it as a condensed version. Rules:
- Target ~250 words; do not exceed 265.
- Structure the condensed version as EXACTLY 6 paragraphs separated by a blank line, with the content distributed evenly across the 6 paragraphs.
- Stay in the first person and preserve the narrator's original voice, tone and writing style — do not turn it into a neutral report.
- Each paragraph must be coherent, and the 6 paragraphs together must read as one well-connected story that any reader can easily follow and understand.
- Keep the essential events, emotions and circumstances; remove only repetition and minor detail.
- Write in the SAME language as the input narrative.

Output format — follow it EXACTLY and output nothing else:
- The first line must be `TITLE: ` followed by a short, evocative title (at most ~10 words, same language as the input, no quotes).
- Then one empty line.
- Then the condensed version: exactly 6 paragraphs separated by one empty line, with no heading, labels, quotes or numbering."""

_TITLE_PREFIX = "TITLE:"


def is_claude_available() -> bool:
    """Return True if the `claude` CLI is on PATH. Cheap; safe to call per render."""
    return shutil.which("claude") is not None


def _split_title_and_summary(output: str) -> ShortenedStory:
    """Parse the `TITLE:` first line; the remainder is the 6-paragraph summary."""
    text = (output or "").strip()
    title = ""
    summary = text

    newline = text.find("\n")
    first_line = text if newline == -1 else text[:newline]
    if first_line.strip().upper().startswith(_TITLE_PREFIX):
        title = first_line.split(":", 1)[1].strip().strip('"').strip("'").strip()
        summary = (text[newline + 1:].strip() if newline != -1 else "")

    if not summary:
        raise ClaudeCliError("claude returned an empty summary")
    return ShortenedStory(title=title, summary=summary)


def summarize_story(narrative: str, timeout: int = 240) -> ShortenedStory:
    """
    Condense a first-person life narrative via the Claude Code CLI.

    Runs the logged-in Claude Code subscription (Sonnet, medium effort) and
    returns a title plus a ~250-word, 6-paragraph summary that preserves the
    narrator's voice and the input language.

    Args:
        narrative: The raw first-person narrative.
        timeout: Seconds to wait for the CLI before giving up.

    Returns:
        A ShortenedStory(title, summary).

    Raises:
        ClaudeCliError: if the CLI is missing, times out, or fails.
    """
    claude = shutil.which("claude")
    if not claude:
        raise ClaudeCliError("claude-not-found")

    # Strip ANTHROPIC_API_KEY so the call always uses the subscription OAuth
    # login, never pay-per-token API billing.
    env = os.environ.copy()
    env.pop("ANTHROPIC_API_KEY", None)

    cmd = [
        claude, "-p", _SUMMARIZE_PROMPT,
        "--model", "sonnet",            # latest Sonnet
        "--effort", "medium",           # explicit, as requested
        "--output-format", "text",
        "--tools", "",                  # pure text generation, no tool/file access
        "--strict-mcp-config",          # do not load any MCP servers
        "--disable-slash-commands",
        "--no-session-persistence",
    ]

    # Hide the console window that would otherwise flash on Windows; 0 on POSIX.
    creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)

    try:
        # Run in a throwaway directory so no project CLAUDE.md, hooks, or
        # settings are picked up — the call stays deterministic.
        with tempfile.TemporaryDirectory() as workdir:
            proc = subprocess.run(
                cmd,
                input=narrative,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=timeout,
                cwd=workdir,
                env=env,
                creationflags=creation_flags,
            )
    except subprocess.TimeoutExpired:
        raise ClaudeCliError(f"claude timed out after {timeout}s")
    except OSError as e:
        raise ClaudeCliError(str(e))

    if proc.returncode != 0:
        err = (proc.stderr or "").strip() or f"claude exited with code {proc.returncode}"
        logger.error(f"claude CLI failed: {err}")
        raise ClaudeCliError(err)

    summary_text = (proc.stdout or "").strip()
    if not summary_text:
        raise ClaudeCliError("claude returned an empty response")

    return _split_title_and_summary(summary_text)
