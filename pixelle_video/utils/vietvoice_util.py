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
VietVoice-TTS utility — local, offline Vietnamese Text-to-Speech (ONNX Runtime).

Provides natural Vietnamese voices for both genders and all three regional
accents, selected via ``gender`` ("male"/"female") and ``area``
("northern"/"central"/"southern"). The model (~140MB) is downloaded on first
use and cached under ``~/.cache/vietvoicetts``; no API key, runs on CPU.

Repo: https://github.com/nguyenvulebinh/VietVoice-TTS (MIT License)

Notes
-----
* The ~140MB ONNX session is loaded once and reused across calls via a
  process-wide singleton (cheap per-call inference, expensive one-time load).
* Synthesis is serialized with an asyncio.Lock because the ONNX session is
  shared; the heavy work runs in a worker thread (``asyncio.to_thread``) so the
  event loop stays responsive.
* VietVoice outputs WAV; we transcode to MP3 with ffmpeg (already a project
  dependency) and apply the speed multiplier with the ``atempo`` filter
  (tempo change without pitch shift), matching the project's 0.5x–2.0x range.
"""

import asyncio
import os
import sys
import uuid
from pathlib import Path
from typing import Optional

from loguru import logger


def _ensure_torch_importable() -> None:
    """Install a no-op ``torch`` stub if torch isn't installed.

    vietvoicetts 0.1.0 contains a single *dead* ``import torch`` in
    ``core/tts_engine.py`` — torch is never actually used (all inference runs on
    ONNX Runtime + numpy) and the package doesn't even declare torch as a
    dependency. Stubbing it lets us import vietvoicetts without pulling the heavy
    (~hundreds of MB) torch wheel, keeping this path lightweight. A real torch
    install, if present, always takes precedence.
    """
    import sys
    import types

    if "torch" in sys.modules:
        return
    try:
        import torch  # noqa: F401 — prefer real torch when available
    except ModuleNotFoundError:
        sys.modules["torch"] = types.ModuleType("torch")
        logger.debug("Installed no-op 'torch' stub for vietvoicetts (dead import)")


def _ensure_utf8_stdout() -> None:
    """Make stdout/stderr UTF-8 so vietvoicetts' emoji prints don't crash.

    On a Windows cp1252 console, vietvoicetts printing "✅ Model downloaded..."
    raises UnicodeEncodeError, which the library then mis-reports as a download
    failure. Reconfiguring with errors="replace" makes those prints harmless.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            if stream is not None and hasattr(stream, "reconfigure"):
                enc = (getattr(stream, "encoding", "") or "").lower()
                if enc not in ("utf-8", "utf8"):
                    stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


# Run both shims as soon as this module loads, before any vietvoicetts import.
_ensure_utf8_stdout()
_ensure_torch_importable()


# Process-wide singletons so the ONNX session is loaded only once.
_tts_api = None
_api_init_lock: Optional[asyncio.Lock] = None
_synth_lock: Optional[asyncio.Lock] = None
_lock_loop = None


def _get_locks():
    """Get (init_lock, synth_lock) bound to the current running event loop."""
    global _api_init_lock, _synth_lock, _lock_loop
    loop = asyncio.get_running_loop()
    if _lock_loop is not loop or _api_init_lock is None or _synth_lock is None:
        _api_init_lock = asyncio.Lock()
        _synth_lock = asyncio.Lock()
        _lock_loop = loop
    return _api_init_lock, _synth_lock


def _create_api():
    """Create the VietVoice TTSApi (loads ONNX models; downloads on first run)."""
    # Lazy import: keeps the dependency optional and the rest of the app working
    # even when VietVoice-TTS isn't installed.
    from vietvoicetts import TTSApi, ModelConfig

    logger.info("⏳ Loading VietVoice-TTS model (first run downloads ~140MB)...")
    # speed lives in ModelConfig in vietvoicetts; keep native synthesis at 1.0 and
    # apply the user's speed afterwards via ffmpeg atempo (see _wav_to_mp3).
    cfg = ModelConfig()
    try:
        cfg.speed = 1.0
    except Exception:
        pass
    api = TTSApi(cfg)
    logger.success("✅ VietVoice-TTS model loaded")
    return api


async def _get_api():
    """Lazily create and cache the TTSApi singleton (thread-safe per loop)."""
    global _tts_api
    if _tts_api is not None:
        return _tts_api
    init_lock, _ = _get_locks()
    async with init_lock:
        if _tts_api is None:
            _tts_api = await asyncio.to_thread(_create_api)
    return _tts_api


def _synth_to_wav(api, text: str, gender: str, area: str, wav_path: str) -> None:
    """Synchronous synthesis to a WAV file (runs in a worker thread).

    Prefers the session-reusing ``TTSApi.synthesize_to_file`` and falls back to
    the module-level ``synthesize`` convenience function if the method name
    differs in the installed version.
    """
    if hasattr(api, "synthesize_to_file"):
        api.synthesize_to_file(text, wav_path, gender=gender, area=area)
        return
    # Fallback: top-level convenience function (may reload the model per call)
    from vietvoicetts import synthesize
    synthesize(text, wav_path, gender=gender, area=area)


def _wav_to_mp3(wav_path: str, mp3_path: str, speed: float = 1.0) -> None:
    """Transcode WAV -> MP3, applying an optional speed change via ffmpeg atempo."""
    import ffmpeg

    audio = ffmpeg.input(wav_path).audio
    # atempo handles 0.5x–2.0x in a single stage, which covers the UI slider range.
    if speed and abs(speed - 1.0) > 1e-3:
        audio = audio.filter("atempo", max(0.5, min(2.0, float(speed))))
    out = ffmpeg.output(audio, mp3_path, **{"b:a": "192k"})
    ffmpeg.run(out, overwrite_output=True, quiet=True)


async def vietvoice_tts(
    text: str,
    gender: str = "female",
    area: str = "northern",
    speed: float = 1.0,
    output_path: str = None,
) -> str:
    """
    Generate Vietnamese speech with VietVoice-TTS and write an MP3 to output_path.

    Args:
        text: Text to synthesize.
        gender: "male" or "female".
        area: "northern" (miền Bắc), "central" (miền Trung), or "southern" (miền Nam).
        speed: Speed multiplier (1.0 = native), applied via ffmpeg atempo.
        output_path: Destination .mp3 path (auto-generated under output/ if None).

    Returns:
        Path to the generated MP3 file.
    """
    # Resolve output path / ensure directory exists
    if not output_path:
        Path("output").mkdir(parents=True, exist_ok=True)
        output_path = f"output/{uuid.uuid4().hex}.mp3"
    else:
        parent = os.path.dirname(output_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

    api = await _get_api()

    # Synthesize to a temporary WAV. Serialize because the ONNX session is shared.
    tmp_wav = f"{os.path.splitext(output_path)[0]}.vietvoice.wav"
    _, synth_lock = _get_locks()
    async with synth_lock:
        await asyncio.to_thread(_synth_to_wav, api, text, gender, area, tmp_wav)

    # Transcode to MP3 + apply speed (off the event loop)
    await asyncio.to_thread(_wav_to_mp3, tmp_wav, output_path, speed)

    # Best-effort cleanup of the temp WAV
    try:
        os.remove(tmp_wav)
    except OSError:
        pass

    logger.debug(f"VietVoice-TTS wrote: {output_path}")
    return output_path
