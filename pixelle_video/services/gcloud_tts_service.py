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
Google Cloud Text-to-Speech engine (cloud neural, SSML-driven).

A self-contained wrapper around ``google-cloud-texttospeech``. Unlike the local
engines (VieNeu/clones) this calls Google's servers, so it is:
- **rock-solid consistent** — the same server voice every call, zero per-segment
  timbre/tone drift, and
- **expressive** — two paths: WaveNet/Standard voices render the SSML prosody/
  emphasis/breaks built by :mod:`pixelle_video.services.prosody`; Chirp3-HD voices
  ignore SSML but carry their own lively, dynamic delivery from plain text.

Pick the input mode with ``synthesize(..., is_ssml=...)``: SSML for WaveNet/Standard,
plain text for Chirp3-HD (which also has no pitch control — pass ``pitch=None``).
Credentials come from the standard ``GOOGLE_APPLICATION_CREDENTIALS`` env var
(path to a service-account JSON) or an explicit ``credentials_path``.

Used on CPU machines because all synthesis happens server-side — the local box
just sends text and receives mp3, so it is fast regardless of CPU.
"""

import os
import time
from typing import Optional

from loguru import logger

# Transient errors worth retrying vs. config/quota errors we should fail fast on.
_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 1.0  # seconds (exponential backoff)


class GCloudTTSEngine:
    """Lazy singleton wrapper around the Google Cloud TextToSpeechClient."""

    _instance: Optional["GCloudTTSEngine"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._client = None
        return cls._instance

    # ------------------------------------------------------------------ #
    # Lazy client
    # ------------------------------------------------------------------ #
    def _ensure_client(self, credentials_path: Optional[str] = None):
        if self._client is not None:
            return self._client

        try:
            from google.cloud import texttospeech
        except ImportError as e:
            raise ImportError(
                "Google Cloud TTS requires 'google-cloud-texttospeech'. Install it with: "
                "uv pip install --python .venv/Scripts/python.exe google-cloud-texttospeech"
            ) from e

        try:
            if credentials_path:
                if not os.path.exists(credentials_path):
                    raise FileNotFoundError(
                        f"Google Cloud credentials file not found: {credentials_path}"
                    )
                self._client = texttospeech.TextToSpeechClient.from_service_account_file(
                    credentials_path
                )
                logger.info(f"🔑 Google Cloud TTS using key file: {credentials_path}")
            else:
                # Reads GOOGLE_APPLICATION_CREDENTIALS (or other Application Default
                # Credentials). Fails clearly below if none are configured.
                self._client = texttospeech.TextToSpeechClient()
                logger.info("🔑 Google Cloud TTS using Application Default Credentials")
        except Exception as e:
            raise RuntimeError(
                "Failed to initialize Google Cloud TTS client. Set up credentials: enable the "
                "Text-to-Speech API, create a service-account key (JSON), and either set the env "
                "var GOOGLE_APPLICATION_CREDENTIALS to its path or set "
                "comfyui.tts.local.gcloud_credentials in config.yaml. "
                f"Original error: {e}"
            ) from e

        logger.info("✅ Google Cloud TTS client ready")
        return self._client

    # ------------------------------------------------------------------ #
    # Synthesis (blocking — caller runs it in a thread)
    # ------------------------------------------------------------------ #
    def synthesize(
        self,
        content: str,
        voice_name: str,
        output_mp3: str,
        *,
        is_ssml: bool = True,
        language_code: str = "vi-VN",
        speaking_rate: float = 1.0,
        pitch: Optional[float] = 0.0,
        credentials_path: Optional[str] = None,
    ) -> str:
        """Synthesize ``content`` with a Google voice; write an mp3 to ``output_mp3``.

        Two input modes:
        - ``is_ssml=True`` (default): ``content`` is SSML. Expressive rise/fall/rhythm
          comes from the SSML itself (see :mod:`pixelle_video.services.prosody`) — used
          by WaveNet/Standard voices.
        - ``is_ssml=False``: ``content`` is plain text — used by Chirp3-HD voices, which
          ignore SSML and carry their own expressive delivery.

        ``speaking_rate``/``pitch`` are the global AudioConfig controls (kept neutral by
        default; for SSML voices the SSML is the single source of prosody). ``pitch`` is
        omitted entirely when ``None`` because Chirp3-HD does not support a pitch control.

        Retries transient server errors with exponential backoff; fails fast (with a
        clear message) on auth/quota/permission errors.
        """
        from google.cloud import texttospeech

        client = self._ensure_client(credentials_path)
        os.makedirs(os.path.dirname(output_mp3) or ".", exist_ok=True)

        synthesis_input = (
            texttospeech.SynthesisInput(ssml=content)
            if is_ssml
            else texttospeech.SynthesisInput(text=content)
        )
        voice = texttospeech.VoiceSelectionParams(
            language_code=language_code,
            name=voice_name,
        )
        audio_kwargs = {
            "audio_encoding": texttospeech.AudioEncoding.MP3,
            "speaking_rate": speaking_rate,
        }
        if pitch is not None:  # Chirp3-HD has no pitch control -> omit the field.
            audio_kwargs["pitch"] = pitch
        audio_config = texttospeech.AudioConfig(**audio_kwargs)

        last_error: Optional[Exception] = None
        for attempt in range(_MAX_RETRIES):
            try:
                response = client.synthesize_speech(
                    input=synthesis_input,
                    voice=voice,
                    audio_config=audio_config,
                )
                with open(output_mp3, "wb") as f:
                    f.write(response.audio_content)
                logger.info(f"✅ Google Cloud TTS synthesized '{voice_name}': {output_mp3}")
                return output_mp3
            except Exception as e:  # noqa: BLE001 — classified below
                if not _is_retryable(e) or attempt == _MAX_RETRIES - 1:
                    last_error = e
                    break
                delay = _RETRY_BASE_DELAY * (2 ** attempt)
                logger.warning(
                    f"⚠️  Google Cloud TTS transient error (attempt {attempt + 1}/{_MAX_RETRIES}), "
                    f"retrying in {delay:.1f}s: {e}"
                )
                time.sleep(delay)

        raise RuntimeError(
            f"Google Cloud TTS failed for voice '{voice_name}': {last_error}"
        ) from last_error


def _is_retryable(error: Exception) -> bool:
    """True for transient server/network errors; False for auth/quota/bad-request."""
    try:
        from google.api_core import exceptions as gexc
    except ImportError:
        return False
    return isinstance(
        error,
        (
            gexc.ServiceUnavailable,
            gexc.DeadlineExceeded,
            gexc.InternalServerError,
            gexc.GatewayTimeout,
            gexc.TooManyRequests,
        ),
    )
