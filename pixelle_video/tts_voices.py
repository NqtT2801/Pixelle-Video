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
TTS Voice Configuration

Defines available voices for local Edge TTS inference.
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional


# ---------------------------------------------------------------------------
# Custom cloned voices (Edge TTS base + OpenVoice tone-color conversion)
# ---------------------------------------------------------------------------
# Any audio file dropped into `voices/` becomes a selectable voice. Selecting it
# generates speech with `DEFAULT_CLONE_BASE` (Edge TTS) and then converts the
# timbre to that reference clip via OpenVoice. See
# `pixelle_video.services.voice_conversion`.
VOICES_DIR = "voices"
CLONE_VOICE_PREFIX = "clone:"
# VieNeu-TTS preset voices are selected by id (see VIENEU_VOICES below).
VIENEU_VOICE_PREFIX = "vieneu:"
# Google Cloud TTS voices are selected by id "gcloud:<google_voice_name>" (see GCLOUD_VOICES below).
GCLOUD_VOICE_PREFIX = "gcloud:"
# Northern Vietnamese male Edge voice used to drive pronunciation/prosody before
# the timbre is converted to the reference clip.
DEFAULT_CLONE_BASE = "vi-VN-NamMinhNeural"
_CLONE_AUDIO_EXTS = (".mp3", ".wav", ".flac", ".m4a", ".ogg")


# Edge TTS voice presets for local inference
EDGE_TTS_VOICES: List[Dict[str, Any]] = [
    # Chinese voices
    {
        "id": "zh-CN-XiaoxiaoNeural",
        "label_key": "tts.voice.zh_CN_XiaoxiaoNeural",
        "locale": "zh-CN",
        "gender": "female"
    },
    {
        "id": "zh-CN-XiaoyiNeural",
        "label_key": "tts.voice.zh_CN_XiaoyiNeural",
        "locale": "zh-CN",
        "gender": "female"
    },
    {
        "id": "zh-CN-YunjianNeural",
        "label_key": "tts.voice.zh_CN_YunjianNeural",
        "locale": "zh-CN",
        "gender": "male"
    },
    {
        "id": "zh-CN-YunxiNeural",
        "label_key": "tts.voice.zh_CN_YunxiNeural",
        "locale": "zh-CN",
        "gender": "male"
    },
    {
        "id": "zh-CN-YunyangNeural",
        "label_key": "tts.voice.zh_CN_YunyangNeural",
        "locale": "zh-CN",
        "gender": "male"
    },
    {
        "id": "zh-CN-YunyeNeural",
        "label_key": "tts.voice.zh_CN_YunyeNeural",
        "locale": "zh-CN",
        "gender": "male"
    },
    {
        "id": "zh-CN-YunfengNeural",
        "label_key": "tts.voice.zh_CN_YunfengNeural",
        "locale": "zh-CN",
        "gender": "male"
    },
    {
        "id": "zh-CN-liaoning-XiaobeiNeural",
        "label_key": "tts.voice.zh_CN_liaoning_XiaobeiNeural",
        "locale": "zh-CN",
        "gender": "female"
    },
    {
        "id": "en-US-AriaNeural",
        "label_key": "tts.voice.en_US_AriaNeural",
        "locale": "en-US",
        "gender": "female"
    },
    {
        "id": "en-US-JennyNeural",
        "label_key": "tts.voice.en_US_JennyNeural",
        "locale": "en-US",
        "gender": "female"
    },
    {
        "id": "en-US-GuyNeural",
        "label_key": "tts.voice.en_US_GuyNeural",
        "locale": "en-US",
        "gender": "male"
    },
    {
        "id": "en-US-DavisNeural",
        "label_key": "tts.voice.en_US_DavisNeural",
        "locale": "en-US",
        "gender": "male"
    },
    {
        "id": "en-GB-SoniaNeural",
        "label_key": "tts.voice.en_GB_SoniaNeural",
        "locale": "en-GB",
        "gender": "female"
    },
    {
        "id": "en-GB-RyanNeural",
        "label_key": "tts.voice.en_GB_RyanNeural",
        "locale": "en-GB",
        "gender": "male"
    },
    {
        "id": "ko-KR-InJoonNeural",
        "label_key": "tts.voice.ko-KR-InJoonNeural",
        "locale": "ko-KR",
        "gender": "male"
    },
    {
        "id": "ko-KR-SunHiNeural",
        "label_key": "tts.voice.ko-KR-SunHiNeural",
        "locale": "ko-KR",
        "gender": "female"
    },
    {
        "id": "fr-FR-EloiseNeural",
        "label_key": "tts.voice.fr-FR-EloiseNeural",
        "locale": "fr-FR",
        "gender": "female"
    },
    {
        "id": "fr-FR-HenriNeural",
        "label_key": "tts.voice.fr-FR-HenriNeural",
        "locale": "fr-FR",
        "gender": "male"
    },
    {
        "id": "pt-PT-DuarteNeural",
        "label_key": "tts.voice.pt-PT-DuarteNeural",
        "locale": "pt-PT",
        "gender": "male"
    },
    {
        "id": "pt-PT-RaquelNeural",
        "label_key": "tts.voice.pt-PT-RaquelNeural",
        "locale": "pt-PT",
        "gender": "female"
    },
    {
        "id": "de-DE-AmalaNeural",
        "label_key": "tts.voice.de-DE-AmalaNeural",
        "locale": "de-DE",
        "gender": "female"
    },
    {
        "id": "de-DE-ConradNeural",
        "label_key": "tts.voice.de-DE-ConradNeural",
        "locale": "de-DE",
        "gender": "male"
    },
    
    # English voices
    {
        "id": "ru-RU-DmitryNeural",
        "label_key": "tts.voice.ru-RU-DmitryNeural",
        "locale": "ru-RU",
        "gender": "male"
    },
    {
        "id": "ru-RU-SvetlanaNeural",
        "label_key": "tts.voice.ru-RU-SvetlanaNeural",
        "locale": "ru-RU",
        "gender": "female"
    },
    {
        "id": "tr-TR-AhmetNeural",
        "label_key": "tts.voice.tr-TR-AhmetNeural",
        "locale": "tr-TR",
        "gender": "male"
    },
    {
        "id": "tr-TR-EmelNeural",
        "label_key": "tts.voice.tr-TR-EmelNeural",
        "locale": "tr-TR",
        "gender": "female"
    },
    {
        "id": "es-ES-AlvaroNeural",
        "label_key": "tts.voice.es-ES-AlvaroNeural",
        "locale": "es-ES",
        "gender": "male"
    },
    {
        "id": "es-ES-ElviraNeural",
        "label_key": "tts.voice.es-ES-ElviraNeural",
        "locale": "es-ES",
        "gender": "female"
    },

    # Vietnamese voices
    {
        "id": "vi-VN-HoaiMyNeural",
        "label_key": "tts.voice.vi-VN-HoaiMyNeural",
        "locale": "vi-VN",
        "gender": "female"
    },
    {
        "id": "vi-VN-NamMinhNeural",
        "label_key": "tts.voice.vi-VN-NamMinhNeural",
        "locale": "vi-VN",
        "gender": "male"
    },
]


# ---------------------------------------------------------------------------
# VieNeu-TTS preset voices (Vietnamese named-voice engine)
# ---------------------------------------------------------------------------
# VieNeu-TTS (https://github.com/pnnbao97/VieNeu-TTS) ships built-in preset voices,
# synthesized by `pixelle_video.services.vieneu_service.VieNeuEngine`. Unlike `clone:`
# voices these are named speakers selected by id (no reference clip), so — like Edge
# TTS — they are a static registry. `voice_id` is the name passed to VieNeu's
# `infer(voice=...)`; `name` is the friendly label shown in the UI.
#
# Roster below is the v3 Turbo set (vieneu==3.0.4, assets/voices_v3_turbo.json). Refresh
# after a package update with:
#   python -c "from pixelle_video.services.vieneu_service import VieNeuEngine as E; print(E().list_preset_voices())"
VIENEU_VOICES: List[Dict[str, Any]] = [
    {"id": "vieneu:Ngọc Lan", "voice_id": "Ngọc Lan", "name": "Ngọc Lan - nữ, giọng dịu dàng (VieNeu)", "locale": "vi-VN", "gender": "female"},
    {"id": "vieneu:Gia Bảo", "voice_id": "Gia Bảo", "name": "Gia Bảo - nam, giọng mượt mà (VieNeu)", "locale": "vi-VN", "gender": "male"},
    {"id": "vieneu:Thái Sơn", "voice_id": "Thái Sơn", "name": "Thái Sơn - nam, giọng chắc khỏe (VieNeu)", "locale": "vi-VN", "gender": "male"},
    {"id": "vieneu:Đức Trí", "voice_id": "Đức Trí", "name": "Đức Trí - nam, giọng rõ ràng (VieNeu)", "locale": "vi-VN", "gender": "male"},
    {"id": "vieneu:Mỹ Duyên", "voice_id": "Mỹ Duyên", "name": "Mỹ Duyên - nữ, giọng mượt mà (VieNeu)", "locale": "vi-VN", "gender": "female"},
    {"id": "vieneu:Trúc Ly", "voice_id": "Trúc Ly", "name": "Trúc Ly - nữ, giọng trẻ trung (VieNeu)", "locale": "vi-VN", "gender": "female"},
    {"id": "vieneu:Xuân Vĩnh", "voice_id": "Xuân Vĩnh", "name": "Xuân Vĩnh - nam, giọng vui tươi (VieNeu)", "locale": "vi-VN", "gender": "male"},
    {"id": "vieneu:Trọng Hữu", "voice_id": "Trọng Hữu", "name": "Trọng Hữu - nam, giọng uyên bác (VieNeu)", "locale": "vi-VN", "gender": "male"},
    {"id": "vieneu:Bình An", "voice_id": "Bình An", "name": "Bình An - nam, giọng điềm đạm (VieNeu)", "locale": "vi-VN", "gender": "male"},
    {"id": "vieneu:Ngọc Linh", "voice_id": "Ngọc Linh", "name": "Ngọc Linh - nữ, giọng tươi sáng (VieNeu)", "locale": "vi-VN", "gender": "female"},
]


# ---------------------------------------------------------------------------
# Google Cloud TTS voices (cloud neural, Northern Vietnamese)
# ---------------------------------------------------------------------------
# Selected by id "gcloud:<google_voice_name>" and synthesized by
# `pixelle_video.services.gcloud_tts_service.GCloudTTSEngine`. Two families:
# - Chirp3-HD (PRIMARY): modern generative voices, lively/dynamic delivery from PLAIN
#   TEXT (they ignore SSML). Default — WaveNet+SSML came out too flat/đều đều.
# - WaveNet (FALLBACK): driven by the SSML prosody layer (`pixelle_video.services.prosody`,
#   <prosody>/<emphasis>/<break>). Kept selectable for comparison.
# The synthesis path is chosen automatically via `gcloud_voice_supports_ssml()`.
# `voice_id` is the Google voice name passed to the API; `name` is the UI label.
GCLOUD_VOICES: List[Dict[str, Any]] = [
    # Chirp3-HD — auditioned & chosen 2026-06-22 (Northern male + female).
    {"id": "gcloud:vi-VN-Chirp3-HD-Enceladus", "voice_id": "vi-VN-Chirp3-HD-Enceladus", "name": "Nam miền Bắc (Google Chirp3-HD)", "locale": "vi-VN", "gender": "male"},
    {"id": "gcloud:vi-VN-Chirp3-HD-Despina", "voice_id": "vi-VN-Chirp3-HD-Despina", "name": "Nữ miền Bắc (Google Chirp3-HD)", "locale": "vi-VN", "gender": "female"},
    # WaveNet — SSML fallback (kept for comparison).
    {"id": "gcloud:vi-VN-Wavenet-D", "voice_id": "vi-VN-Wavenet-D", "name": "Nam miền Bắc (Google WaveNet)", "locale": "vi-VN", "gender": "male"},
    {"id": "gcloud:vi-VN-Wavenet-C", "voice_id": "vi-VN-Wavenet-C", "name": "Nữ miền Bắc (Google WaveNet)", "locale": "vi-VN", "gender": "female"},
]


# Voices offered for selection in the "Voiceover" (section.tts) UI sections.
# The Google Cloud neural voices (Chirp3-HD primary + WaveNet fallback, Northern,
# male+female) are the active voiceover engine. The VieNeu presets above stay registered so any previously-saved
# `vieneu:` id still resolves via resolve_vieneu_voice(), but they are no longer in
# the picker. Order here is the order shown in the dropdown.
VOICEOVER_VOICES: List[Dict[str, Any]] = list(GCLOUD_VOICES)


def list_custom_voices() -> List[Dict[str, Any]]:
    """
    Scan the ``voices/`` directory for reference clips and expose each one as a
    selectable cloned voice.

    Dropping ``voices/<name>.mp3`` makes a voice ``clone:<name>`` available; at
    synthesis time it is produced with the Edge base voice ``DEFAULT_CLONE_BASE``
    and converted to the reference timbre via OpenVoice.

    Returns:
        List of voice configs with keys: id, name, ref, base, locale, gender.
    """
    voices_dir = Path(VOICES_DIR)
    if not voices_dir.is_dir():
        return []

    custom: List[Dict[str, Any]] = []
    for path in sorted(voices_dir.iterdir()):
        if path.name.startswith(".") or not path.is_file():
            continue
        if path.suffix.lower() not in _CLONE_AUDIO_EXTS:
            continue
        stem = path.stem
        custom.append({
            "id": f"{CLONE_VOICE_PREFIX}{stem}",
            "name": f"{stem} (clone)",
            "ref": str(path).replace("\\", "/"),
            # Base TTS that provides pronunciation/accent before timbre conversion.
            # gTTS Vietnamese is Northern (Hanoi); the Edge vi-VN voices are Southern,
            # so we use gTTS as the base for a Northern clone.
            "base_engine": "gtts",
            "base_lang": "vi",
            "base": DEFAULT_CLONE_BASE,  # Edge fallback (Southern) if base_engine="edge"
            "locale": "vi-VN",
            # Gender is unknown for an arbitrary dropped clip and unused by the default
            # f5tts engine (it clones gender straight from the reference).
            "gender": "unknown",
        })
    return custom


def resolve_custom_voice(voice_id: Optional[str]) -> Optional[Dict[str, Any]]:
    """
    Return the custom-voice config for a ``clone:<name>`` id, or None if the id is
    not a cloned voice (e.g. a normal Edge voice).
    """
    if not voice_id or not voice_id.startswith(CLONE_VOICE_PREFIX):
        return None
    return next((v for v in list_custom_voices() if v["id"] == voice_id), None)


def resolve_vieneu_voice(voice_id: Optional[str]) -> Optional[Dict[str, Any]]:
    """
    Return the VieNeu preset config for a ``vieneu:<name>`` id, or None if the id is
    not a VieNeu voice (e.g. a normal Edge or cloned voice).
    """
    if not voice_id or not voice_id.startswith(VIENEU_VOICE_PREFIX):
        return None
    return next((v for v in VIENEU_VOICES if v["id"] == voice_id), None)


def resolve_gcloud_voice(voice_id: Optional[str]) -> Optional[Dict[str, Any]]:
    """
    Return the Google Cloud TTS voice config for a ``gcloud:<name>`` id, or None if
    the id is not a Google Cloud voice (e.g. a VieNeu, cloned, or Edge voice).
    """
    if not voice_id or not voice_id.startswith(GCLOUD_VOICE_PREFIX):
        return None
    return next((v for v in GCLOUD_VOICES if v["id"] == voice_id), None)


# Google voice families that do NOT support SSML — they ignore <prosody>/<emphasis>/
# <break> and must be fed plain text. Everything else (WaveNet/Standard/Neural2) does.
_NON_SSML_GCLOUD_MARKERS = ("chirp", "journey")


def gcloud_voice_supports_ssml(voice_name: str) -> bool:
    """True if a Google voice renders SSML; False for Chirp3-HD/Journey (plain text only).

    ``voice_name`` is the bare Google voice id (e.g. ``vi-VN-Wavenet-D`` or
    ``vi-VN-Chirp3-HD-Aoede``), not the ``gcloud:`` prefixed id. Detection is by name
    family so voices added later are classified correctly without a registry change.
    """
    name = (voice_name or "").lower()
    return not any(marker in name for marker in _NON_SSML_GCLOUD_MARKERS)


def get_voice_display_name(voice_id: str, tr_func=None, locale: str = "zh_CN") -> str:
    """
    Get display name for voice

    Args:
        voice_id: Voice ID (e.g., "zh-CN-YunjianNeural")
        tr_func: Translation function (optional)
        locale: Current locale (default: "zh_CN")

    Returns:
        Display name (translated label if in Chinese, otherwise voice ID)
    """
    # Cloned voices: show their friendly name (e.g. "quan (clone)")
    custom = resolve_custom_voice(voice_id)
    if custom:
        return custom["name"]

    # Google Cloud TTS voices: show their friendly Vietnamese name
    gcloud = resolve_gcloud_voice(voice_id)
    if gcloud:
        return gcloud["name"]

    # VieNeu preset voices: show their friendly Vietnamese name
    vieneu = resolve_vieneu_voice(voice_id)
    if vieneu:
        return vieneu["name"]

    # Find voice config
    voice_config = next((v for v in EDGE_TTS_VOICES if v["id"] == voice_id), None)

    if not voice_config:
        return voice_id
    
    # If Chinese locale and translation function available, use translated label
    if locale == "zh_CN" and tr_func:
        label_key = voice_config["label_key"]
        return tr_func(label_key)
    
    # For other locales, return voice ID
    return voice_id


def speed_to_rate(speed: float) -> str:
    """
    Convert speed multiplier to Edge TTS rate parameter
    
    Args:
        speed: Speed multiplier (1.0 = normal, 1.2 = 120%)
    
    Returns:
        Rate string (e.g., "+20%", "-10%")
    
    Examples:
        1.0 → "+0%"
        1.2 → "+20%"
        0.8 → "-20%"
    """
    percentage = int((speed - 1.0) * 100)
    sign = "+" if percentage >= 0 else ""
    return f"{sign}{percentage}%"

