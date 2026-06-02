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

Defines the Vietnamese voices available for local TTS inference. Two providers
are supported:
  - "edge"      : Microsoft Edge TTS (online, the vi-VN-* voices)
  - "vietvoice" : VietVoice-TTS (offline ONNX, natural Northern voices)
"""

from typing import List, Dict, Any


# Vietnamese voice presets for local inference (Edge + VietVoice).
EDGE_TTS_VOICES: List[Dict[str, Any]] = [
    # Vietnamese Northern voices — VietVoice-TTS (local offline ONNX, provider: vietvoice)
    {
        "id": "vietvoice-vi-bac-nu",
        "label_key": "tts.voice.vietvoice-vi-bac-nu",
        "locale": "vi-VN",
        "gender": "female",
        "provider": "vietvoice",
        "params": {"gender": "female", "area": "northern"},
    },
    {
        "id": "vietvoice-vi-bac-nam",
        "label_key": "tts.voice.vietvoice-vi-bac-nam",
        "locale": "vi-VN",
        "gender": "male",
        "provider": "vietvoice",
        "params": {"gender": "male", "area": "northern"},
    },

    # Vietnamese voices — Microsoft Edge TTS (provider: edge)
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

# Alias: the list holds local voices from multiple providers (edge, vietvoice).
LOCAL_TTS_VOICES = EDGE_TTS_VOICES


def get_voice_display_name(voice_id: str, tr_func=None, locale: str = "zh_CN") -> str:
    """
    Get display name for voice

    Args:
        voice_id: Voice ID (e.g., "vietvoice-vi-bac-nam")
        tr_func: Translation function (optional)
        locale: Current locale (default: "zh_CN")

    Returns:
        Display name (translated label when available, otherwise voice ID)
    """
    # Find voice config
    voice_config = next((v for v in LOCAL_TTS_VOICES if v["id"] == voice_id), None)

    if not voice_config:
        return voice_id

    # Use the translated label whenever it resolves (tr() falls back to en_US,
    # then to the key itself). Shows a friendly name in every locale that has a
    # label, instead of the raw voice id.
    if tr_func:
        label_key = voice_config["label_key"]
        label = tr_func(label_key)
        if label and label != label_key:
            return label

    # No translation available: fall back to the raw voice ID
    return voice_id


def get_voice_provider(voice_id: str) -> str:
    """
    Return the local TTS provider for a voice id.

    Defaults to "edge" (Microsoft Edge TTS) for voices without an explicit
    provider; VietVoice voices return "vietvoice".
    """
    voice_config = next((v for v in LOCAL_TTS_VOICES if v["id"] == voice_id), None)
    if not voice_config:
        return "edge"
    return voice_config.get("provider", "edge")


def get_voice_params(voice_id: str) -> dict:
    """
    Return provider-specific synthesis params for a voice id.

    For VietVoice voices this is e.g. {"gender": "female", "area": "northern"};
    returns an empty dict for voices without params (e.g. Edge voices).
    """
    voice_config = next((v for v in LOCAL_TTS_VOICES if v["id"] == voice_id), None)
    if not voice_config:
        return {}
    return dict(voice_config.get("params", {}))


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
