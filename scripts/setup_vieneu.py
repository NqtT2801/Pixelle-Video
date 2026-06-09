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
One-time setup for the VieNeu-TTS (Vietnamese) named-voice engine.

Warms the VieNeu-TTS v3 Turbo model (downloaded from HuggingFace on first use,
CPU/ONNX torch-free) and prints the available preset voices. Idempotent: re-running
just re-verifies the cache. The preset voices are registered in
``pixelle_video.tts_voices.VIENEU_VOICES`` and selectable as ``vieneu:<name>``.

Usage:
    .venv/Scripts/python.exe scripts/setup_vieneu.py
"""

import sys

# VieNeu voice names are Vietnamese; avoid a UnicodeEncodeError on the Windows console.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def main() -> int:
    try:
        from vieneu import Vieneu
    except ImportError:
        print(
            "vieneu is required. Install it with:\n"
            "  uv pip install --python .venv/Scripts/python.exe vieneu",
            file=sys.stderr,
        )
        return 1

    print("[..] Loading VieNeu-TTS v3 Turbo (downloads the model from HuggingFace on first run) ...")
    tts = Vieneu()  # mode='v3turbo'; CPU -> ONNX (torch-free), CUDA -> PyTorch
    voices = tts.list_preset_voices()
    print(f"[ok] VieNeu-TTS ready — {len(voices)} preset voices:")
    for label, voice_id in voices:
        print(f"   - vieneu:{voice_id}  ({label})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
