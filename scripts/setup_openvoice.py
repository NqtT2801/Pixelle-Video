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
One-time setup for local voice cloning (Edge TTS + OpenVoice v2).

Downloads the OpenVoice v2 ToneColorConverter checkpoints into
``models/openvoice/converter/``. Idempotent: re-running skips existing files.

Usage:
    .venv/Scripts/python.exe scripts/setup_openvoice.py
"""

import os
import sys

CONVERTER_DIR = os.path.join("models", "openvoice", "converter")
HF_REPO = "myshell-ai/OpenVoiceV2"
FILES = ["converter/config.json", "converter/checkpoint.pth"]


def main() -> int:
    config_path = os.path.join(CONVERTER_DIR, "config.json")
    ckpt_path = os.path.join(CONVERTER_DIR, "checkpoint.pth")
    if os.path.exists(config_path) and os.path.exists(ckpt_path):
        print(f"[ok] OpenVoice converter already present in {CONVERTER_DIR}")
        return 0

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print(
            "huggingface_hub is required. Install it with:\n"
            "  uv pip install --python .venv/Scripts/python.exe huggingface_hub",
            file=sys.stderr,
        )
        return 1

    os.makedirs(CONVERTER_DIR, exist_ok=True)
    print(f"[..] Downloading OpenVoice v2 converter from {HF_REPO} ...")
    for rel in FILES:
        dst = os.path.join("models", "openvoice", rel.replace("/", os.sep))
        if os.path.exists(dst):
            print(f"   - {rel} (cached)")
            continue
        path = hf_hub_download(
            repo_id=HF_REPO,
            filename=rel,
            local_dir=os.path.join("models", "openvoice"),
        )
        print(f"   - {rel} -> {path}")

    print(f"[ok] Done. Checkpoints in {CONVERTER_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
