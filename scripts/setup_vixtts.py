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
One-time setup for the viXTTS voice-cloning engine.

Downloads the viXTTS (XTTS-v2 fine-tuned for Vietnamese) checkpoints into
``models/vixtts/``. Idempotent: re-running skips existing files.

Usage:
    .venv/Scripts/python.exe scripts/setup_vixtts.py
"""

import os
import sys

VIXTTS_DIR = os.path.join("models", "vixtts")
HF_REPO = "capleaf/viXTTS"
FILES = ["model.pth", "config.json", "vocab.json"]


def main() -> int:
    have = [f for f in FILES if os.path.exists(os.path.join(VIXTTS_DIR, f))]
    if len(have) == len(FILES):
        print(f"[ok] viXTTS already present in {VIXTTS_DIR}")
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

    os.makedirs(VIXTTS_DIR, exist_ok=True)
    print(f"[..] Downloading viXTTS from {HF_REPO} (~2GB on first run) ...")
    for fname in FILES:
        dst = os.path.join(VIXTTS_DIR, fname)
        if os.path.exists(dst):
            print(f"   - {fname} (cached)")
            continue
        path = hf_hub_download(repo_id=HF_REPO, filename=fname, local_dir=VIXTTS_DIR)
        print(f"   - {fname} -> {path}")

    print(f"[ok] Done. viXTTS files in {VIXTTS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
