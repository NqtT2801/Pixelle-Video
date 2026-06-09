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
One-time setup for the F5-TTS Vietnamese voice-cloning engine.

Downloads the F5-TTS Vietnamese checkpoint (hynt/F5-TTS-Vietnamese-ViVoice) into
``models/f5tts_vi/``: ``model_last.pt`` plus the vocab (the repo's ``config.json``
is actually the F5 vocab; we save it as ``vocab.txt``). Idempotent.

Model licence: CC-BY-NC-SA-4.0 (non-commercial research use).

Usage:
    .venv/Scripts/python.exe scripts/setup_f5tts_vi.py
"""

import os
import shutil
import sys

F5_DIR = os.path.join("models", "f5tts_vi")
HF_REPO = "hynt/F5-TTS-Vietnamese-ViVoice"


def main() -> int:
    ckpt = os.path.join(F5_DIR, "model_last.pt")
    vocab = os.path.join(F5_DIR, "vocab.txt")
    if os.path.exists(ckpt) and os.path.exists(vocab):
        print(f"[ok] F5-TTS Vietnamese already present in {F5_DIR}")
        return 0

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("huggingface_hub is required.", file=sys.stderr)
        return 1

    os.makedirs(F5_DIR, exist_ok=True)
    print(f"[..] Downloading {HF_REPO} (~1.3GB on first run) ...")

    if not os.path.exists(ckpt):
        p = hf_hub_download(repo_id=HF_REPO, filename="model_last.pt", local_dir=F5_DIR)
        print(f"   - model_last.pt -> {p}")

    # The repo's config.json IS the F5 vocab file; save it as vocab.txt
    if not os.path.exists(vocab):
        p = hf_hub_download(repo_id=HF_REPO, filename="config.json", local_dir=F5_DIR)
        shutil.copyfile(p, vocab)
        print(f"   - config.json -> vocab.txt")

    print(f"[ok] Done. F5-TTS Vietnamese files in {F5_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
