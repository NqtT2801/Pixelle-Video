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
Verify Google Cloud TTS setup and pick the male/female Vietnamese voice by ear.

Run AFTER setting up credentials (see docs / the chat handoff):
    .venv/Scripts/python.exe scripts/check_gcloud_voices.py

It (1) lists every vi-VN voice Google actually offers (name + gender), and (2) renders
short listen-test mp3s into ``output/gcloud_samples/`` so you can choose the nicest
Northern male + female. The audition target is **Chirp3-HD** (modern, expressive,
plain-text — SSML is ignored by these voices); a couple of WaveNet voices are also
rendered through the SSML pipeline as a baseline to A/B against.
"""

import os
import sys

# Make the project importable when run as a plain script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pixelle_video.config import config_manager
from pixelle_video.services.gcloud_tts_service import GCloudTTSEngine
from pixelle_video.services.prosody import (
    build_ssml, plain_text, prosody_settings_from_config,
)

SAMPLE_TITLE = "Tôi đã mất *tất cả* chỉ trong một đêm!"
SAMPLE_BODY = (
    "Năm đó tôi vừa tròn hai mươi tuổi, ôm trong lòng một giấc mơ đổi đời. "
    "Liệu tôi có làm được không? Tôi đã tự hứa sẽ không bao giờ bỏ cuộc."
)

# WaveNet voices rendered (via SSML) as a baseline to A/B against the Chirp3-HD samples.
BASELINE_SSML = ["vi-VN-Wavenet-C", "vi-VN-Wavenet-D"]


def main() -> int:
    local_cfg = config_manager.config.to_dict()["comfyui"]["tts"]["local"]
    credentials_path = local_cfg.get("gcloud_credentials")
    settings = prosody_settings_from_config(local_cfg)

    engine = GCloudTTSEngine()
    client = engine._ensure_client(credentials_path)  # raises a clear error if creds missing

    # 1) List what Google actually offers for vi-VN (definitive names + genders).
    from google.cloud import texttospeech
    resp = client.list_voices(language_code="vi-VN")
    gender_by_name = {
        v.name: texttospeech.SsmlVoiceGender(v.ssml_gender).name for v in resp.voices
    }
    print("\n=== Available vi-VN voices (name — gender) ===")
    for name in sorted(gender_by_name):
        print(f"  {name:28s} {gender_by_name[name]}")
    available = set(gender_by_name)

    # Chirp3-HD voices are the audition target (modern, expressive, no SSML -> plain text).
    chirp_voices = sorted(n for n in available if "chirp3-hd" in n.lower())

    out_dir = os.path.join("output", "gcloud_samples")
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n=== Rendering samples into {out_dir}/ ===")
    if not chirp_voices:
        print("  (!) No Chirp3-HD vi-VN voices offered — upgrade google-cloud-texttospeech?")

    def render(name: str, label: str, *, use_ssml: bool, text: str, is_title: bool):
        out = os.path.join(out_dir, f"{name}.{label}.mp3")
        try:
            if use_ssml:
                engine.synthesize(
                    build_ssml(text, is_title=is_title, settings=settings),
                    name, out, is_ssml=True, credentials_path=credentials_path,
                )
            else:
                engine.synthesize(
                    plain_text(text), name, out,
                    is_ssml=False, pitch=None, credentials_path=credentials_path,
                )
            print(f"  ✓ {os.path.basename(out)}")
        except Exception as e:  # noqa: BLE001
            print(f"  (fail) {name} {label}: {e}")

    # 2a) Chirp3-HD: one plain-text body sample each; gender in the filename for sorting.
    for name in chirp_voices:
        gender = gender_by_name.get(name, "UNKNOWN")
        render(name, f"{gender}.body", use_ssml=False, text=SAMPLE_BODY, is_title=False)

    # 2b) WaveNet baseline (SSML): title + body, to compare against the Chirp3-HD samples.
    for name in BASELINE_SSML:
        if name not in available:
            print(f"  (skip) {name} — not offered by Google for vi-VN")
            continue
        render(name, "title", use_ssml=True, text=SAMPLE_TITLE, is_title=True)
        render(name, "body", use_ssml=True, text=SAMPLE_BODY, is_title=False)

    print(
        "\nDone. The Chirp3-HD files are named <voice>.<GENDER>.body.mp3 — listen, then "
        "tell me which Northern male + female you want (compare with the Wavenet baseline)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
