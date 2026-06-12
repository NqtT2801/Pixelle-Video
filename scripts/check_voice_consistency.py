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
Demonstrate that lowering VieNeu's temperature removes the "voice drift" between segments.

VieNeu preset voices have an anchored timbre, so drift comes only from the autoregressive
sampler. We render one (known drift-prone) line under several sampler profiles, each with a
few different seeds, and report both the clip durations and how much the renderings differ
from one another (``audio_timbre`` distance). The freer the sampler, the more the seed
swings the voice:

  OLD     (temp 0.4, top_k 25, top_p 0.95)  — the previous pipeline default
  NEW     (temp 0.2, top_k 25, top_p 0.95)  — the configured default (lower temp, stable)
  GREEDY  (temp 0)                          — fully deterministic (argmax, no RNG)
  NARROW! (temp 0.2, top_k 10, top_p 0.8)   — cautionary: narrowing top_k/top_p starves the
                                              sampler and causes runaway generation (a clip
                                              that never reaches EOS). Hence we do NOT narrow.

Expect: NEW tighter than OLD; GREEDY ≈ 0 (identical); NARROW occasionally produces a
wildly-long runaway clip (visible in the durations column).

    .venv/Scripts/python.exe scripts/check_voice_consistency.py
"""

import itertools
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np

from pixelle_video.services.vieneu_service import VieNeuEngine
from pixelle_video.utils import audio_timbre as at

# Segment 16's narration — the line that drifted in output/20260611_203548_dc2e.
TEXT = "Ra tòa, cô ấy từ chối quyền nuôi con, còn nói cháu nội thì tự lo."
VOICE = "Đức Trí"
SEEDS = [11, 22, 33]
PROFILES = {
    "OLD     (t0.4/k25/p0.95)": dict(temperature=0.4, top_k=25, top_p=0.95),
    "NEW     (t0.2/k25/p0.95)": dict(temperature=0.2, top_k=25, top_p=0.95),
    "GREEDY  (t0)":             dict(temperature=0.0, top_k=25, top_p=0.95),
    "NARROW! (t0.2/k10/p0.8)":  dict(temperature=0.2, top_k=10, top_p=0.8),
}


def main():
    engine = VieNeuEngine()
    tmp = Path(tempfile.mkdtemp(prefix="vcheck_"))
    print(f"text : {TEXT!r}")
    print(f"voice: {VOICE}   seeds: {SEEDS}\n")
    print(f"{'profile':26} {'durations (s)':22} {'mean dist':>10} {'max':>8}")
    for name, kw in PROFILES.items():
        feats, durs = [], []
        for seed in SEEDS:
            out = tmp / f"{name.split()[0]}_{seed}.wav"
            engine.synthesize(TEXT, VOICE, str(out), seed=seed, **kw)
            y = at._load_mono(out)
            durs.append(round(len(y) / 16000.0, 2))
            feats.append(at.features(out))
        dists = [float(np.linalg.norm(feats[i] - feats[j]))
                 for i, j in itertools.combinations(range(len(feats)), 2)]
        print(f"{name:26} {str(durs):22} {np.mean(dists):10.2f} {np.max(dists):8.2f}")
    print("\nLower mean dist = the seed/sampler swings the voice less => less drift.")
    print("GREEDY ≈ 0 (identical). Watch NARROW's durations for a runaway (>>5s) clip.")


if __name__ == "__main__":
    main()
