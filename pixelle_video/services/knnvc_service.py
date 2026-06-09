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
kNN-VC voice conversion engine.

kNN-VC (Baas et al., "Voice Conversion With Just Nearest Neighbors") is a strong
zero-shot voice converter: it represents speech with WavLM features and replaces
each source frame with its nearest neighbours from a "matching set" built from the
target speaker's audio, then vocodes with HiFi-GAN.

We use it to turn Edge TTS output into a custom cloned voice: Edge TTS provides the
Vietnamese **pronunciation/accent** (we use a Northern Hanoi base voice), and kNN-VC
replaces only the **timbre** with that of a short reference clip (e.g.
``voices/quan.mp3``). Conversion preserves the source's linguistic content, so the
Northern accent is kept; only the voice identity changes. No training required.

The model is loaded from the ``bshall/knn-vc`` repo via ``torch.hub`` (downloads
WavLM + HiFi-GAN on first use). The target "matching set" is cached on disk so
per-line conversion is just a nearest-neighbour match + vocode.
"""

import os
from pathlib import Path
from typing import Optional

from loguru import logger

CACHE_DIR = "voices/.cache"
KNN_VC_REPO = "bshall/knn-vc"
SAMPLE_RATE = 16000  # WavLM / HiFi-GAN operate at 16 kHz


class KNNVCConverter:
    """
    Lazy singleton wrapper around kNN-VC (torch.hub ``bshall/knn-vc``).

    The model is created on first use; per-target matching sets are cached to
    ``voices/.cache/`` so they are only computed once per reference clip.
    """

    _instance: Optional["KNNVCConverter"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._model = None
            cls._instance._device = None
            cls._instance._matching_cache = {}
        return cls._instance

    # ------------------------------------------------------------------ #
    # Lazy model loading
    # ------------------------------------------------------------------ #
    def _ensure_model(self):
        if self._model is not None:
            return self._model

        try:
            import torch
        except ImportError as e:
            raise ImportError("kNN-VC requires 'torch'.") from e

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"🎚️  Loading kNN-VC ({KNN_VC_REPO}) on {self._device}")

        # Loads code + WavLM + HiFi-GAN from the bshall/knn-vc repo (first use
        # downloads checkpoints). prematched=True uses the prematched HiFi-GAN,
        # which gives cleaner conversions.
        self._model = torch.hub.load(
            KNN_VC_REPO, "knn_vc",
            prematched=True, trust_repo=True, pretrained=True,
            device=self._device,
        )
        return self._model

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _to_wav(self, src_path: str, tag: str) -> str:
        """Transcode any input audio to a 16 kHz mono wav for reliable loading."""
        Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)
        wav_path = os.path.join(CACHE_DIR, f"_knn_{tag}.wav")
        import ffmpeg
        (
            ffmpeg
            .input(src_path)
            .output(wav_path, ac=1, ar=SAMPLE_RATE, loglevel="error")
            .overwrite_output()
            .run()
        )
        return wav_path

    def matching_set(self, ref_path: str):
        """Target speaker matching set (WavLM features), cached by file stem."""
        import torch

        stem = Path(ref_path).stem
        if stem in self._matching_cache:
            return self._matching_cache[stem]

        cache_file = os.path.join(CACHE_DIR, f"{stem}.knnvc.pth")
        if os.path.exists(cache_file):
            ms = torch.load(cache_file, map_location=self._device or "cpu")
        else:
            model = self._ensure_model()
            logger.info(f"🎯 Building kNN-VC matching set from {ref_path}")
            ref_wav = self._to_wav(ref_path, f"ref_{stem}")
            ms = model.get_matching_set([ref_wav])
            Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)
            torch.save(ms.cpu() if hasattr(ms, "cpu") else ms, cache_file)
        self._matching_cache[stem] = ms
        return ms

    # ------------------------------------------------------------------ #
    # Conversion
    # ------------------------------------------------------------------ #
    def convert(
        self,
        src_audio_path: str,
        ref_path: str,
        output_wav: str,
        topk: int = 4,
    ) -> str:
        """
        Convert ``src_audio_path`` (Edge TTS output) so it keeps the same speech but
        adopts the timbre of ``ref_path``. Writes a 16 kHz wav.

        Args:
            topk: Number of nearest neighbours averaged per frame. Lower (~4) =
                  stronger identity; higher = smoother but less like the target.
        """
        import soundfile as sf

        model = self._ensure_model()
        ms = self.matching_set(ref_path)

        src_wav = self._to_wav(src_audio_path, "src")
        try:
            query_seq = model.get_features(src_wav)
            out_wav = model.match(query_seq, ms, topk=topk)
        finally:
            try:
                os.unlink(src_wav)
            except OSError:
                pass

        audio = out_wav.detach().cpu().numpy() if hasattr(out_wav, "detach") else out_wav
        os.makedirs(os.path.dirname(output_wav) or ".", exist_ok=True)
        sf.write(output_wav, audio, SAMPLE_RATE)
        logger.info(f"✅ kNN-VC converted to '{Path(ref_path).stem}': {output_wav}")
        return output_wav
