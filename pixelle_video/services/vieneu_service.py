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
VieNeu-TTS (Vietnamese) named-voice engine.

VieNeu-TTS is a *self-contained* Vietnamese TTS engine: it speaks the text in one of
its built-in **preset voices** (selected by name), unlike the clone engines
(F5-TTS / viXTTS / kNN-VC / OpenVoice) which reconstruct a voice from a reference clip.
So it is wired in like Edge TTS — text + voice name -> audio — see
``pixelle_video.tts_voices.VIENEU_VOICES`` for the registered presets.

Uses the default ``mode="v3turbo"`` (48 kHz). On CPU it runs **torch-free via ONNX
Runtime**; on a CUDA machine the package auto-selects the PyTorch backend. The model is
downloaded from HuggingFace (``pnnbao-ump/VieNeu-TTS-v3-Turbo``) on first use.

Package licence: Apache-2.0.
"""

import os
from typing import List, Optional, Tuple

from loguru import logger


class VieNeuEngine:
    """
    Lazy singleton wrapper around VieNeu-TTS (v3 Turbo).

    The model is loaded on first use (and downloaded from HuggingFace the very first
    time). Subsequent calls reuse the same in-memory instance.
    """

    _instance: Optional["VieNeuEngine"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._model = None
        return cls._instance

    # ------------------------------------------------------------------ #
    # Lazy model loading
    # ------------------------------------------------------------------ #
    def _ensure_model(self):
        if self._model is not None:
            return self._model

        try:
            from vieneu import Vieneu
        except ImportError as e:
            raise ImportError(
                "VieNeu-TTS requires 'vieneu'. Install it with: "
                "uv pip install --python .venv/Scripts/python.exe vieneu"
            ) from e

        logger.info(
            "🎚️  Loading VieNeu-TTS v3 Turbo (CPU/ONNX, torch-free; "
            "first call downloads the model)"
        )
        # mode defaults to 'v3turbo'; backend 'auto' -> ONNX on CPU, PyTorch on CUDA.
        self._model = Vieneu()
        logger.info("✅ VieNeu-TTS v3 Turbo ready")
        return self._model

    # ------------------------------------------------------------------ #
    # Preset voices
    # ------------------------------------------------------------------ #
    def list_preset_voices(self) -> List[Tuple[str, str]]:
        """Return ``[(label, voice_id), ...]`` for the built-in preset voices.

        Loads the model (first call only) — used for one-time registration/verification,
        not on the UI hot path (``pixelle_video.tts_voices.VIENEU_VOICES`` is the static
        list the UI reads).
        """
        return self._ensure_model().list_preset_voices()

    # ------------------------------------------------------------------ #
    # Synthesis
    # ------------------------------------------------------------------ #
    def synthesize(
        self,
        text: str,
        voice_id: str,
        output_wav: str,
        temperature: Optional[float] = None,
        seed: Optional[int] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
    ) -> str:
        """Synthesize ``text`` with the named preset ``voice_id``; write a 48 kHz wav.

        VieNeu has no speed control — tempo is adjusted by the caller (ffmpeg ``atempo``)
        when transcoding to the final mp3.

        VieNeu v3 preset voices are *anchored* (fixed reference codes + a reserved speaker
        token), so the timbre is pinned and all voice "drift" between segments of a video
        comes from the autoregressive token sampler. The sampler is controlled by
        ``temperature`` + ``top_k`` + ``top_p``: narrowing them (low temperature, small
        top_k, low top_p) keeps every segment on the same preset speaker, and
        ``temperature=0`` is fully deterministic (greedy argmax — no RNG). Left at None,
        each knob falls back to VieNeu's own permissive default (temp 0.8 / top_k 25 /
        top_p 0.95 / rep-pen 1.2), which drifts. The CPU/ONNX backend samples via NumPy's
        global RNG (``np.random.choice``) and the CUDA/PyTorch backend uses
        ``torch.multinomial`` — so ``seed`` pins both (only matters when temperature > 0).
        """
        model = self._ensure_model()
        os.makedirs(os.path.dirname(output_wav) or ".", exist_ok=True)

        if seed is not None:
            import numpy as np
            np.random.seed(seed)
            try:
                import torch
                torch.manual_seed(seed)
            except ImportError:
                pass  # torch-free ONNX backend: NumPy seed above is sufficient

        # emotion uses the package default ("natural"). Each sampler knob is forwarded only
        # when set, so a None falls back to VieNeu's own infer() default.
        infer_kwargs = {"voice": voice_id}
        if temperature is not None:
            infer_kwargs["temperature"] = temperature
        if top_k is not None:
            infer_kwargs["top_k"] = top_k
        if top_p is not None:
            infer_kwargs["top_p"] = top_p
        if repetition_penalty is not None:
            infer_kwargs["repetition_penalty"] = repetition_penalty
        wav = model.infer(text, **infer_kwargs)
        model.save(wav, output_wav)
        logger.info(f"✅ VieNeu synthesized '{voice_id}': {output_wav}")
        return output_wav
