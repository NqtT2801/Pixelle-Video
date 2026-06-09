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
F5-TTS (Vietnamese) voice cloning engine.

F5-TTS clones a voice **in-context** from a short reference clip: it follows the
reference's timbre AND accent/intonation directly, then speaks the new text. Unlike
voice-conversion (OpenVoice / kNN-VC, which reconstruct from the reference and smear
the Northern accent) or corpus-biased TTS (viXTTS, which is Southern), F5-TTS with a
Northern reference (voices/quan.mp3) produces a faithful Northern Hanoi clone.

Uses the Vietnamese fine-tune `hynt/F5-TTS-Vietnamese-ViVoice` (downloaded by
scripts/setup_f5tts_vi.py). The reference segment + its transcript are extracted and
cached once; per-line synthesis is just F5-TTS inference.

Model licence: CC-BY-NC-SA-4.0 (non-commercial).
"""

import os
from pathlib import Path
from typing import Optional, Tuple

from loguru import logger

F5_DIR = "models/f5tts_vi"
CACHE_DIR = "voices/.cache"
# Reference window for in-context cloning. F5 re-encodes the reference at every
# diffusion step, so a shorter clip is markedly faster on CPU (the ref dominates the
# sequence length). ~7s keeps identity while cutting compute vs the old 11s.
DEFAULT_REF_SEC = 7.0
REF_MIN_SEC = 4.0


class F5TTSEngine:
    """
    Lazy singleton wrapper around F5-TTS (Vietnamese).

    The model is loaded on first use; the reference segment and its transcript are
    cached to ``voices/.cache/`` so they are only computed once per reference clip.
    """

    _instance: Optional["F5TTSEngine"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._model = None
            cls._instance._device = None
            cls._instance._quantized = False
            cls._instance._ref_cache = {}
        return cls._instance

    # ------------------------------------------------------------------ #
    # Lazy model loading
    # ------------------------------------------------------------------ #
    def _ensure_model(self, quantize: bool = False):
        if self._model is not None:
            return self._model

        ckpt = os.path.join(F5_DIR, "model_last.pt")
        vocab = os.path.join(F5_DIR, "vocab.txt")
        if not (os.path.exists(ckpt) and os.path.exists(vocab)):
            raise FileNotFoundError(
                f"F5-TTS Vietnamese checkpoints not found under '{F5_DIR}'. "
                "Run: python scripts/setup_f5tts_vi.py"
            )

        try:
            import torch
            from f5_tts.api import F5TTS
        except ImportError as e:
            raise ImportError(
                "F5-TTS requires 'f5-tts'. Install it with: "
                "uv pip install --python .venv/Scripts/python.exe f5-tts  "
                "(then: uv pip uninstall torchcodec  # broken on Windows)"
            ) from e

        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        # Use all CPU cores for inference (torch defaults to physical cores only)
        if self._device == "cpu":
            try:
                torch.set_num_threads(os.cpu_count() or torch.get_num_threads())
            except Exception:
                pass

        logger.info(
            f"🎚️  Loading F5-TTS Vietnamese on {self._device} "
            f"(threads={torch.get_num_threads()}, first call is slow)"
        )

        self._model = F5TTS(
            model="F5TTS_Base",
            ckpt_file=ckpt,
            vocab_file=vocab,
            device=self._device,
        )

        # Optional int8 dynamic quantization (CPU) — speeds up the Linear-heavy DiT
        if quantize and self._device == "cpu" and not self._quantized:
            try:
                import torch.ao.quantization as tq
                self._model.ema_model = tq.quantize_dynamic(
                    self._model.ema_model, {torch.nn.Linear}, dtype=torch.qint8
                )
                self._quantized = True
                logger.info("⚡ Applied int8 dynamic quantization to F5-TTS")
            except Exception as e:
                logger.warning(f"F5-TTS quantization skipped: {e}")

        return self._model

    # ------------------------------------------------------------------ #
    # Reference (segment + transcript), cached
    # ------------------------------------------------------------------ #
    def _reference(self, ref_path: str, ref_sec: float = DEFAULT_REF_SEC) -> Tuple[str, str]:
        """Return (ref_wav_path, ref_text) for the clip, cached by file stem.

        ``ref_sec`` is only used on a cache miss; delete ``voices/.cache/<stem>.f5ref.*``
        to re-extract with a different length.
        """
        stem = Path(ref_path).stem
        if stem in self._ref_cache:
            return self._ref_cache[stem]

        Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)
        ref_wav = os.path.join(CACHE_DIR, f"{stem}.f5ref.wav")
        ref_txt = os.path.join(CACHE_DIR, f"{stem}.f5ref.txt")

        if not os.path.exists(ref_wav):
            self._extract_ref_segment(ref_path, ref_wav, ref_sec)
        if os.path.exists(ref_txt):
            ref_text = open(ref_txt, encoding="utf-8").read().strip()
        else:
            ref_text = self._transcribe(ref_wav)
            with open(ref_txt, "w", encoding="utf-8") as f:
                f.write(ref_text)

        self._ref_cache[stem] = (ref_wav, ref_text)
        return ref_wav, ref_text

    def _extract_ref_segment(self, ref_path: str, out_wav: str, ref_sec: float = DEFAULT_REF_SEC):
        """Extract a clean voiced segment (~ref_sec) from the reference clip."""
        import librosa
        import numpy as np
        import soundfile as sf

        ref_sec = max(REF_MIN_SEC, float(ref_sec))
        y, sr = librosa.load(ref_path, sr=24000, mono=True)
        intervals = librosa.effects.split(y, top_db=30)
        seg = None
        for s, e in intervals:
            if e - s >= int(REF_MIN_SEC * sr):
                seg = y[s:min(e, s + int(ref_sec * sr))]
                break
        if seg is None:
            buf, tot = [], 0
            for s, e in intervals:
                buf.append(y[s:e]); tot += (e - s)
                if tot >= int(ref_sec * sr):
                    break
            seg = np.concatenate(buf) if buf else y[: int(ref_sec * sr)]
        seg = seg[: int(ref_sec * sr)]
        sf.write(out_wav, seg, sr)
        logger.info(f"🎯 F5-TTS reference segment: {len(seg) / sr:.1f}s")

    def _transcribe(self, wav_path: str) -> str:
        """Transcribe the reference (Vietnamese) with transformers Whisper, once."""
        logger.info("📝 Transcribing F5-TTS reference (one-time, Whisper)")
        from transformers import pipeline
        asr = pipeline("automatic-speech-recognition",
                       model="openai/whisper-small", device=-1)
        res = asr(wav_path, generate_kwargs={"language": "vietnamese", "task": "transcribe"})
        return res["text"].strip().lower()

    # ------------------------------------------------------------------ #
    # Synthesis
    # ------------------------------------------------------------------ #
    def synthesize(
        self,
        text: str,
        ref_path: str,
        output_wav: str,
        speed: float = 1.0,
        nfe_step: int = 16,
        ref_sec: float = DEFAULT_REF_SEC,
        quantize: bool = False,
    ) -> str:
        """Synthesize ``text`` in the cloned voice of ``ref_path``; write a wav.

        ``nfe_step`` (diffusion steps) and ``ref_sec`` (reference length) are the main
        speed/quality knobs; ``quantize`` enables int8 CPU inference.
        """
        model = self._ensure_model(quantize=quantize)
        ref_wav, ref_text = self._reference(ref_path, ref_sec)

        # Model was trained on lowercase text
        gen_text = " ".join(text.split()).lower()

        os.makedirs(os.path.dirname(output_wav) or ".", exist_ok=True)
        model.infer(
            ref_file=ref_wav,
            ref_text=ref_text,
            gen_text=gen_text,
            speed=speed,
            nfe_step=nfe_step,
            remove_silence=True,
            file_wave=output_wav,
            show_info=lambda *a, **k: None,  # suppress prints (Vietnamese -> Windows console crash)
        )
        logger.info(f"✅ F5-TTS synthesized '{Path(ref_path).stem}': {output_wav}")
        return output_wav
