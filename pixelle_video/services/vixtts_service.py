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
viXTTS voice cloning engine.

viXTTS is an XTTS-v2 model fine-tuned for Vietnamese (capleaf/viXTTS). Unlike
OpenVoice tone-color conversion (which only swaps timbre over a base TTS), viXTTS
performs true zero-shot cloning of both timbre and intonation directly from a short
reference clip (e.g. ``voices/quan.mp3``) and speaks Vietnamese natively.

Speaker conditioning latents are extracted once and cached on disk, so per-line
synthesis is just model inference.
"""

import os
from pathlib import Path
from typing import Optional

from loguru import logger

# Model checkpoints (downloaded by scripts/setup_vixtts.py)
VIXTTS_DIR = "models/vixtts"
CACHE_DIR = "voices/.cache"
SAMPLE_RATE = 24000


class ViXTTSEngine:
    """
    Lazy singleton wrapper around viXTTS (Coqui XTTS-v2, Vietnamese fine-tune).

    The torch model is created on first use; speaker conditioning latents are
    cached to ``voices/.cache/`` so they are only computed once per reference clip.
    """

    _instance: Optional["ViXTTSEngine"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._model = None
            cls._instance._device = None
            cls._instance._cond_cache = {}
        return cls._instance

    # ------------------------------------------------------------------ #
    # Lazy model loading
    # ------------------------------------------------------------------ #
    def _ensure_model(self):
        if self._model is not None:
            return self._model

        config_path = os.path.join(VIXTTS_DIR, "config.json")
        ckpt_path = os.path.join(VIXTTS_DIR, "model.pth")
        vocab_path = os.path.join(VIXTTS_DIR, "vocab.json")
        if not (os.path.exists(config_path) and os.path.exists(ckpt_path)
                and os.path.exists(vocab_path)):
            raise FileNotFoundError(
                f"viXTTS checkpoints not found under '{VIXTTS_DIR}'. "
                "Run: python scripts/setup_vixtts.py"
            )

        try:
            import torch
            from TTS.tts.configs.xtts_config import XttsConfig
            from TTS.tts.models.xtts import Xtts
        except ImportError as e:
            raise ImportError(
                "viXTTS requires 'coqui-tts'. Install it with: "
                "uv pip install --python .venv/Scripts/python.exe coqui-tts"
            ) from e

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"🎚️  Loading viXTTS on {self._device} (first call is slow)")

        # The base Coqui tokenizer doesn't register Vietnamese ('vi'); teach it so
        # viXTTS works (it shares the XTTS tokenizer).
        _patch_vi_tokenizer()

        config = XttsConfig()
        config.load_json(config_path)
        model = Xtts.init_from_config(config)
        model.load_checkpoint(
            config,
            checkpoint_path=ckpt_path,
            vocab_path=vocab_path,
            use_deepspeed=False,
        )
        model.to(self._device)
        model.eval()
        # Ensure a char limit exists for 'vi' (defensive; we split text ourselves)
        try:
            model.tokenizer.char_limits.setdefault("vi", 250)
        except Exception:
            pass
        self._model = model
        return model

    # ------------------------------------------------------------------ #
    # Speaker conditioning (cached)
    # ------------------------------------------------------------------ #
    def conditioning(self, ref_path: str):
        """Return (gpt_cond_latent, speaker_embedding) for a reference clip, cached."""
        import torch

        stem = Path(ref_path).stem
        if stem in self._cond_cache:
            return self._cond_cache[stem]

        cache_file = os.path.join(CACHE_DIR, f"{stem}.xtts.pth")
        if os.path.exists(cache_file):
            data = torch.load(cache_file, map_location=self._device or "cpu")
            latents = (data["gpt_cond_latent"], data["speaker_embedding"])
        else:
            model = self._ensure_model()
            logger.info(f"🎯 Extracting viXTTS conditioning from {ref_path}")
            gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
                audio_path=[ref_path]
            )
            Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)
            torch.save(
                {"gpt_cond_latent": gpt_cond_latent.cpu(),
                 "speaker_embedding": speaker_embedding.cpu()},
                cache_file,
            )
            latents = (gpt_cond_latent, speaker_embedding)

        self._cond_cache[stem] = latents
        return latents

    # ------------------------------------------------------------------ #
    # Synthesis
    # ------------------------------------------------------------------ #
    def synthesize(
        self,
        text: str,
        ref_path: str,
        output_wav: str,
        speed: float = 1.0,
        language: str = "vi",
    ) -> str:
        """
        Synthesize ``text`` in the cloned voice of ``ref_path`` and write a wav.

        We split the text into sentences ourselves (XTTS's built-in splitter has no
        Vietnamese char limit -> KeyError 'vi'), synthesize each chunk, and join with
        short pauses.

        Returns the output wav path.
        """
        import numpy as np
        import soundfile as sf

        model = self._ensure_model()
        gpt_cond_latent, speaker_embedding = self.conditioning(ref_path)

        chunks = _split_text(text)
        pause = np.zeros(int(0.25 * SAMPLE_RATE), dtype=np.float32)
        pieces = []
        for chunk in chunks:
            out = model.inference(
                chunk,
                language,
                gpt_cond_latent,
                speaker_embedding,
                temperature=0.7,
                speed=speed,
                enable_text_splitting=False,
            )
            pieces.append(np.asarray(out["wav"], dtype=np.float32))
            pieces.append(pause)

        wav = np.concatenate(pieces[:-1]) if pieces else np.zeros(1, dtype=np.float32)

        os.makedirs(os.path.dirname(output_wav) or ".", exist_ok=True)
        sf.write(output_wav, wav, SAMPLE_RATE)
        logger.info(f"✅ viXTTS synthesized '{Path(ref_path).stem}': {output_wav}")
        return output_wav


def _patch_vi_tokenizer():
    """
    Register Vietnamese in the Coqui XTTS tokenizer (it ships without 'vi').

    Routes 'vi' through a minimal cleaner (lowercase + whitespace collapse, with a
    best-effort number→words expansion) instead of the English-centric
    multilingual_cleaners whose per-language dicts lack 'vi'.
    """
    from TTS.tts.layers.xtts import tokenizer as tok

    if getattr(tok.VoiceBpeTokenizer, "_vi_patched", False):
        return

    _orig = tok.VoiceBpeTokenizer.preprocess_text

    def _preprocess_text(self, txt, lang):
        if lang == "vi":
            t = txt.replace('"', "")
            t = tok.lowercase(t)
            try:
                t = tok.expand_numbers_multilingual(t, "vi")
            except Exception:
                pass
            return tok.collapse_whitespace(t)
        return _orig(self, txt, lang)

    tok.VoiceBpeTokenizer.preprocess_text = _preprocess_text
    tok.VoiceBpeTokenizer._vi_patched = True


def _split_text(text: str, max_len: int = 200) -> list:
    """
    Split text into synthesis-sized chunks: by sentence punctuation first, then by
    commas / hard length for over-long sentences. Keeps chunks <= ~max_len chars.
    """
    import re

    text = " ".join(text.split())
    if not text:
        return [""]

    # Split into sentences on . ! ? … ; and newlines (keep it simple, language-agnostic)
    sentences = [s.strip() for s in re.split(r"(?<=[\.!\?…;])\s+|\n+", text) if s.strip()]
    if not sentences:
        sentences = [text]

    chunks = []
    for s in sentences:
        if len(s) <= max_len:
            chunks.append(s)
            continue
        # Over-long: split on commas, then hard-wrap by length
        buf = ""
        for part in re.split(r"(?<=,)\s+", s):
            if len(buf) + len(part) + 1 <= max_len:
                buf = f"{buf} {part}".strip()
            else:
                if buf:
                    chunks.append(buf)
                while len(part) > max_len:
                    chunks.append(part[:max_len])
                    part = part[max_len:]
                buf = part
        if buf:
            chunks.append(buf)

    return chunks or [text]
