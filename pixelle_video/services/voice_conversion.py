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
OpenVoice v2 tone-color (timbre) conversion.

Used to turn Edge-TTS output into a custom cloned voice: Edge TTS provides the
Vietnamese pronunciation/prosody, and this converter replaces only the timbre with
that of a short reference clip (e.g. ``voices/quan.mp3``). No training is required.

Speaker embeddings ("SE") are extracted once and cached on disk, so per-line
conversion is fast.
"""

import os
from pathlib import Path
from typing import Optional

from loguru import logger

# Converter checkpoints (downloaded by scripts/setup_openvoice.py)
CONVERTER_DIR = "models/openvoice/converter"
CACHE_DIR = "voices/.cache"


class OpenVoiceConverter:
    """
    Lazy singleton wrapper around OpenVoice v2's ToneColorConverter.

    Heavy objects (torch model) are created on first use. Target/source speaker
    embeddings are cached to ``voices/.cache/`` so they are only computed once.
    """

    _instance: Optional["OpenVoiceConverter"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._converter = None
            cls._instance._device = None
            cls._instance._target_se_cache = {}
            cls._instance._source_se_cache = {}
        return cls._instance

    # ------------------------------------------------------------------ #
    # Lazy model loading
    # ------------------------------------------------------------------ #
    def _ensure_converter(self):
        """Load the ToneColorConverter on first use."""
        if self._converter is not None:
            return self._converter

        config_path = os.path.join(CONVERTER_DIR, "config.json")
        ckpt_path = os.path.join(CONVERTER_DIR, "checkpoint.pth")
        if not os.path.exists(config_path) or not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                "OpenVoice converter checkpoints not found under "
                f"'{CONVERTER_DIR}'. Run: python scripts/setup_openvoice.py"
            )

        try:
            import torch
            from openvoice.api import ToneColorConverter
        except ImportError as e:
            raise ImportError(
                "Voice cloning requires 'torch' and 'openvoice'. Install them with: "
                "uv pip install --python .venv/Scripts/python.exe torch torchaudio "
                "--index-url https://download.pytorch.org/whl/cpu  &&  "
                "uv pip install git+https://github.com/myshell-ai/OpenVoice.git"
            ) from e

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"🎚️  Loading OpenVoice ToneColorConverter on {self._device}")

        converter = ToneColorConverter(config_path, device=self._device)
        converter.load_ckpt(ckpt_path)
        self._converter = converter
        return converter

    # ------------------------------------------------------------------ #
    # Speaker-embedding extraction (cached)
    # ------------------------------------------------------------------ #
    def _to_wav(self, src_path: str, tag: str) -> str:
        """Transcode any input audio to a mono wav for reliable loading."""
        Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)
        wav_path = os.path.join(CACHE_DIR, f"_{tag}.wav")
        import ffmpeg
        (
            ffmpeg
            .input(src_path)
            .output(wav_path, ac=1, ar=44100, loglevel="error")
            .overwrite_output()
            .run()
        )
        return wav_path

    def _extract_se(self, audio_path: str):
        """
        Extract a speaker embedding from a clip via the converter's reference
        encoder, averaging over voiced segments for a cleaner timbre.

        Silence/noise is dropped with ``librosa.effects.split`` (no Whisper/VAD
        network needed); the voiced chunks are averaged by the converter's
        ``extract_se``. Falls back to the whole clip if no segments are found.
        """
        import librosa
        import soundfile as sf

        converter = self._ensure_converter()

        Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)
        try:
            y, sr = librosa.load(audio_path, sr=44100, mono=True)
            intervals = librosa.effects.split(y, top_db=30)
        except Exception as e:
            logger.warning(f"Segmenting failed ({e}); using whole clip for SE")
            return converter.extract_se([audio_path])

        min_len = int(1.0 * sr)      # keep voiced chunks >= 1s
        max_total = int(30.0 * sr)   # cap total material used
        chunk_paths = []
        total = 0
        for i, (start, end) in enumerate(intervals):
            if end - start < min_len:
                continue
            seg = y[start:end]
            cp = os.path.join(CACHE_DIR, f"_seg_{i}.wav")
            sf.write(cp, seg, sr)
            chunk_paths.append(cp)
            total += (end - start)
            if total >= max_total:
                break

        if not chunk_paths:
            logger.warning("No voiced segments found; using whole clip for SE")
            return converter.extract_se([audio_path])

        logger.info(f"Extracting SE from {len(chunk_paths)} voiced segment(s)")
        try:
            return converter.extract_se(chunk_paths)
        finally:
            for cp in chunk_paths:
                try:
                    os.unlink(cp)
                except OSError:
                    pass

    def target_se(self, ref_path: str):
        """Speaker embedding of the reference (target) voice, cached by file stem."""
        import torch
        stem = Path(ref_path).stem
        if stem in self._target_se_cache:
            return self._target_se_cache[stem]

        cache_file = os.path.join(CACHE_DIR, f"{stem}.se.v2.pth")
        if os.path.exists(cache_file):
            se = torch.load(cache_file, map_location=self._device or "cpu")
        else:
            logger.info(f"🎯 Extracting target speaker embedding from {ref_path}")
            wav = self._to_wav(ref_path, f"ref_{stem}")
            se = self._extract_se(wav)
            Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)
            torch.save(se, cache_file)
        self._target_se_cache[stem] = se
        return se

    def source_se(self, base_voice: str, sample_wav: str):
        """
        Speaker embedding of the Edge base voice, cached per base voice id.

        Extracted once from the first base sample we generate; reused afterwards.
        """
        import torch
        if base_voice in self._source_se_cache:
            return self._source_se_cache[base_voice]

        cache_file = os.path.join(CACHE_DIR, f"src_{base_voice}.se.v2.pth")
        if os.path.exists(cache_file):
            se = torch.load(cache_file, map_location=self._device or "cpu")
        else:
            logger.info(f"🎙️  Extracting source embedding for base voice {base_voice}")
            wav = self._to_wav(sample_wav, f"src_{base_voice}")
            se = self._extract_se(wav)
            Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)
            torch.save(se, cache_file)
        self._source_se_cache[base_voice] = se
        return se

    # ------------------------------------------------------------------ #
    # Conversion
    # ------------------------------------------------------------------ #
    def convert(
        self,
        src_audio_path: str,
        base_voice: str,
        ref_path: str,
        output_path: str,
        tau: float = 0.3,
    ) -> str:
        """
        Convert ``src_audio_path`` (Edge TTS output for ``base_voice``) so it keeps
        the same speech but adopts the timbre of ``ref_path``.

        Args:
            tau: Conversion strength. Lower preserves the Edge base's clarity and
                 pronunciation/accent; higher pushes harder toward the reference
                 timbre (can muddy the accent).

        Returns the output path (overwrites ``output_path``).
        """
        converter = self._ensure_converter()
        tgt_se = self.target_se(ref_path)
        src_se = self.source_se(base_voice, src_audio_path)

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        converter.convert(
            audio_src_path=src_audio_path,
            src_se=src_se,
            tgt_se=tgt_se,
            output_path=output_path,
            tau=tau,
        )
        logger.info(f"✅ Voice-converted to '{Path(ref_path).stem}': {output_path}")
        return output_path
