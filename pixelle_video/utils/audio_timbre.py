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
Acoustic *timbre* fingerprinting for narration segments.

Why this exists: VieNeu-TTS uses a stochastic token sampler, so independent segments of
the same video occasionally drift to a different-sounding voice (see
``pixelle_video.services.vieneu_service``). We can't *listen* to check, so instead we
fingerprint each segment's timbre — pitch (f0) + spectral envelope — and measure how far
a segment sits from the others. A drifted segment shows up as a clear numeric outlier.

The same fingerprint serves two callers:
  * the one-off repair script (``scripts/regen_segment.py``) — rank candidate re-takes of
    a bad segment against the other (good) segments and pick the closest match;
  * the pipeline's consistency check — compare each freshly synthesized segment against an
    anchor built from the already-accepted segments and re-roll outliers.

``features(path)`` returns a fixed-length vector. It prefers ``librosa`` (MFCC + YIN
pitch) when importable and falls back to a NumPy-only spectral fingerprint otherwise.
Loudness is normalized away (we compare *timbre*, not level). All paths compared together
MUST be fingerprinted by the same process/extractor so the vectors are commensurable.
"""

import subprocess
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union

import numpy as np
from loguru import logger

PathLike = Union[str, Path]

# Analysis sample rate. 16 kHz captures speaker pitch + formants while keeping FFTs cheap.
_SR = 16000


# --------------------------------------------------------------------------- #
# Decoding
# --------------------------------------------------------------------------- #
def _load_mono(path: PathLike, sr: int = _SR) -> np.ndarray:
    """Decode any audio file to a mono float32 waveform at ``sr`` via ffmpeg."""
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path),
         "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"],
        capture_output=True,
    )
    if proc.returncode != 0:
        err = proc.stderr.decode(errors="ignore")[:500]
        raise RuntimeError(f"ffmpeg failed to decode {path}: {err}")
    y = np.frombuffer(proc.stdout, dtype=np.float32).astype(np.float64)
    if y.size == 0:
        raise RuntimeError(f"decoded empty audio: {path}")
    # Remove DC and normalize peak so the fingerprint reflects timbre, not loudness.
    y = y - float(np.mean(y))
    peak = float(np.max(np.abs(y)))
    return y / peak if peak > 0 else y


# --------------------------------------------------------------------------- #
# Feature extraction
# --------------------------------------------------------------------------- #
def features(path: PathLike) -> np.ndarray:
    """Return a fixed-length timbre fingerprint for one audio file.

    Uses librosa (MFCC means + YIN pitch stats) when available, else a NumPy spectral
    fingerprint. Raises on decode failure; never returns NaNs.
    """
    y = _load_mono(path)
    try:
        feat = _librosa_features(y, _SR)
    except ImportError:
        feat = _numpy_features(y, _SR)
    except Exception as e:  # librosa present but choked — degrade gracefully
        logger.debug(f"librosa feature extraction failed ({e!r}); using numpy fallback")
        feat = _numpy_features(y, _SR)
    return np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)


def _librosa_features(y: np.ndarray, sr: int) -> np.ndarray:
    """13 MFCC means (spectral envelope) + median/IQR of log-f0 (speaker pitch)."""
    import librosa  # noqa: PLC0415 — optional dependency, imported lazily

    mfcc = librosa.feature.mfcc(y=y.astype(np.float32), sr=sr, n_mfcc=13)
    f0 = librosa.yin(y.astype(np.float32), fmin=65, fmax=400, sr=sr, frame_length=1024)
    logf0 = np.log(np.clip(f0, 1e-3, None))
    pitch = [float(np.median(logf0)),
             float(np.percentile(logf0, 75) - np.percentile(logf0, 25))]
    return np.concatenate([mfcc.mean(axis=1), pitch]).astype(np.float64)


def _numpy_features(y: np.ndarray, sr: int, frame: int = 1024, hop: int = 512) -> np.ndarray:
    """Pure-NumPy fallback: mean/std of spectral centroid, bandwidth, rolloff, flatness, ZCR."""
    if len(y) < frame:
        y = np.pad(y, (0, frame - len(y)))
    n_frames = 1 + (len(y) - frame) // hop
    starts = hop * np.arange(n_frames)
    idx = starts[:, None] + np.arange(frame)[None, :]
    frames = y[idx] * np.hanning(frame)[None, :]

    mag = np.abs(np.fft.rfft(frames, axis=1)) + 1e-9
    freqs = np.fft.rfftfreq(frame, 1.0 / sr)

    msum = mag.sum(axis=1)
    centroid = (mag * freqs[None, :]).sum(axis=1) / msum
    bandwidth = np.sqrt((mag * (freqs[None, :] - centroid[:, None]) ** 2).sum(axis=1) / msum)
    csum = np.cumsum(mag, axis=1)
    rolloff = freqs[(csum >= 0.85 * csum[:, -1:]).argmax(axis=1)]
    flatness = np.exp(np.mean(np.log(mag), axis=1)) / np.mean(mag, axis=1)
    zcr = np.mean(np.abs(np.diff(np.sign(frames), axis=1)) > 0, axis=1)

    feats: List[float] = []
    for arr in (centroid, bandwidth, rolloff, flatness, zcr):
        feats += [float(np.mean(arr)), float(np.std(arr))]
    return np.asarray(feats, dtype=np.float64)


# --------------------------------------------------------------------------- #
# Similarity model
# --------------------------------------------------------------------------- #
class TimbreModel:
    """A reference timbre built from one or more "known-good" segments.

    ``score(feat)`` is the standardized Euclidean distance from the reference centroid:
    each feature is z-scored by the reference set's own mean/std, so all dimensions
    contribute comparably. Lower = more similar to the reference voice.
    """

    def __init__(self, reference_feats: Sequence[np.ndarray]):
        ref = np.vstack([np.asarray(f, dtype=np.float64) for f in reference_feats])
        self._mu = ref.mean(axis=0)
        sd = ref.std(axis=0)
        sd[sd == 0] = 1.0  # constant dims carry no information
        self._sd = sd
        self._z = (ref - self._mu) / self._sd
        self._centroid = self._z.mean(axis=0)

    def score(self, feat: np.ndarray) -> float:
        z = (np.asarray(feat, dtype=np.float64) - self._mu) / self._sd
        return float(np.linalg.norm(z - self._centroid))

    def reference_scores(self) -> np.ndarray:
        """Distances of the reference members themselves — the 'normal' spread."""
        return np.linalg.norm(self._z - self._centroid, axis=1)

    def is_outlier(self, feat: np.ndarray, n_sigma: float = 3.0) -> bool:
        """True if ``feat`` lies more than ``n_sigma`` beyond the reference spread."""
        ref = self.reference_scores()
        threshold = float(ref.mean() + n_sigma * (ref.std() or 1e-9))
        return self.score(feat) > threshold


def rank(reference_feats: Sequence[np.ndarray],
         candidate_feats: Dict[str, np.ndarray]) -> List[Tuple[str, float]]:
    """Rank named candidates by similarity to the reference set (closest first)."""
    model = TimbreModel(reference_feats)
    scored = [(name, model.score(feat)) for name, feat in candidate_feats.items()]
    return sorted(scored, key=lambda t: t[1])
