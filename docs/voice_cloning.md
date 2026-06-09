# Custom voice cloning

Turn any reference clip in `voices/` into a selectable narration voice. Engines are
selectable via `config.yaml` → `comfyui.tts.local.clone_engine`:

- **`f5tts`** (default, recommended) — **F5-TTS Vietnamese** (`hynt/F5-TTS-Vietnamese-ViVoice`).
  Clones the reference voice **in-context**: it follows the reference's timbre AND
  accent/intonation directly, then speaks the new text. With a Northern (Hanoi)
  reference this gives a faithful **Northern** clone — the only approach here that
  delivered both correct accent and identity. No training.
- **`knnvc`** — gTTS Northern base + kNN-VC voice conversion. Conversion tends to
  smear the Northern accent; weaker than f5tts here.
- **`openvoice`** — Edge base + OpenVoice tone-color conversion. Weakest identity.
- **`vixtts`** — viXTTS (XTTS-v2 VN). **Southern-biased accent regardless of the
  reference — do NOT use for Northern voices.**

All engines clone from the same `voices/<name>.mp3` and appear in the voice selector
as "<name> (clone)". For a Northern Hanoi voice, keep `clone_engine: f5tts`.

> Why the others struggled with a Northern voice: Microsoft Edge's vi-VN voices
> (`vi-VN-NamMinhNeural`, `vi-VN-HoaiMyNeural`) are **Southern**; gTTS Vietnamese is
> Northern but voice-conversion (OpenVoice/kNN-VC) reconstructs from the reference and
> loses the Northern character; viXTTS imposes its own Southern accent. F5-TTS
> generates directly from the reference, so the reference's Northern accent is kept.

---

## F5-TTS engine (default)

`clone_engine: f5tts`. Licence: **CC-BY-NC-SA-4.0** (non-commercial).

### Install (one-time)

```bash
uv pip install --python .venv/Scripts/python.exe f5-tts
# f5-tts pulls torchcodec, whose native DLLs do NOT load on Windows -> remove it
# (transformers/torchaudio fall back to soundfile):
uv pip uninstall --python .venv/Scripts/python.exe torchcodec
# download the Vietnamese checkpoint (~1.3GB) into models/f5tts_vi/:
.venv/Scripts/python.exe scripts/setup_f5tts_vi.py
```

(Requires torch < 2.9 — already pinned to 2.8 for this project.)

### Notes
- On first use it (a) extracts a clean voiced segment (`clone_ref_sec`, ~7 s) from the
  reference clip, (b) transcribes it once with transformers Whisper
  (`openai/whisper-small`, downloaded on demand), and (c) loads the F5 model. The
  segment + transcript are cached to `voices/.cache/<name>.f5ref.wav` / `.f5ref.txt` —
  delete them to re-extract after replacing the clip or changing `clone_ref_sec`.
- Implementation: `pixelle_video/services/f5tts_service.py` (`F5TTSEngine`) +
  `tts_service._call_f5tts`. Setup: `scripts/setup_f5tts_vi.py`.

### Speed on CPU
F5-TTS is slow on CPU (no GPU here). The model loads once per process; per-sentence
cost ≈ `nfe_step × (reference_frames + gen_frames)`, and the reference dominates. The
project ships tuned for speed (~6× faster than F5 defaults: ~46 s vs ~300 s for a ~3 s
sentence). Knobs in `config.yaml` → `comfyui.tts.local`:
- `clone_nfe_step` (shipped **8**): diffusion steps. 32 = best quality, 16 = balanced,
  8 = fastest (slight quality drop).
- `clone_ref_sec` (shipped **7**): reference length in seconds — **the dominant cost**.
  Shorter = faster (e.g. 5). After changing it, delete `voices/.cache/<name>.f5ref.*`.
- `clone_quantize` (shipped **true**): int8 dynamic CPU inference (~1.4× faster).
All CPU cores are used automatically (`torch.set_num_threads`). Still slow vs a GPU;
the one-time model load (~1 min) is paid once per app process.

### Loudness (voice level + BGM balance)
All local narration (every clone + Edge) is normalized to a standard loudness so voices
come out **equal and at a normal level**, independent of the reference clip's loudness:
- `comfyui.tts.local.target_lufs` (default **−16** LUFS). Implemented in
  `tts_service._normalize_loudness` as a TP-safe `loudnorm` pass + a corrective
  gain+limiter (plain `loudnorm` won't boost peaky speech to target).
- BGM is normalized **relative to the voice**: `add_bgm` loudnorm's the BGM to the same
  target, then applies `bgm_volume` as a voice-relative fraction. So **`bgm_volume`
  default 0.13 ≈ 17–18 dB below the voice** for any BGM file. Adjust the BGM slider to
  taste (higher = louder music).

---

## kNN-VC engine

`clone_engine: knnvc` — gTTS (Northern) base + kNN-VC timbre conversion. Fetches
kNN-VC from `bshall/knn-vc` via `torch.hub` (WavLM + HiFi-GAN ~1.2 GB on first use);
needs `gtts`. Matching set cached to `voices/.cache/<name>.knnvc.pth`. Implementation:
`pixelle_video/services/knnvc_service.py` + `tts_service._call_knnvc`.

---

## viXTTS engine (default)

### Install (one-time)

```bash
# XTTS fork. NOTE the version pins below matter on Windows:
uv pip install --python .venv/Scripts/python.exe coqui-tts
# coqui-tts needs transformers with isin_mps_friendly (removed in 5.0):
uv pip install --python .venv/Scripts/python.exe "transformers>=4.57,<5"
# torch must be <2.9, else coqui-tts requires torchcodec (no good Windows wheel):
uv pip install --python .venv/Scripts/python.exe "torch==2.8.0" "torchaudio==2.8.0" \
  --index-url https://download.pytorch.org/whl/cpu
```

If a `torch` downgrade reports "missing RECORD" / leaves a stale
`torch-<old>.dist-info`, delete that stale dist-info folder so
`importlib.metadata.version("torch")` resolves (transformers reads it).

Download the model (~2 GB) into `models/vixtts/`:

```bash
.venv/Scripts/python.exe scripts/setup_vixtts.py
```

### Notes
- Vietnamese isn't registered in the stock Coqui tokenizer; the engine monkeypatches
  it (`_patch_vi_tokenizer` in `pixelle_video/services/vixtts_service.py`) and splits
  sentences itself (XTTS's built‑in splitter has no `vi` char limit).
- Speaker conditioning latents are cached to `voices/.cache/<name>.xtts.pth` (computed
  once; delete to re-extract after replacing the clip).
- Licence: XTTS‑v2 / viXTTS use the Coqui Public Model Licence (non‑commercial).
- Implementation: `pixelle_video/services/vixtts_service.py` (`ViXTTSEngine`) +
  `tts_service._call_vixtts`.

---

## OpenVoice engine (fallback)

Set `clone_engine: openvoice`.

### Install (one-time)

Into the project `.venv` (uv-managed):

```bash
# CPU torch (a CUDA build also works and is faster)
uv pip install --python .venv/Scripts/python.exe torch torchaudio \
  --index-url https://download.pytorch.org/whl/cpu

# OpenVoice converter (no-deps to avoid the faster-whisper/av build that fails on
# Windows) + the few runtime deps the converter actually needs
uv pip install --python .venv/Scripts/python.exe --no-deps \
  git+https://github.com/myshell-ai/OpenVoice.git
uv pip install --python .venv/Scripts/python.exe \
  librosa huggingface_hub wavmark inflect unidecode eng-to-ipa pypinyin cn2an jieba
```

Download the converter checkpoints (~200 MB) into `models/openvoice/converter/`:

```bash
.venv/Scripts/python.exe scripts/setup_openvoice.py
```

## 2. Add a voice

Drop an audio file into `voices/`, e.g. `voices/quan.mp3` (20–60 s of clean speech
is plenty). It immediately appears in the TTS voice selector as **"quan (clone)"**
with id `clone:quan`. Any `.mp3/.wav/.flac/.m4a/.ogg` works.

The base Edge voice that drives pronunciation/prosody is
`DEFAULT_CLONE_BASE = "vi-VN-NamMinhNeural"` (Northern Vietnamese male) in
`pixelle_video/tts_voices.py`.

## 3. Use

- **Quick Create**: TTS section → **Local** mode → pick **"quan (clone)"** → Preview
  or generate.
- **Default**: `config.yaml` sets `comfyui.tts.inference_mode: local` and
  `local.voice: clone:quan`, so new videos use quan's voice out of the box.

## Tuning the clone

In `config.yaml` under `comfyui.tts.local`:

- `speed` — narration speed (default `1.0`; `1.2` can sound rushed/less natural).
- `clone_tau` — conversion strength (default `0.25`). **Lower** preserves the Edge
  base's Northern pronunciation/clarity; **higher** pushes harder toward quan's
  timbre but can muddy the accent.

The accent itself comes from the base Edge voice `vi-VN-NamMinhNeural` (Northern).
The speaker embedding is built from the **voiced segments** of the reference clip
(silence dropped), cached as `voices/.cache/<name>.se.v2.pth` — delete it to
re-extract after replacing the clip.

## Notes

- First call loads the model, extracts speaker embeddings, and downloads the
  wavmark model (needs internet once). Embeddings are cached under `voices/.cache/`,
  so later calls are faster (~10 s/line on CPU; faster on GPU).
- Implementation: `pixelle_video/services/voice_conversion.py`
  (`OpenVoiceConverter`) and the cloned-voice branch in
  `pixelle_video/services/tts_service.py::_call_cloned_tts`.
