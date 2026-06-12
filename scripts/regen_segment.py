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
Repair a single narration segment whose VieNeu voice drifted, then re-stitch the video.

VieNeu-TTS's sampler is stochastic, so one segment of a video occasionally renders in a
different-sounding voice than the rest. Re-running with the *same* seed+text reproduces
the same drift, so this tool re-rolls the segment with alternative seeds / lower
temperatures (which pull the preset speaker back toward its dominant timbre) and lets you
pick the take that matches by ear — the difference is too fine-grained for an automatic
acoustic metric to judge reliably.

Workflow (two phases):

  1. generate — synthesize several candidate takes of the segment into a `_fix<NN>`
     folder, alongside the ORIGINAL take and two NEIGHBOUR segments for easy A/B:

         .venv/Scripts/python.exe scripts/regen_segment.py generate \
             output/20260611_203548_dc2e 16

  2. apply — once you've chosen a candidate by ear, swap it in, rebuild that frame's
     video segment, and re-concatenate the final video (same BGM) to a `(fixed)` file:

         .venv/Scripts/python.exe scripts/regen_segment.py apply \
             output/20260611_203548_dc2e 16 \
             output/20260611_203548_dc2e/frames/_fix16/cand_t0.20_s0042.mp3

`<NN>` is the segment number exactly as it appears in the filename (e.g. 16 for
`16_audio.mp3`); internally that is storyboard frame index NN-1.

Candidate generation routes through the real `TTSService` local path, so the ffmpeg
transcode and -16 LUFS loudness normalization are byte-for-byte identical to how the
originals were produced — only `vieneu_seed` / `vieneu_temperature` change per take.
"""

import argparse
import asyncio
import json
import shutil
import sys
from pathlib import Path

# Allow running as `python scripts/regen_segment.py` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Vietnamese titles/text crash the default Windows console codepage (cp1252) on print.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

from loguru import logger

from pixelle_video.services.tts_service import TTSService
from pixelle_video.services.video import VideoService
from pixelle_video.utils.os_util import sanitize_filename


# --------------------------------------------------------------------------- #
# Run metadata
# --------------------------------------------------------------------------- #
def _load_run(run_dir: Path, seg_num: int):
    """Read voice/speed/BGM/title/narration for the target segment from a run's outputs."""
    meta = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    sb = json.loads((run_dir / "storyboard.json").read_text(encoding="utf-8"))
    inp = meta.get("input", {})

    frame_index = seg_num - 1
    frame = next((f for f in sb["frames"] if f["index"] == frame_index), None)
    if frame is None:
        raise SystemExit(f"frame index {frame_index} (segment {seg_num}) not found in storyboard.json")

    return {
        "voice": inp.get("tts_voice") or sb.get("config", {}).get("voice_id"),
        "speed": inp.get("tts_speed", 1.0),
        "narration": frame["narration"],
        "title": inp.get("title") or sb.get("title") or "final",
        "bgm_path": inp.get("bgm_path"),
        "bgm_volume": inp.get("bgm_volume", 0.2),
        "bgm_mode": inp.get("bgm_mode", "loop"),
        "fps": inp.get("video_fps", 30),
        "frame_index": frame_index,
    }


def _tts_service(voice: str, speed: float) -> TTSService:
    """A TTSService wired for the local VieNeu path (no ComfyUI/core needed)."""
    config = {
        "comfyui": {
            "tts": {
                "inference_mode": "local",
                "local": {
                    "voice": voice,
                    "speed": speed,
                    "target_lufs": -16,  # same standard level as the original run
                },
            }
        }
    }
    return TTSService(config, core=None)


def _seg(run_dir: Path, seg_num: int, kind: str) -> Path:
    return run_dir / "frames" / f"{seg_num:02d}_{kind}.{'mp3' if kind == 'audio' else 'mp4' if kind == 'segment' else 'png'}"


# --------------------------------------------------------------------------- #
# generate
# --------------------------------------------------------------------------- #
async def _generate(run_dir: Path, seg_num: int, temps, seeds):
    info = _load_run(run_dir, seg_num)
    out_dir = run_dir / "frames" / f"_fix{seg_num:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Segment {seg_num} (frame {info['frame_index']}) — voice={info['voice']} speed={info['speed']}")
    logger.info(f"Text: {info['narration']!r}")

    tts = _tts_service(info["voice"], info["speed"])
    local_cfg = tts.config["local"]  # same dict object TTSService reads per call

    # Reference takes to A/B against (the original + two neighbours that sound right).
    refs = []
    orig = _seg(run_dir, seg_num, "audio")
    if orig.exists():
        dst = out_dir / f"_ORIGINAL_{seg_num:02d}.mp3"
        shutil.copy2(orig, dst); refs.append(dst)
    for nb in (seg_num - 1, seg_num + 1):
        nb_audio = _seg(run_dir, nb, "audio")
        if nb_audio.exists():
            dst = out_dir / f"_NEIGHBOUR_{nb:02d}.mp3"
            shutil.copy2(nb_audio, dst); refs.append(dst)

    candidates = []
    for temp in temps:
        for seed in seeds:
            local_cfg["vieneu_temperature"] = temp
            local_cfg["vieneu_seed"] = seed
            out_path = out_dir / f"cand_t{temp:.2f}_s{seed:04d}.mp3"
            await tts(
                text=info["narration"], voice=info["voice"], speed=info["speed"],
                inference_mode="local", output_path=str(out_path),
            )
            candidates.append((temp, seed, out_path))
            logger.info(f"  ✓ candidate temp={temp} seed={seed} -> {out_path.name}")

    print("\n" + "=" * 72)
    print(f"Generated {len(candidates)} candidate take(s) in:\n  {out_dir}")
    print("Reference files in the same folder for A/B comparison:")
    for r in refs:
        print(f"  {r.name}")
    print("\nCandidates (listen and compare to the NEIGHBOUR files):")
    for temp, seed, p in candidates:
        print(f"  temp={temp:<4} seed={seed:<5} {p.name}")
    print("\nThen apply your pick, e.g.:")
    best = candidates[0][2] if candidates else out_dir / "cand_*.mp3"
    print(f'  .venv/Scripts/python.exe scripts/regen_segment.py apply "{run_dir}" {seg_num} "{best}"')
    print("=" * 72)


# --------------------------------------------------------------------------- #
# apply
# --------------------------------------------------------------------------- #
def _apply(run_dir: Path, seg_num: int, chosen: Path):
    if not chosen.exists():
        raise SystemExit(f"chosen candidate not found: {chosen}")
    info = _load_run(run_dir, seg_num)
    vs = VideoService()

    audio = _seg(run_dir, seg_num, "audio")
    segment = _seg(run_dir, seg_num, "segment")
    composed = _seg(run_dir, seg_num, "composed")

    # Back up originals once (don't clobber a backup from an earlier run).
    for f in (audio, segment):
        bak = f.with_suffix(f".orig{f.suffix}")
        if f.exists() and not bak.exists():
            shutil.copy2(f, bak)
            logger.info(f"backed up {f.name} -> {bak.name}")

    # 1) Swap in the chosen take.
    shutil.copy2(chosen, audio)
    logger.info(f"installed chosen take -> {audio.name}")

    # 2) Rebuild this frame's video segment (static composed image + new narration).
    vs.create_video_from_image(image=str(composed), audio=str(audio), output=str(segment), fps=info["fps"])
    logger.info(f"rebuilt {segment.name}")

    # 3) Re-concatenate all segments (+ same BGM) into a `(fixed)` final video.
    seg_paths = sorted(str(p) for p in (run_dir / "frames").glob("[0-9][0-9]_segment.mp4"))
    out_path = str(run_dir / f"{sanitize_filename(info['title'])} (fixed).mp4")
    bgm = info["bgm_path"]
    try:
        vs.concat_videos(seg_paths, out_path, method="demuxer",
                         bgm_path=bgm, bgm_volume=info["bgm_volume"], bgm_mode=info["bgm_mode"])
    except Exception as e:
        logger.warning(f"demuxer concat failed ({e}); retrying with filter method")
        vs.concat_videos(seg_paths, out_path, method="filter",
                         bgm_path=bgm, bgm_volume=info["bgm_volume"], bgm_mode=info["bgm_mode"])

    print("\n" + "=" * 72)
    print(f"Re-stitched final video:\n  {out_path}")
    print(f"(original kept; segment {seg_num} backups: *.orig.*)")
    print("=" * 72)


# --------------------------------------------------------------------------- #
def _floats(s): return [float(x) for x in s.split(",") if x.strip()]
def _ints(s): return [int(x) for x in s.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser(description="Repair a drifted narration segment and re-stitch the video.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="synthesize candidate re-takes for audition")
    g.add_argument("run_dir", type=Path)
    g.add_argument("seg_num", type=int, help="segment number as in the filename, e.g. 16")
    g.add_argument("--temps", type=_floats, default=[0.1, 0.2, 0.3], help="comma-separated temperatures")
    g.add_argument("--seeds", type=_ints, default=[1234, 42, 777], help="comma-separated seeds")

    a = sub.add_parser("apply", help="swap in a chosen take, rebuild segment, re-stitch")
    a.add_argument("run_dir", type=Path)
    a.add_argument("seg_num", type=int)
    a.add_argument("chosen", type=Path, help="path to the chosen candidate mp3")

    args = ap.parse_args()
    if args.cmd == "generate":
        asyncio.run(_generate(args.run_dir, args.seg_num, args.temps, args.seeds))
    else:
        _apply(args.run_dir, args.seg_num, args.chosen)


if __name__ == "__main__":
    main()
