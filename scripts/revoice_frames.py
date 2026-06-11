# Re-voice specific frames whose VieNeu narration drifted to a different-sounding
# speaker, then rebuild their segments and re-concat the final video.
#
# VieNeu's sampler is deterministic for a fixed (text, seed), but a given seed can
# land on a drifted timbre for *some* texts. Every frame in a task is synthesized
# with the same seed (default 1234); a couple of frames occasionally come out sounding
# like a different person. Fix = re-synthesize just those frames with a DIFFERENT seed
# that matches the rest, then rebuild.
#
# Because the match is a perceptual (by-ear) judgement, this is a two-step tool:
#
#   # 1) Generate candidates for the bad frames (does NOT touch originals). For each
#   #    seed it writes a normalized clip plus an in-context preview
#   #    (prev_frame + candidate + next_frame) so you can hear whether it blends.
#   .venv/Scripts/python.exe scripts/revoice_frames.py sweep 10 16
#
#   # 2) Apply the chosen seed per frame (overwrites audio in place, backs up the
#   #    final video, rebuilds the two segments, re-concatenates + BGM).
#   .venv/Scripts/python.exe scripts/revoice_frames.py apply 10:42 16:777
#
# Frame numbers are 1-based (file names): "10" == frames/10_audio.mp3 == index 9.

import asyncio
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import librosa
import numpy as np

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

from pixelle_video.service import PixelleVideoCore
from pixelle_video.services.vieneu_service import VieNeuEngine
from pixelle_video.tts_voices import resolve_vieneu_voice
from pixelle_video.utils.template_util import (
    parse_template_video_region,
    parse_template_size,
    resolve_template_path,
)

TASK_DIR = Path(r"D:\Projects\Pixelle-Video\output\20260610_120202_6573")
STORYBOARD = TASK_DIR / "storyboard.json"
METADATA = TASK_DIR / "metadata.json"
CAND_DIR = Path(r"D:\Projects\Pixelle-Video\temp\voice_cand")
TARGET_LUFS = -16.0
DEFAULT_SEEDS = [1234, 0, 1, 7, 42, 99, 777, 2024, 8888]


# --------------------------------------------------------------------------- #
# audio helpers
# --------------------------------------------------------------------------- #
def measure_lufs(path: str):
    r = subprocess.run(
        ["ffmpeg", "-i", path, "-af", "loudnorm=I=-16:print_format=json", "-f", "null", "-"],
        capture_output=True, text=True,
    )
    m = re.search(r'"input_i"\s*:\s*"(-?[0-9.]+)"', r.stderr)
    return float(m.group(1)) if m else None


def normalize_to_lufs(path: str, target: float = TARGET_LUFS):
    """Two-pass loudnorm to an exact integrated loudness, replacing in place (matches
    the pipeline's TTS normalization so re-voiced frames sit at the same level)."""
    s1, s2 = f"{path}.s1.mp3", f"{path}.s2.mp3"
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", path, "-af", f"loudnorm=I={target}:TP=-1.5:LRA=11",
             "-ar", "44100", "-loglevel", "error", s1],
            check=True, capture_output=True, text=True,
        )
        measured = measure_lufs(s1)
        gain = 0.0 if measured is None else (target - measured)
        subprocess.run(
            ["ffmpeg", "-y", "-i", s1, "-af", f"volume={gain:.2f}dB,alimiter=limit=0.95",
             "-ar", "44100", "-loglevel", "error", s2],
            check=True, capture_output=True, text=True,
        )
        for _ in range(10):
            try:
                os.replace(s2, path)
                break
            except PermissionError:
                time.sleep(0.5)
    finally:
        for tmp in (s1, s2):
            if os.path.exists(tmp):
                try:
                    os.unlink(tmp)
                except OSError:
                    pass


def synth_frame(eng, text, voice_id, seed, temperature, speed, out_mp3):
    """VieNeu synth -> mp3 (atempo for speed) -> normalize to -16 LUFS, exactly like
    the TTS service does for a pipeline frame."""
    wav = out_mp3 + ".vieneu.wav"
    try:
        eng.synthesize(text, voice_id, wav, temperature, seed)
        stream = __import__("ffmpeg").input(wav)
        if speed and abs(speed - 1.0) > 0.01:
            tempo = max(0.5, min(2.0, speed))
            stream = stream.output(out_mp3, **{"filter:a": f"atempo={tempo}"}, loglevel="error")
        else:
            stream = stream.output(out_mp3, loglevel="error")
        stream.overwrite_output().run()
    finally:
        if os.path.exists(wav):
            try:
                os.unlink(wav)
            except OSError:
                pass
    normalize_to_lufs(out_mp3)
    return out_mp3


def med_f0(path: str):
    y, sr = librosa.load(str(path), sr=16000, mono=True)
    y, _ = librosa.effects.trim(y, top_db=30)
    f0, _, _ = librosa.pyin(y, sr=sr, fmin=70, fmax=400, frame_length=1024)
    f0v = f0[~np.isnan(f0)]
    return float(np.median(f0v)) if f0v.size else float("nan")


def timbre(path: str):
    y, sr = librosa.load(str(path), sr=16000, mono=True)
    y, _ = librosa.effects.trim(y, top_db=30)
    mf = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)[1:]
    return np.concatenate([mf.mean(1), mf.std(1)])


def concat_audio(parts, out_mp3):
    """Concatenate mp3 parts into one clip (for in-context previews)."""
    args = ["ffmpeg", "-y"]
    for p in parts:
        args += ["-i", p]
    n = len(parts)
    fc = "".join(f"[{i}:a]" for i in range(n)) + f"concat=n={n}:v=0:a=1[a]"
    args += ["-filter_complex", fc, "-map", "[a]", "-loglevel", "error", out_mp3]
    subprocess.run(args, check=True)


# --------------------------------------------------------------------------- #
# context
# --------------------------------------------------------------------------- #
def load_sb():
    return json.loads(STORYBOARD.read_text(encoding="utf-8"))


def voice_params(sb):
    cfg = sb["config"]
    voice = resolve_vieneu_voice(cfg["voice_id"])
    if not voice:
        raise SystemExit(f"voice_id {cfg['voice_id']!r} is not a VieNeu preset")
    return voice["voice_id"], 0.4, float(cfg.get("tts_speed", 1.0))


# --------------------------------------------------------------------------- #
# sweep
# --------------------------------------------------------------------------- #
def cmd_sweep(nums, seeds):
    sb = load_sb()
    voice_id, temperature, speed = voice_params(sb)
    frames = sb["frames"]
    CAND_DIR.mkdir(parents=True, exist_ok=True)
    eng = VieNeuEngine()

    # global pitch center of the frames the user is keeping (exclude the bad ones)
    good = [med_f0(f["audio_path"]) for f in frames if (f["index"] + 1) not in nums]
    center = float(np.median([x for x in good if not np.isnan(x)]))
    print(f"target pitch (median of kept frames) ~= {center:.0f} Hz\n")

    for n in nums:
        fr = frames[n - 1]
        text = fr["narration"]
        ev = timbre(fr["audio_path"])
        ef0 = med_f0(fr["audio_path"])
        prev_a = frames[n - 2]["audio_path"] if n - 2 >= 0 else None
        next_a = frames[n]["audio_path"] if n < len(frames) else None
        print(f"=== frame {n} (existing medF0={ef0:.0f} Hz) ===")
        print(f"    {text}")
        print(f"    {'seed':>6} {'medF0':>6} {'dF0':>6} {'distExisting':>12}  preview")
        rows = []
        for seed in seeds:
            mp3 = str(CAND_DIR / f"{n:02d}_seed{seed}.mp3")
            synth_frame(eng, text, voice_id, seed, temperature, speed, mp3)
            f0 = med_f0(mp3)
            dist = float(np.linalg.norm(timbre(mp3) - ev))
            ctx = str(CAND_DIR / f"{n:02d}_seed{seed}_ctx.mp3")
            parts = [p for p in (prev_a, mp3, next_a) if p]
            concat_audio(parts, ctx)
            rows.append((seed, f0, dist))
            tag = "  <- current (drifted)" if dist < 2.0 else ""
            print(f"    {seed:>6} {f0:>6.0f} {f0-center:>+6.0f} {dist:>12.2f}  "
                  f"{Path(ctx).name}{tag}")
        # suggest: most different from the drifted original, pitch closest to center
        cand = [r for r in rows if r[2] > 5.0]
        cand.sort(key=lambda r: abs(r[1] - center))
        if cand:
            print(f"    suggestion by pitch-match: seed {cand[0][0]} "
                  f"(medF0={cand[0][1]:.0f}); but verify the *_ctx.mp3 previews by ear")
        print(f"    previews + clips in: {CAND_DIR}\n")


# --------------------------------------------------------------------------- #
# apply
# --------------------------------------------------------------------------- #
async def cmd_apply(choices):
    core = PixelleVideoCore()
    await core.initialize()
    try:
        sb = load_sb()
        voice_id, temperature, speed = voice_params(sb)
        cfg = sb["config"]
        video = core.video
        eng = VieNeuEngine()

        tpl = resolve_template_path(cfg["frame_template"])
        region = parse_template_video_region(tpl)
        canvas = parse_template_size(tpl)
        fps = int(cfg.get("video_fps", 30))

        for n, seed in choices:
            fr = sb["frames"][n - 1]
            audio = fr["audio_path"]
            if os.path.exists(audio):
                bak = audio + ".old.mp3"
                if not os.path.exists(bak):
                    os.replace(audio, bak)
                    print(f"[{n:02d}] backed up old audio -> {Path(bak).name}")
            synth_frame(eng, fr["narration"], voice_id, seed, temperature, speed, audio)
            print(f"[{n:02d}] re-voiced with seed {seed}: medF0={med_f0(audio):.0f} Hz, "
                  f"{measure_lufs(audio):.2f} LUFS")

            overlay = fr["video_path"] + "_overlay.mp4"
            video.composite_video_in_region(
                video=fr["video_path"], overlay_image=fr["composed_image_path"],
                output=overlay, region=region, canvas_size=canvas,
                bg_color=region.get("bg_color", "#000000"), fps=fps,
            )
            video.merge_audio_video(
                video=overlay, audio=audio, output=fr["video_segment_path"],
                replace_audio=True, audio_volume=1.0,
            )
            if os.path.exists(overlay):
                os.unlink(overlay)
            print(f"[{n:02d}] segment rebuilt: {Path(fr['video_segment_path']).name}")

        # re-concat final (+ BGM from metadata), backing up the existing final once
        final = sb.get("final_video_path") or str(TASK_DIR / f"{sb['title']}.mp4")
        meta = json.loads(METADATA.read_text(encoding="utf-8"))
        bgm_path = meta["input"].get("bgm_path", "default.mp3")
        bgm_volume = meta["input"].get("bgm_volume", 0.13)
        if os.path.exists(final):
            bak = final + ".prevoice.mp4"
            if not os.path.exists(bak):
                import shutil
                shutil.copy2(final, bak)
                print(f"backed up final -> {Path(bak).name}")
        segments = [f["video_segment_path"] for f in sb["frames"]]
        print(f"concat {len(segments)} segments -> {final} (bgm={bgm_path} vol={bgm_volume})")
        video.concat_videos(
            videos=segments, output=final, bgm_path=bgm_path,
            bgm_volume=bgm_volume, bgm_mode="loop",
        )
        print(f"final duration: {video._get_video_duration(final):.2f}s")
        print("DONE")
    finally:
        await core.cleanup()


# --------------------------------------------------------------------------- #
def main():
    if len(sys.argv) < 3:
        print(__doc__ or "usage: revoice_frames.py sweep N [N..] | apply N:SEED [..]")
        print("usage: revoice_frames.py sweep N [N ...] [--seeds 1,2,3]")
        print("       revoice_frames.py apply N:SEED [N:SEED ...]")
        sys.exit(2)

    mode = sys.argv[1]
    rest = sys.argv[2:]
    if mode == "sweep":
        seeds = DEFAULT_SEEDS
        if "--seeds" in rest:
            i = rest.index("--seeds")
            seeds = [int(x) for x in rest[i + 1].split(",")]
            rest = rest[:i]
        nums = [int(x) for x in rest]
        cmd_sweep(nums, seeds)
    elif mode == "apply":
        choices = []
        for tok in rest:
            n, s = tok.split(":")
            choices.append((int(n), int(s)))
        asyncio.run(cmd_apply(choices))
    else:
        print(f"unknown mode {mode!r} (use 'sweep' or 'apply')")
        sys.exit(2)


if __name__ == "__main__":
    main()
