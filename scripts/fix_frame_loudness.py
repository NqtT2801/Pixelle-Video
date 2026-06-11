# Targeted fix: re-normalize specific frames' narration to -16 LUFS, rebuild their
# segments, and re-concatenate the final video. Used to repair frames whose loudness
# normalization was skipped during a run (transient Windows file lock / WinError 5).
#
#   .venv/Scripts/python.exe scripts/fix_frame_loudness.py 13        # fix frame 13
#   .venv/Scripts/python.exe scripts/fix_frame_loudness.py 13 7      # several frames

import asyncio
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

from pixelle_video.service import PixelleVideoCore
from pixelle_video.utils.template_util import (
    parse_template_video_region,
    parse_template_size,
    resolve_template_path,
)

TASK_DIR = Path(r"D:\Projects\Pixelle-Video\output\20260610_120202_6573")
STORYBOARD = TASK_DIR / "storyboard.json"
METADATA = TASK_DIR / "metadata.json"
TARGET_LUFS = -16.0


def measure_lufs(path: str):
    r = subprocess.run(
        ["ffmpeg", "-i", path, "-af", "loudnorm=I=-16:print_format=json", "-f", "null", "-"],
        capture_output=True, text=True,
    )
    m = re.search(r'"input_i"\s*:\s*"(-?[0-9.]+)"', r.stderr)
    return float(m.group(1)) if m else None


def normalize_to_lufs(path: str, target: float = TARGET_LUFS):
    """Two-pass loudnorm to an exact integrated loudness, replacing the file in place
    with a retry loop (the in-pipeline replace can hit a transient WinError 5 lock)."""
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
        # Robust in-place replace: retry through transient Access-Denied locks.
        last = None
        for attempt in range(10):
            try:
                os.replace(s2, path)
                last = None
                break
            except PermissionError as e:
                last = e
                time.sleep(0.5)
        if last is not None:
            raise last
    finally:
        for tmp in (s1, s2):
            if os.path.exists(tmp):
                try:
                    os.unlink(tmp)
                except OSError:
                    pass


def concat_final(sb: dict, video):
    final = sb.get("final_video_path") or str(TASK_DIR / f"{sb['title']}.mp4")
    meta = json.loads(METADATA.read_text(encoding="utf-8"))
    bgm_path = meta["input"].get("bgm_path", "default.mp3")
    bgm_volume = meta["input"].get("bgm_volume", 0.13)
    segments = [f["video_segment_path"] for f in sb["frames"]]
    print(f"concat {len(segments)} segments -> {final} (bgm={bgm_path} vol={bgm_volume})")
    video.concat_videos(
        videos=segments, output=final,
        bgm_path=bgm_path, bgm_volume=bgm_volume, bgm_mode="loop",
    )
    print(f"final duration: {video._get_video_duration(final):.2f}s")


async def main():
    nums = [int(a) for a in sys.argv[1:]]
    if not nums:
        print("usage: fix_frame_loudness.py <frame_num> [frame_num ...]")
        sys.exit(2)

    core = PixelleVideoCore()
    await core.initialize()
    try:
        sb = json.loads(STORYBOARD.read_text(encoding="utf-8"))
        cfg = sb["config"]
        video = core.video
        by_num = {f["index"] + 1: f for f in sb["frames"]}

        tpl = resolve_template_path(cfg["frame_template"])
        region = parse_template_video_region(tpl)
        canvas = parse_template_size(tpl)
        fps = int(cfg.get("video_fps", 30))

        for n in nums:
            fr = by_num[n]
            audio = fr["audio_path"]
            before = measure_lufs(audio)
            normalize_to_lufs(audio)
            after = measure_lufs(audio)
            print(f"[{n:02d}] loudness {before:.2f} -> {after:.2f} LUFS")

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
            print(f"[{n:02d}] segment rebuilt: {fr['video_segment_path']}")

        concat_final(sb, video)
        print("DONE")
    finally:
        await core.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
