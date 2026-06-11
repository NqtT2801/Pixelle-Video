# One-off: re-voice an existing task's video with a different TTS voice.
#
# Keeps the (expensive) AI video clips and composed subtitle overlays; only the
# narration audio is re-synthesized, then each segment is rebuilt and the final
# video is re-concatenated + BGM — exactly as the standard pipeline does.
#
# Usage (run from the project root so config.yaml / templates / bgm resolve):
#   .venv/Scripts/python.exe scripts/regen_voice_ductri.py smoke   # frame 0 -> temp only
#   .venv/Scripts/python.exe scripts/regen_voice_ductri.py run     # full re-voice in place

import asyncio
import json
import os
import sys
import time
from pathlib import Path

# Console on Windows defaults to cp1252, which can't encode Vietnamese / emoji.
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
VOICE = "vieneu:Đức Trí"


def load_storyboard() -> dict:
    return json.loads(STORYBOARD.read_text(encoding="utf-8"))


async def smoke(core: PixelleVideoCore):
    sb = load_storyboard()
    fr = sb["frames"][0]
    out = Path("temp") / "_ductri_smoke.mp3"
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"[smoke] synthesizing frame 0 with {VOICE!r}")
    print(f"[smoke] text: {fr['narration']}")
    t0 = time.time()
    path = await core.tts(
        text=fr["narration"],
        voice=VOICE,
        speed=sb["config"].get("tts_speed", 1.0),
        inference_mode="local",
        output_path=str(out),
    )
    dur = core.video._get_audio_duration(path)
    print(f"[smoke] OK -> {path} ({dur:.2f}s) in {time.time()-t0:.1f}s")


async def run(core: PixelleVideoCore):
    sb = load_storyboard()
    frames = sb["frames"]
    cfg = sb["config"]
    video = core.video

    # Template compositing geometry (square video window on a cream canvas).
    tpl = resolve_template_path(cfg["frame_template"])
    region = parse_template_video_region(tpl)
    canvas = parse_template_size(tpl)
    fps = int(cfg.get("video_fps", 30))
    print(f"template={tpl}\n region={region} canvas={canvas} fps={fps}")

    segment_paths = []
    for fr in frames:
        n = fr["index"] + 1
        audio_path = fr["audio_path"]
        clip = fr["video_path"]
        composed = fr["composed_image_path"]
        segment = fr["video_segment_path"]

        t0 = time.time()
        # 1) Re-synthesize narration with the new voice (overwrite in place).
        await core.tts(
            text=fr["narration"],
            voice=VOICE,
            speed=cfg.get("tts_speed", 1.0),
            inference_mode="local",
            output_path=audio_path,
        )
        adur = video._get_audio_duration(audio_path)

        # 2) Composite the AI clip into the template window + subtitle overlay.
        overlay = clip + "_overlay.mp4"
        video.composite_video_in_region(
            video=clip,
            overlay_image=composed,
            output=overlay,
            region=region,
            canvas_size=canvas,
            bg_color=region.get("bg_color", "#000000"),
            fps=fps,
        )
        # 3) Merge the new narration (auto-trims/pads video to audio length).
        video.merge_audio_video(
            video=overlay,
            audio=audio_path,
            output=segment,
            replace_audio=True,
            audio_volume=1.0,
        )
        if os.path.exists(overlay):
            os.unlink(overlay)

        segment_paths.append(segment)
        print(f"[{n:02d}/{len(frames)}] audio={adur:.2f}s "
              f"segment={video._get_video_duration(segment):.2f}s "
              f"({time.time()-t0:.1f}s)")

    # 4) Concatenate + BGM (mirror metadata's bgm settings), overwrite final video.
    final = sb.get("final_video_path") or str(TASK_DIR / f"{sb['title']}.mp4")
    meta = json.loads(METADATA.read_text(encoding="utf-8"))
    bgm_path = meta["input"].get("bgm_path", "default.mp3")
    bgm_volume = meta["input"].get("bgm_volume", 0.13)

    if os.path.exists(final):
        backup = final + ".old.mp4"
        if not os.path.exists(backup):
            os.replace(final, backup)
            print(f"backed up old final -> {backup}")

    print(f"concat {len(segment_paths)} segments -> {final} (bgm={bgm_path} vol={bgm_volume})")
    video.concat_videos(
        videos=segment_paths,
        output=final,
        bgm_path=bgm_path,
        bgm_volume=bgm_volume,
        bgm_mode="loop",
    )
    print(f"final duration: {video._get_video_duration(final):.2f}s")

    # 5) Keep records consistent with the new voice.
    sb["config"]["voice_id"] = VOICE
    STORYBOARD.write_text(json.dumps(sb, ensure_ascii=False, indent=2), encoding="utf-8")
    meta["input"]["tts_voice"] = VOICE
    METADATA.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print("updated storyboard.json + metadata.json voice fields")
    print("DONE")


async def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    core = PixelleVideoCore()
    await core.initialize()
    try:
        if mode == "smoke":
            await smoke(core)
        elif mode == "run":
            await run(core)
        else:
            print(f"unknown mode: {mode!r} (use 'smoke' or 'run')")
            sys.exit(2)
    finally:
        await core.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
