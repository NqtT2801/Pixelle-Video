# Re-voice ALL frames of a completed task to a single chosen VieNeu voice, then rebuild
# every segment and re-concat the final video (+ BGM). Used to convert an existing video's
# narration from one preset speaker to another (e.g. "Đức Trí" -> "Ngọc Linh").
#
# Unlike scripts/revoice_frames.py (which fixes a few drifted frames and reuses the voice
# from the storyboard), this overrides the voice for the *whole* video and processes every
# frame with one seed.
#
#   .venv/Scripts/python.exe scripts/revoice_all.py \
#       --task-dir output/20260611_214715_6b52 \
#       --voice "vieneu:Ngọc Linh" --seed 1234
#
# Originals are backed up: frames/XX_audio.mp3 -> *.old.mp3, final -> *.prevoice.mp4.

import argparse
import asyncio
import json
import os
import shutil
import sys
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Reuse the audio helpers that already match the pipeline's TTS normalization.
from revoice_frames import synth_frame, measure_lufs, med_f0  # noqa: E402

from pixelle_video.service import PixelleVideoCore  # noqa: E402
from pixelle_video.services.vieneu_service import VieNeuEngine  # noqa: E402
from pixelle_video.tts_voices import resolve_vieneu_voice  # noqa: E402
from pixelle_video.utils.template_util import (  # noqa: E402
    parse_template_video_region,
    parse_template_size,
    resolve_template_path,
)


async def revoice_all(task_dir: Path, voice_arg: str, seed: int):
    storyboard_path = task_dir / "storyboard.json"
    metadata_path = task_dir / "metadata.json"
    sb = json.loads(storyboard_path.read_text(encoding="utf-8"))
    cfg = sb["config"]

    voice = resolve_vieneu_voice(voice_arg)
    if not voice:
        raise SystemExit(f"voice {voice_arg!r} is not a VieNeu preset")
    voice_id = voice["voice_id"]
    temperature = 0.4
    speed = float(cfg.get("tts_speed", 1.0))
    print(f"target voice: {voice['name']} (voice_id={voice_id!r}), seed={seed}, "
          f"speed={speed}, frames={len(sb['frames'])}\n")

    core = PixelleVideoCore()
    await core.initialize()
    try:
        video = core.video
        eng = VieNeuEngine()

        tpl = resolve_template_path(cfg["frame_template"])
        region = parse_template_video_region(tpl)
        canvas = parse_template_size(tpl)
        fps = int(cfg.get("video_fps", 30))

        for fr in sb["frames"]:
            n = fr["index"] + 1
            audio = fr["audio_path"]
            if os.path.exists(audio):
                bak = audio + ".old.mp3"
                if not os.path.exists(bak):
                    shutil.copy2(audio, bak)

            synth_frame(eng, fr["narration"], voice_id, seed, temperature, speed, audio)
            print(f"[{n:02d}] re-voiced: medF0={med_f0(audio):.0f} Hz, "
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
        final = sb.get("final_video_path") or str(task_dir / f"{sb['title']}.mp4")
        meta = json.loads(metadata_path.read_text(encoding="utf-8"))
        bgm_path = meta["input"].get("bgm_path", "default.mp3")
        bgm_volume = meta["input"].get("bgm_volume", 0.13)
        if os.path.exists(final):
            bak = final + ".prevoice.mp4"
            if not os.path.exists(bak):
                shutil.copy2(final, bak)
                print(f"backed up final -> {Path(bak).name}")
        segments = [f["video_segment_path"] for f in sb["frames"]]
        print(f"concat {len(segments)} segments -> {final} "
              f"(bgm={bgm_path} vol={bgm_volume})")
        video.concat_videos(
            videos=segments, output=final, bgm_path=bgm_path,
            bgm_volume=bgm_volume, bgm_mode="loop",
        )
        print(f"final duration: {video._get_video_duration(final):.2f}s")
        print("DONE")
    finally:
        await core.cleanup()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-dir", default=r"output/20260611_214715_6b52")
    ap.add_argument("--voice", default="vieneu:Ngọc Linh")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()
    asyncio.run(revoice_all(Path(args.task_dir), args.voice, args.seed))


if __name__ == "__main__":
    main()
