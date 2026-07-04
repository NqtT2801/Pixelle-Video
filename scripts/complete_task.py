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
Finish a `fixed`-mode run that crashed mid-way, REUSING every scene already on disk.

A run that fails inside ``produce_assets`` never reaches ``finalize``, so no
``storyboard.json``/``metadata.json`` was written — but the per-scene assets that DID
complete are on disk under ``output/<task_id>/frames/`` (``NN_audio.mp3`` / ``NN_video.mp4``
/ ``NN_composed.png`` / ``NN_segment.mp4``). This script reconstructs the storyboard from
the ORIGINAL story text (fixed mode splits it deterministically into the same narrations),
reuses finished scenes verbatim, regenerates only the unfinished ones (reusing any scene's
existing TTS audio), concatenates all scenes into the final mp4, and writes
``storyboard.json``/``metadata.json`` so the task becomes self-consistent (and
``regenerate_frame.py``-able) afterwards.

Per-scene state is detected from disk:
    - segment present            -> DONE: reused as-is (no regeneration, no cost)
    - only audio present         -> reuse audio, regenerate media + compose + segment
    - nothing present            -> full generation (audio + media + compose + segment)

Usage:
    .venv/Scripts/python.exe scripts/complete_task.py <task_id> \
        --text-file story.txt \
        --media-workflow runninghub/video_wan2.2.json \
        [--split-mode paragraph] [--expected-scenes 17] \
        [--voice gcloud:vi-VN-Chirp3-HD-Enceladus] [--no-prefix] [--prompt-prefix "..."] \
        [--media-width 720] [--media-height 720] \
        [--frame-template 1080x1920/video_minimalist_cartoon.html] \
        [--title "..."] [--bgm-path default.mp3] [--bgm-volume 0.2] [--bgm-mode loop]

Notes:
    - ``--media-workflow`` MUST be a *video* workflow (name contains ``video_``) to match the
      finished scenes; an image workflow would render the new scenes as static pictures.
    - The text + ``--split-mode`` must reproduce exactly ``--expected-scenes`` segments, or the
      script aborts (so the finished, position-keyed scenes can't get misaligned).
    - Regenerating media calls the configured media workflow (e.g. RunningHub) and incurs cost
      ONLY for the unfinished scenes.
"""

import argparse
import asyncio
import sys
from datetime import datetime
from pathlib import Path

# Vietnamese titles / emoji logs -> avoid UnicodeEncodeError on the Windows console.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from loguru import logger

from pixelle_video import pixelle_video
from pixelle_video.models.storyboard import Storyboard, StoryboardConfig, StoryboardFrame
from pixelle_video.utils.content_generators import (
    generate_image_prompts,
    generate_title,
    split_narration_script,
)
from pixelle_video.utils.os_util import get_task_frame_path, get_task_final_video_path
from pixelle_video.utils.prompt_helper import build_image_prompt


def _probe_duration(path: str) -> float:
    """Duration (seconds) of a media file via ffmpeg-python (already a dependency)."""
    try:
        import ffmpeg

        return float(ffmpeg.probe(path)["format"]["duration"])
    except Exception as e:  # pragma: no cover - defensive
        logger.warning(f"Could not probe duration for {path}: {e}")
        return 0.0


async def complete(args) -> int:
    task_id = args.task_id
    expected = args.expected_scenes

    # --- read original text ---
    if args.text_file:
        text = Path(args.text_file).read_text(encoding="utf-8")
    else:
        text = args.text

    await pixelle_video.initialize()
    try:
        # 1. Reproduce the narrations (fixed mode = deterministic split) + hard count guard.
        narrations = await split_narration_script(text, split_mode=args.split_mode)
        logger.info(f"Split original text into {len(narrations)} narrations (mode={args.split_mode})")
        if len(narrations) != expected:
            logger.error(
                f"Text split into {len(narrations)} segments but expected {expected}. "
                f"Aborting — proceeding would misalign the finished, position-keyed scenes. "
                f"Make sure the text + --split-mode match the original run exactly "
                f"(blank-line / whitespace structure matters)."
            )
            return 1

        # 2. Detect per-scene state from disk.
        frames_dir = Path(get_task_frame_path(task_id, 0, "audio")).parent
        if not frames_dir.exists():
            logger.error(f"Task frames dir not found: {frames_dir}")
            return 1

        def existing(i: int, kind: str) -> str | None:
            p = get_task_frame_path(task_id, i, kind)
            return str(p) if Path(p).exists() else None

        done_idx, audio_only_idx, missing_idx = [], [], []
        for i in range(expected):
            if existing(i, "segment"):
                done_idx.append(i)
            elif existing(i, "audio"):
                audio_only_idx.append(i)
            else:
                missing_idx.append(i)
        todo_idx = sorted(audio_only_idx + missing_idx)
        logger.info(
            f"Scene state: {len(done_idx)} done, "
            f"{len(audio_only_idx)} audio-only {[i + 1 for i in audio_only_idx]}, "
            f"{len(missing_idx)} missing {[i + 1 for i in missing_idx]}"
        )
        if not todo_idx:
            logger.warning("Nothing to do — every scene already has a segment. Will just re-concat.")
        for i in todo_idx:
            logger.info(f"  scene {i + 1} (idx {i}) narration: {narrations[i][:70]!r}")

        # 3. Generate image prompts for the scenes that need media (audio-only + missing).
        prompt_by_idx: dict[int, str] = {}
        if todo_idx:
            base = await generate_image_prompts(pixelle_video.llm, [narrations[i] for i in todo_idx])
            if args.no_prefix:
                prefix = ""
            elif args.prompt_prefix is not None:
                prefix = args.prompt_prefix
            else:
                prefix = pixelle_video.config.get("comfyui", {}).get("image", {}).get("prompt_prefix", "")
            for i, base_prompt in zip(todo_idx, base):
                prompt_by_idx[i] = build_image_prompt(base_prompt, prefix) if prefix else base_prompt

        # 4. Title (only affects the output filename).
        title = args.title or await generate_title(pixelle_video.llm, text, strategy="llm")
        logger.info(f"Title: {title!r}")

        # 5. Reconstruct config + storyboard.
        config = StoryboardConfig(
            media_width=args.media_width,
            media_height=args.media_height,
            task_id=task_id,
            n_storyboard=expected,
            video_fps=30,
            tts_inference_mode="local",
            voice_id=args.voice,
            tts_speed=args.tts_speed,
            media_workflow=args.media_workflow,
            frame_template=args.frame_template,
            subtitle_sync=False,
        )
        sb = Storyboard(title=title, config=config, created_at=datetime.now())

        for i in range(expected):
            if i in done_idx:
                # Reuse verbatim. image_prompt=None so it can never be regenerated.
                seg = existing(i, "segment")
                sb.frames.append(
                    StoryboardFrame(
                        index=i,
                        narration=narrations[i],
                        image_prompt=None,
                        media_type="video",
                        audio_path=existing(i, "audio"),
                        video_path=existing(i, "video"),
                        composed_image_path=existing(i, "composed"),
                        video_segment_path=seg,
                        duration=_probe_duration(seg),
                    )
                )
            elif i in audio_only_idx:
                # Reuse existing audio; regenerate media + compose + segment. Set duration
                # from the audio NOW because the processor skips the audio step (which is
                # what normally sets frame.duration) — video gen needs the target length.
                audio = existing(i, "audio")
                sb.frames.append(
                    StoryboardFrame(
                        index=i,
                        narration=narrations[i],
                        image_prompt=prompt_by_idx[i],
                        audio_path=audio,
                        duration=_probe_duration(audio),
                    )
                )
            else:  # missing -> full generation
                sb.frames.append(
                    StoryboardFrame(index=i, narration=narrations[i], image_prompt=prompt_by_idx[i])
                )

        # 6. Process only the unfinished scenes (in index order). Audio is reused where set;
        #    the media-retry in MediaService handles transient RunningHub failures.
        for i in todo_idx:
            logger.info(f"🎬 Generating scene {i + 1} (idx {i}) ...")
            await pixelle_video.frame_processor(
                frame=sb.frames[i], storyboard=sb, config=config, total_frames=expected
            )
            if not sb.frames[i].video_segment_path:
                logger.error(f"Scene {i + 1} produced no segment; aborting before concat.")
                return 1
            logger.success(f"✅ Scene {i + 1} done: {sb.frames[i].video_segment_path}")

        # 7. Concatenate all scenes (in order) + BGM.
        ordered = sorted(sb.frames, key=lambda f: f.index)
        missing_seg = [f.index + 1 for f in ordered if not f.video_segment_path]
        if missing_seg:
            logger.error(f"Cannot concat — missing segments for scenes: {missing_seg}")
            return 1
        segment_paths = [f.video_segment_path for f in ordered]
        final_path = get_task_final_video_path(task_id, title)
        logger.info(
            f"🔗 Concatenating {len(segment_paths)} segments → {final_path} "
            f"(bgm={args.bgm_path}, volume={args.bgm_volume}, mode={args.bgm_mode})"
        )
        pixelle_video.video.concat_videos(
            videos=segment_paths,
            output=final_path,
            bgm_path=args.bgm_path,
            bgm_volume=args.bgm_volume,
            bgm_mode=args.bgm_mode,
        )

        # 8. Persist storyboard.json + metadata.json so the task is whole + resumable.
        sb.final_video_path = final_path
        sb.total_duration = sum(f.duration for f in ordered)
        sb.completed_at = datetime.now()
        await pixelle_video.persistence.save_storyboard(task_id, sb)

        file_size = Path(final_path).stat().st_size if Path(final_path).exists() else 0
        metadata = {
            "task_id": task_id,
            "created_at": sb.created_at.isoformat() if sb.created_at else None,
            "completed_at": sb.completed_at.isoformat() if sb.completed_at else None,
            "status": "completed",
            "input": {
                "text": text,
                "mode": "fixed",
                "split_mode": args.split_mode,
                "title": title,
                "media_workflow": args.media_workflow,
                "frame_template": args.frame_template,
                "tts_voice": args.voice,
                "media_width": args.media_width,
                "media_height": args.media_height,
                "bgm_path": args.bgm_path,
                "bgm_volume": args.bgm_volume,
                "bgm_mode": args.bgm_mode,
            },
            "result": {
                "video_path": final_path,
                "duration": sb.total_duration,
                "file_size": file_size,
                "n_frames": len(ordered),
            },
            "config": {"recovered_by": "scripts/complete_task.py"},
        }
        await pixelle_video.persistence.save_task_metadata(task_id, metadata)

        logger.success(f"🎉 Done. Final video rebuilt: {final_path}")
        logger.info(f"   Duration: {sb.total_duration:.2f}s, scenes: {len(ordered)}")
        return 0
    finally:
        await pixelle_video.cleanup()


def main() -> int:
    p = argparse.ArgumentParser(
        description="Finish a crashed fixed-mode run, reusing scenes already on disk."
    )
    p.add_argument("task_id", help="Task ID (folder name under output/)")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--text-file", default=None, help="Path to a UTF-8 file with the ORIGINAL story text")
    src.add_argument("--text", default=None, help="The ORIGINAL story text (inline)")
    p.add_argument("--media-workflow", required=True, help="VIDEO media workflow key, e.g. runninghub/video_wan2.2.json")
    p.add_argument("--split-mode", default="paragraph", choices=["paragraph", "line", "sentence"])
    p.add_argument("--expected-scenes", type=int, default=17, help="Number of scenes the text must split into")
    p.add_argument("--voice", default="gcloud:vi-VN-Chirp3-HD-Enceladus", help="Local TTS voice id")
    p.add_argument("--tts-speed", type=float, default=1.0)
    p.add_argument("--prompt-prefix", default=None, help="Override style prefix (default: config comfyui.image.prompt_prefix)")
    p.add_argument("--no-prefix", action="store_true", help="Use generated prompts verbatim (no style prefix)")
    p.add_argument("--media-width", type=int, default=720)
    p.add_argument("--media-height", type=int, default=720)
    p.add_argument("--frame-template", default="1080x1920/video_minimalist_cartoon.html")
    p.add_argument("--title", default=None, help="Output filename title (default: LLM-generated from text)")
    p.add_argument("--bgm-path", default="default.mp3", help="BGM filename/path (bare name resolved from bgm/); '' to disable")
    p.add_argument("--bgm-volume", type=float, default=0.2)
    p.add_argument("--bgm-mode", default="loop", choices=["once", "loop"])
    args = p.parse_args()

    if args.bgm_path == "":
        args.bgm_path = None
    if "video_" not in args.media_workflow.lower():
        logger.warning(
            f"--media-workflow {args.media_workflow!r} has no 'video_' in its name; "
            f"the new scenes may render as static images instead of video."
        )

    return asyncio.run(complete(args))


if __name__ == "__main__":
    raise SystemExit(main())
