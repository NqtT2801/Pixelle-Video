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
Regenerate a single frame of an already-completed task, then rebuild the final video.

This implements the single-frame "redo" that ``HistoryManager.regenerate_frame`` left
as a stub: it loads the saved storyboard, optionally rewrites one frame's image prompt,
re-runs only that frame through the normal ``FrameProcessor`` (reusing the existing TTS
audio so no narration is re-synthesized), and then re-concatenates all segments into the
final mp4 — reusing the BGM settings recorded in the task's ``metadata.json``.

All outputs are overwritten in place (frame video/composed/segment + final mp4 +
storyboard.json), so the task directory stays self-consistent.

Usage:
    .venv/Scripts/python.exe scripts/regenerate_frame.py <task_id> <frame_index> \
        [--prompt "<new base image prompt>"] [--no-prefix] \
        [--bgm-path default.mp3] [--bgm-volume 0.13] [--bgm-mode loop]

Notes:
    - ``frame_index`` is 0-based (matches storyboard.json). File names are 1-based,
      so index 12 corresponds to ``frames/13_*``.
    - ``--prompt`` is a BASE prompt; the configured style prefix
      (``comfyui.image.prompt_prefix``) is prepended automatically, exactly like the
      pipeline does. Pass ``--no-prefix`` if your prompt is already fully formed.
    - Regenerating media calls the configured media workflow (e.g. RunningHub) and may
      incur API cost.
"""

import argparse
import asyncio
import sys

# Vietnamese titles / emoji logs -> avoid UnicodeEncodeError on the Windows console.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from loguru import logger

from pixelle_video import pixelle_video
from pixelle_video.utils.os_util import get_task_final_video_path
from pixelle_video.utils.prompt_helper import build_image_prompt


async def regenerate(
    task_id: str,
    frame_index: int,
    new_prompt: str | None,
    apply_prefix: bool,
    bgm_path: str | None,
    bgm_volume: float | None,
    bgm_mode: str | None,
) -> int:
    await pixelle_video.initialize()
    try:
        # 1. Load the saved storyboard for the task.
        sb = await pixelle_video.persistence.load_storyboard(task_id)
        if sb is None:
            logger.error(f"No storyboard found for task '{task_id}'")
            return 1

        frames = sorted(sb.frames, key=lambda f: f.index)
        if not (0 <= frame_index < len(frames)):
            logger.error(
                f"frame_index {frame_index} out of range (task has {len(frames)} frames, 0..{len(frames) - 1})"
            )
            return 1
        frame = frames[frame_index]

        # 2. Optionally rewrite the image prompt (apply the configured style prefix the
        #    same way StandardPipeline.plan_visuals does).
        if new_prompt is not None:
            if apply_prefix:
                prefix = (
                    pixelle_video.config.get("comfyui", {})
                    .get("image", {})
                    .get("prompt_prefix", "")
                )
                final_prompt = build_image_prompt(new_prompt, prefix)
            else:
                final_prompt = new_prompt
            logger.info(f"✏️  Frame {frame_index} new image_prompt:\n{final_prompt}")
            frame.image_prompt = final_prompt
        else:
            logger.info(f"♻️  Re-rolling frame {frame_index} with its existing prompt")

        if not frame.image_prompt:
            logger.error("Frame has no image_prompt; nothing to generate.")
            return 1

        # 3. Reset downstream media so it regenerates cleanly, but KEEP the TTS audio
        #    (FrameProcessor skips audio generation when audio_path is already set).
        frame.media_type = None
        frame.image_path = None
        frame.video_path = None
        frame.composed_image_path = None
        frame.composed_image_paths = None
        frame.video_segment_path = None

        # 4. Re-run only this frame through the normal processor.
        logger.info(f"🎬 Regenerating frame {frame_index} (file {frame_index + 1:02d}_*) ...")
        await pixelle_video.frame_processor(
            frame=frame,
            storyboard=sb,
            config=sb.config,
            total_frames=len(frames),
        )
        logger.success(f"✅ Frame {frame_index} regenerated: {frame.video_segment_path}")

        # 5. Re-concatenate all segments into the final mp4 (reuse original BGM settings
        #    from metadata.json unless overridden on the CLI).
        metadata = await pixelle_video.persistence.load_task_metadata(task_id)
        meta_input = (metadata or {}).get("input", {})
        final_bgm_path = bgm_path if bgm_path is not None else meta_input.get("bgm_path")
        final_bgm_volume = (
            bgm_volume if bgm_volume is not None else meta_input.get("bgm_volume", 0.2)
        )
        final_bgm_mode = bgm_mode if bgm_mode is not None else meta_input.get("bgm_mode", "loop")

        segment_paths = [f.video_segment_path for f in frames]
        missing = [f.index for f in frames if not f.video_segment_path]
        if missing:
            logger.error(f"Cannot concat — missing segments for frames: {missing}")
            return 1

        final_path = get_task_final_video_path(task_id, sb.title)
        logger.info(
            f"🔗 Concatenating {len(segment_paths)} segments → {final_path} "
            f"(bgm={final_bgm_path}, volume={final_bgm_volume}, mode={final_bgm_mode})"
        )
        pixelle_video.video.concat_videos(
            videos=segment_paths,
            output=final_path,
            bgm_path=final_bgm_path,
            bgm_volume=final_bgm_volume,
            bgm_mode=final_bgm_mode,
        )

        # 6. Persist the updated storyboard (records the new prompt + refreshed duration).
        sb.final_video_path = final_path
        sb.total_duration = sum(f.duration for f in frames)
        await pixelle_video.persistence.save_storyboard(task_id, sb)

        logger.success(f"🎉 Done. Final video rebuilt: {final_path}")
        return 0
    finally:
        await pixelle_video.cleanup()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Regenerate a single frame of a completed task and rebuild the final video."
    )
    parser.add_argument("task_id", help="Task ID (folder name under output/)")
    parser.add_argument("frame_index", type=int, help="Frame index, 0-based (index 12 == file 13_*)")
    parser.add_argument(
        "--prompt",
        default=None,
        help="New BASE image prompt (style prefix is prepended automatically). "
        "Omit to re-roll with the existing prompt.",
    )
    parser.add_argument(
        "--no-prefix",
        action="store_true",
        help="Use --prompt verbatim instead of prepending the configured style prefix.",
    )
    parser.add_argument("--bgm-path", default=None, help="Override BGM (default: from metadata.json)")
    parser.add_argument("--bgm-volume", type=float, default=None, help="Override BGM volume")
    parser.add_argument(
        "--bgm-mode", default=None, choices=["once", "loop"], help="Override BGM mode"
    )
    args = parser.parse_args()

    return asyncio.run(
        regenerate(
            task_id=args.task_id,
            frame_index=args.frame_index,
            new_prompt=args.prompt,
            apply_prefix=not args.no_prefix,
            bgm_path=args.bgm_path,
            bgm_volume=args.bgm_volume,
            bgm_mode=args.bgm_mode,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
