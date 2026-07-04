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
Frame processor - Process single frame through complete pipeline

Orchestrates: TTS → Image Generation → Frame Composition → Video Segment

Key Feature:
- TTS-driven video duration: Audio duration from TTS is passed to video generation workflows
  to ensure perfect sync between audio and video (no padding, no trimming needed)
"""

import re
from typing import Callable, List, Optional

import httpx
from loguru import logger

from pixelle_video.models.progress import ProgressEvent
from pixelle_video.models.storyboard import Storyboard, StoryboardFrame, StoryboardConfig


# Default negative prompt for video (image→video) generation. Image→video models
# (e.g. Wan2.2) intermittently produce anatomy/body-horror artifacts on people —
# detached heads, extra/fused limbs, melting — especially with babies/children. This
# discourages those failure modes. Override per-deployment via config
# ``media.negative_prompt`` (set to "" to disable). Only applied when the workflow
# itself maps a negative-prompt input; harmless otherwise.
DEFAULT_VIDEO_NEGATIVE_PROMPT = (
    "deformed anatomy, detached head, decapitation, severed head, floating head, "
    "dismemberment, body horror, gore, extra limbs, extra fingers, fused limbs, "
    "distorted body, malformed face, malformed baby, mutation, disfigured, "
    "broken neck, melting, glitch, flickering, morphing"
)


def _allocate_durations(chunks: List[str], total: float, min_dur: float = 0.4) -> List[float]:
    """
    Split ``total`` seconds across subtitle ``chunks`` proportional to their text
    length (non-space characters), with a small per-chunk floor so very short phrases
    don't flash by.

    The returned durations always sum to ``total`` (any floating-point remainder is
    added to the last chunk). Returns ``[]`` when there are no chunks.
    """
    n = len(chunks)
    if n == 0:
        return []
    if total <= 0:
        return [0.0] * n
    if n == 1:
        return [float(total)]

    weights = [max(len(re.sub(r'\s+', '', c)), 1) for c in chunks]

    if min_dur * n >= total:
        # Floor can't fit -> even split.
        durs = [total / n] * n
    else:
        free = total - min_dur * n
        wsum = sum(weights)
        durs = [min_dur + free * (w / wsum) for w in weights]

    # Correct floating-point drift so the durations sum to exactly `total`.
    durs[-1] += total - sum(durs)
    return durs


class FrameProcessor:
    """Frame processor"""
    
    def __init__(self, pixelle_video_core):
        """
        Initialize
        
        Args:
            pixelle_video_core: PixelleVideoCore instance
        """
        self.core = pixelle_video_core
    
    async def __call__(
        self,
        frame: StoryboardFrame,
        storyboard: 'Storyboard',
        config: StoryboardConfig,
        total_frames: int = 1,
        progress_callback: Optional[Callable[[ProgressEvent], None]] = None
    ) -> StoryboardFrame:
        """
        Process single frame through complete pipeline
        
        Steps:
        1. Generate audio (TTS)
        2. Generate image (ComfyKit)
        3. Compose frame (add subtitle)
        4. Create video segment (image + audio)
        
        Args:
            frame: Storyboard frame to process
            storyboard: Storyboard instance
            config: Storyboard configuration
            total_frames: Total number of frames in storyboard
            progress_callback: Optional callback for progress updates (receives ProgressEvent)
            
        Returns:
            Processed frame with all paths filled
        """
        logger.info(f"Processing frame {frame.index}...")
        
        frame_num = frame.index + 1
        
        # Determine if this frame needs image generation
        # If image_path or video_path is already set (e.g. asset-based pipeline), we consider it "has existing media" but skip generation
        has_existing_media = frame.image_path is not None or frame.video_path is not None
        needs_generation = frame.image_prompt is not None
        
        try:
            # Step 1: Generate audio (TTS)
            if not frame.audio_path:
                if progress_callback:
                    progress_callback(ProgressEvent(
                        event_type="frame_step",
                        progress=0.0,
                        frame_current=frame_num,
                        frame_total=total_frames,
                        step=1,
                        action="audio"
                    ))
                await self._step_generate_audio(frame, config)
            else:
                logger.debug(f"  1/4: Using existing audio: {frame.audio_path}")
            
            # Step 2: Generate media (image or video, conditional)
            if needs_generation:
                if progress_callback:
                    progress_callback(ProgressEvent(
                        event_type="frame_step",
                        progress=0.25,
                        frame_current=frame_num,
                        frame_total=total_frames,
                        step=2,
                        action="media"
                    ))
                await self._step_generate_media(frame, config)
            elif has_existing_media:
                # Log appropriate message based on media type
                if frame.video_path:
                    logger.debug(f"  2/4: Using existing video: {frame.video_path}")
                else:
                    logger.debug(f"  2/4: Using existing image: {frame.image_path}")
            else:
                frame.image_path = None
                frame.media_type = None
                logger.debug(f"  2/4: Skipped media generation (not required by template)")
        
            # Step 3: Compose frame (add subtitle)
            if progress_callback:
                progress_callback(ProgressEvent(
                    event_type="frame_step",
                    progress=0.50 if (needs_generation or has_existing_media) else 0.33,
                    frame_current=frame_num,
                    frame_total=total_frames,
                    step=3,
                    action="compose"
                ))
            await self._step_compose_frame(frame, storyboard, config)
            
            # Step 4: Create video segment
            if progress_callback:
                progress_callback(ProgressEvent(
                    event_type="frame_step",
                    progress=0.75 if (needs_generation or has_existing_media) else 0.67,
                    frame_current=frame_num,
                    frame_total=total_frames,
                    step=4,
                    action="video"
                ))
            
            await self._step_create_video_segment(frame, config)
            
            logger.info(f"✅ Frame {frame.index} completed")
            return frame

        except Exception as e:
            logger.error(f"❌ Failed to process frame {frame.index}: {e}")
            raise
    
    async def _step_generate_audio(
        self,
        frame: StoryboardFrame,
        config: StoryboardConfig
    ):
        """Step 1: Generate audio using TTS"""
        logger.debug(f"  1/4: Generating audio for frame {frame.index}...")
        
        # Generate output path using task_id
        from pixelle_video.utils.os_util import get_task_frame_path
        output_path = get_task_frame_path(config.task_id, frame.index, "audio")
        
        # Build TTS params based on inference mode
        tts_params = {
            "text": frame.narration,
            "inference_mode": config.tts_inference_mode,
            "output_path": output_path,
            "index": frame.index + 1,  # 1-based index for workflow
        }

        # The first segment is the title/hook — ask the prosody layer to deliver it
        # with extra punch (slower + brighter + dramatic trailing pause) to grab the
        # viewer in the first second. Ignored by engines that don't use it.
        if frame.index == 0:
            tts_params["emphasis"] = "title"

        if config.tts_inference_mode == "local":
            # Local mode: pass voice and speed
            if config.voice_id:
                tts_params["voice"] = config.voice_id
            if config.tts_speed is not None:
                tts_params["speed"] = config.tts_speed
        else:  # comfyui
            # ComfyUI mode: pass workflow, voice, speed, and ref_audio
            if config.tts_workflow:
                tts_params["workflow"] = config.tts_workflow
            if config.voice_id:
                tts_params["voice"] = config.voice_id
            if config.tts_speed is not None:
                tts_params["speed"] = config.tts_speed
            if config.ref_audio:
                tts_params["ref_audio"] = config.ref_audio
        
        audio_path = await self.core.tts(**tts_params)
        
        frame.audio_path = audio_path
        
        # Get audio duration
        frame.duration = await self._get_audio_duration(audio_path)
        
        logger.debug(f"  ✓ Audio generated: {audio_path} ({frame.duration:.2f}s)")
    
    async def _step_generate_media(
        self,
        frame: StoryboardFrame,
        config: StoryboardConfig
    ):
        """Step 2: Generate media (image or video) using ComfyKit"""
        logger.debug(f"  2/4: Generating media for frame {frame.index}...")
        
        # Determine media type based on workflow
        # video_ prefix in workflow name indicates video generation
        workflow_name = config.media_workflow or ""
        is_video_workflow = "video_" in workflow_name.lower()
        media_type = "video" if is_video_workflow else "image"
        
        logger.debug(f"  → Media type: {media_type} (workflow: {workflow_name})")
        
        # Build media generation parameters
        media_params = {
            "prompt": frame.image_prompt,
            "workflow": config.media_workflow,  # Pass workflow from config (None = use default)
            "media_type": media_type,
            "width": config.media_width,
            "height": config.media_height,
            "index": frame.index + 1,  # 1-based index for workflow
        }
        
        # For video workflows: pass audio duration as target video duration
        # This ensures video length matches audio length from the source
        if is_video_workflow and frame.duration:
            media_params["duration"] = frame.duration
            logger.info(f"  → Generating video with target duration: {frame.duration:.2f}s (from TTS audio)")

        # For video workflows: steer away from anatomy/body-horror artifacts with a
        # default negative prompt (overridable via config ``media.negative_prompt``,
        # "" to disable). No-op for workflows that don't map a negative-prompt input.
        if is_video_workflow:
            media_cfg = self.core.config.get("media", {}) if self.core.config else {}
            negative_prompt = media_cfg.get("negative_prompt", DEFAULT_VIDEO_NEGATIVE_PROMPT)
            if negative_prompt:
                media_params["negative_prompt"] = negative_prompt

        # Call Media generation
        media_result = await self.core.media(**media_params)
        
        # Store media type
        frame.media_type = media_result.media_type
        
        if media_result.is_image:
            # Download image to local (pass task_id)
            local_path = await self._download_media(
                media_result.url,
                frame.index,
                config.task_id,
                media_type="image"
            )
            frame.image_path = local_path
            logger.debug(f"  ✓ Image generated: {local_path}")
        
        elif media_result.is_video:
            # Download video to local (pass task_id)
            local_path = await self._download_media(
                media_result.url,
                frame.index,
                config.task_id,
                media_type="video"
            )
            frame.video_path = local_path
            
            # Update duration from video if available
            if media_result.duration:
                frame.duration = media_result.duration
                logger.debug(f"  ✓ Video generated: {local_path} (duration: {frame.duration:.2f}s)")
            else:
                # Get video duration from file
                frame.duration = await self._get_video_duration(local_path)
                logger.debug(f"  ✓ Video generated: {local_path} (duration: {frame.duration:.2f}s)")
        
        else:
            raise ValueError(f"Unknown media type: {media_result.media_type}")
    
    async def _step_compose_frame(
        self,
        frame: StoryboardFrame,
        storyboard: 'Storyboard',
        config: StoryboardConfig
    ):
        """Step 3: Compose frame with subtitle using HTML template"""
        logger.debug(f"  3/4: Composing frame {frame.index}...")

        # Generate output path using task_id
        from pixelle_video.utils.os_util import get_task_frame_path, get_task_path

        # Progressive (read-along) subtitles: render one composed frame per subtitle
        # chunk (same media/title, different subtitle text) so the subtitle can advance
        # with the voice. Falls back to a single render for short / single-chunk
        # narrations (no behavior change, no extra renders).
        if getattr(config, "subtitle_sync", False) and frame.duration and frame.narration:
            from pixelle_video.utils.content_generators import split_into_subtitle_chunks

            chunks = split_into_subtitle_chunks(frame.narration, config.subtitle_max_chars)
            if len(chunks) > 1:
                frame.subtitle_chunks = chunks
                frame.subtitle_durations = _allocate_durations(chunks, frame.duration)
                frame.composed_image_paths = []
                for ci, chunk in enumerate(chunks):
                    chunk_path = get_task_path(
                        config.task_id, "frames",
                        f"{frame.index + 1:02d}_composed_{ci:02d}.png"
                    )
                    composed = await self._compose_frame_html(
                        frame, storyboard, config, chunk_path, text_override=chunk
                    )
                    frame.composed_image_paths.append(composed)
                # Keep composed_image_path set (first chunk) for back-compat.
                frame.composed_image_path = frame.composed_image_paths[0]
                logger.debug(
                    f"  ✓ Frame composed into {len(chunks)} subtitle chunk(s)"
                )
                return

        # Default: single composed image with the full narration.
        # For video type: HTML renders as a transparent overlay image.
        # For image type: HTML renders with the image background.
        output_path = get_task_frame_path(config.task_id, frame.index, "composed")
        composed_path = await self._compose_frame_html(frame, storyboard, config, output_path)

        frame.composed_image_path = composed_path

        logger.debug(f"  ✓ Frame composed: {composed_path}")
    
    async def _compose_frame_html(
        self,
        frame: StoryboardFrame,
        storyboard: 'Storyboard',
        config: StoryboardConfig,
        output_path: str,
        text_override: Optional[str] = None
    ) -> str:
        """Compose frame using HTML template.

        ``text_override`` lets callers render a subtitle chunk instead of the full
        narration (used for progressive read-along subtitles).
        """
        from pixelle_video.services.frame_html import HTMLFrameGenerator
        from pixelle_video.utils.template_util import resolve_template_path
        
        # Resolve template path (handles various input formats)
        template_path = resolve_template_path(config.frame_template)
        
        # Get content metadata from storyboard
        content_metadata = storyboard.content_metadata if storyboard else None
        
        # Build ext data
        ext = {
            "index": frame.index + 1,
        }
        
        # Add custom template parameters
        if config.template_params:
            ext.update(config.template_params)
        
        # Generate frame using HTML (size is auto-parsed from template path)
        generator = HTMLFrameGenerator(template_path)
        
        # Use video_path for video media, image_path for images
        media_path = frame.video_path if frame.media_type == "video" else frame.image_path
        logger.debug(f"Generating frame with media: '{media_path}' (type: {frame.media_type})")
        
        composed_path = await generator.generate_frame(
            title=storyboard.title,
            text=text_override if text_override is not None else frame.narration,
            image=media_path,  # HTMLFrameGenerator handles both image and video paths
            ext=ext,
            output_path=output_path
        )

        return composed_path
    
    async def _step_create_video_segment(
        self,
        frame: StoryboardFrame,
        config: StoryboardConfig
    ):
        """Step 4: Create video segment from media + audio"""
        logger.debug(f"  4/4: Creating video segment for frame {frame.index}...")
        
        # Generate output path using task_id
        from pixelle_video.utils.os_util import get_task_frame_path
        output_path = get_task_frame_path(config.task_id, frame.index, "segment")
        
        from pixelle_video.services.video import VideoService
        video_service = VideoService()
        
        # Branch based on media type
        if frame.media_type == "video":
            # Video workflow: overlay HTML template on video, then add audio
            logger.debug(f"  → Using video-based composition with HTML overlay")

            # Step 1: Overlay the transparent HTML image(s) on the video.
            # With progressive subtitles, composed_image_paths holds one overlay per
            # subtitle chunk (each shown during its time window); otherwise a single
            # composed_image_path overlay is used for the whole segment.
            temp_video_with_overlay = get_task_frame_path(config.task_id, frame.index, "video") + "_overlay.mp4"

            # If the template declares a video window rect, place the video into
            # that window (subtitle sits below it). Otherwise fall back to the
            # full-frame overlay.
            from pixelle_video.utils.template_util import (
                parse_template_video_region, resolve_template_path, parse_template_size
            )
            region = parse_template_video_region(
                resolve_template_path(config.frame_template)
            )

            if frame.composed_image_paths:
                # Build cumulative time windows for the per-chunk overlays. Extend the
                # last window well past the end so the final subtitle never vanishes a
                # frame early when the video is slightly longer than the audio.
                overlays = []
                start = 0.0
                for img, dur in zip(frame.composed_image_paths, frame.subtitle_durations):
                    overlays.append({"image": img, "start": start, "end": start + dur})
                    start += dur
                overlays[-1]["end"] = start + 3600.0

                if region:
                    canvas_w, canvas_h = parse_template_size(
                        resolve_template_path(config.frame_template)
                    )
                    video_service.composite_video_in_region_timed(
                        video=frame.video_path,
                        overlays=overlays,
                        output=temp_video_with_overlay,
                        region=region,
                        canvas_size=(canvas_w, canvas_h),
                        bg_color=region.get("bg_color", "#000000"),
                        fps=config.video_fps,
                    )
                else:
                    video_service.overlay_images_on_video_timed(
                        video=frame.video_path,
                        overlays=overlays,
                        output=temp_video_with_overlay,
                        scale_mode="contain",
                    )
            elif region:
                canvas_w, canvas_h = parse_template_size(
                    resolve_template_path(config.frame_template)
                )
                video_service.composite_video_in_region(
                    video=frame.video_path,
                    overlay_image=frame.composed_image_path,
                    output=temp_video_with_overlay,
                    region=region,
                    canvas_size=(canvas_w, canvas_h),
                    bg_color=region.get("bg_color", "#000000"),
                    fps=config.video_fps,
                )
            else:
                video_service.overlay_image_on_video(
                    video=frame.video_path,
                    overlay_image=frame.composed_image_path,
                    output=temp_video_with_overlay,
                    scale_mode="contain"  # Scale video to fit template size (contain mode)
                )
            
            # Step 2: Add narration audio to the overlaid video
            # Note: The video might have audio (replaced) or be silent (audio added)
            segment_path = video_service.merge_audio_video(
                video=temp_video_with_overlay,
                audio=frame.audio_path,
                output=output_path,
                replace_audio=True,  # Replace video audio with narration
                audio_volume=1.0
            )
            
            # Clean up temp file (best-effort). On Windows the ffmpeg subprocess can
            # briefly keep the handle open after returning, so a failed unlink must not
            # abort the whole frame — the segment is already written at this point.
            import os
            if os.path.exists(temp_video_with_overlay):
                try:
                    os.unlink(temp_video_with_overlay)
                except OSError as e:
                    logger.warning(f"Could not remove temp overlay file {temp_video_with_overlay}: {e}")
        
        elif frame.media_type == "image" or frame.media_type is None:
            # Image workflow: Use composed image directly
            # The asset_default.html template includes the image in the composition
            logger.debug(f"  → Using image-based composition")

            if frame.composed_image_paths:
                # Progressive subtitles: still "slideshow" of per-chunk images + audio.
                segment_path = video_service.create_video_from_image_sequence(
                    images=frame.composed_image_paths,
                    durations=frame.subtitle_durations,
                    audio=frame.audio_path,
                    output=output_path,
                    fps=config.video_fps,
                )
            else:
                segment_path = video_service.create_video_from_image(
                    image=frame.composed_image_path,
                    audio=frame.audio_path,
                    output=output_path,
                    fps=config.video_fps
                )
        
        else:
            raise ValueError(f"Unknown media type: {frame.media_type}")
        
        frame.video_segment_path = segment_path
        
        logger.debug(f"  ✓ Video segment created: {segment_path}")
    
    async def _get_audio_duration(self, audio_path: str) -> float:
        """Get audio duration in seconds"""
        try:
            # Try using ffmpeg-python
            import ffmpeg
            probe = ffmpeg.probe(audio_path)
            duration = float(probe['format']['duration'])
            return duration
        except Exception as e:
            logger.warning(f"Failed to get audio duration: {e}, using estimate")
            # Fallback: estimate based on file size (very rough)
            import os
            file_size = os.path.getsize(audio_path)
            # Assume ~16kbps for MP3, so 2KB per second
            estimated_duration = file_size / 2000
            return max(1.0, estimated_duration)  # At least 1 second
    
    async def _download_media(
        self,
        url: str,
        frame_index: int,
        task_id: str,
        media_type: str
    ) -> str:
        """Download media (image or video) from URL to local file.

        Retries on transient network errors (connection dropped mid-download,
        read timeouts, 5xx responses), which are common with the remote media
        host and would otherwise fail the whole frame/pipeline.
        """
        import asyncio

        from pixelle_video.utils.os_util import get_task_frame_path
        output_path = get_task_frame_path(task_id, frame_index, media_type)

        timeout = httpx.Timeout(connect=10.0, read=120, write=60, pool=60)
        max_attempts = 4

        last_error: Optional[Exception] = None
        for attempt in range(1, max_attempts + 1):
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    async with client.stream("GET", url) as response:
                        response.raise_for_status()
                        with open(output_path, 'wb') as f:
                            async for chunk in response.aiter_bytes(chunk_size=65536):
                                f.write(chunk)
                return output_path
            except (httpx.TransportError, httpx.HTTPStatusError) as e:
                # Don't retry client errors (4xx) - they won't recover
                if isinstance(e, httpx.HTTPStatusError) and e.response.status_code < 500:
                    raise
                last_error = e
                if attempt < max_attempts:
                    backoff = 2 ** (attempt - 1)  # 1s, 2s, 4s
                    logger.warning(
                        f"  ⚠ Download attempt {attempt}/{max_attempts} failed for "
                        f"frame {frame_index} ({type(e).__name__}); retrying in {backoff}s"
                    )
                    await asyncio.sleep(backoff)

        raise RuntimeError(
            f"Failed to download media after {max_attempts} attempts: {url}"
        ) from last_error
    
    async def _get_video_duration(self, video_path: str) -> float:
        """Get video duration in seconds"""
        try:
            import ffmpeg
            probe = ffmpeg.probe(video_path)
            duration = float(probe['format']['duration'])
            return duration
        except Exception as e:
            logger.warning(f"Failed to get video duration: {e}, using audio duration")
            # Fallback: use audio duration if available
            return 1.0  # Default to 1 second if unable to determine

