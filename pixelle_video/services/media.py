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
Media Generation Service - ComfyUI Workflow-based implementation

Supports both image and video generation workflows.
Automatically detects output type based on ExecuteResult.
"""

import asyncio
from typing import Optional

from comfykit import ComfyKit
from loguru import logger

from pixelle_video.services.comfy_base_service import ComfyBaseService
from pixelle_video.models.media import MediaResult


def _format_media_error(result) -> str:
    """Build an actionable error string from a failed ExecuteResult.

    comfykit surfaces a RunningHub task failure as e.g. "RunningHub task <id> failed: success".
    That trailing "success" is the HTTP envelope's ``msg`` (always "success" on HTTP 200), NOT
    the real reason — the RunningHub status API does not return one. Rewrite it into a message
    that points the user at the dashboard, where the actual reason can be seen.
    """
    raw = (result.msg or "Unknown error").strip()
    task_id = getattr(result, "prompt_id", None)
    if task_id and "failed" in raw.lower():
        return (
            f"RunningHub task {task_id} failed server-side. The status API does not "
            f"return a reason — open the task in the RunningHub dashboard to see why "
            f"(common causes: out of credits, workflow/node error, GPU OOM, content moderation). "
            f"[raw: {raw}]"
        )
    return raw


class MediaService(ComfyBaseService):
    """
    Media generation service - Workflow-based
    
    Uses ComfyKit to execute image/video generation workflows.
    Supports both image_ and video_ workflow prefixes.
    
    Usage:
        # Use default workflow (workflows/image_flux.json)
        media = await pixelle_video.media(prompt="a cat")
        if media.is_image:
            print(f"Generated image: {media.url}")
        elif media.is_video:
            print(f"Generated video: {media.url} ({media.duration}s)")
        
        # Use specific workflow
        media = await pixelle_video.media(
            prompt="a cat",
            workflow="image_flux.json"
        )
        
        # List available workflows
        workflows = pixelle_video.media.list_workflows()
    """
    
    WORKFLOW_PREFIX = ""  # Will be overridden by _scan_workflows
    DEFAULT_WORKFLOW = None  # No hardcoded default, must be configured
    WORKFLOWS_DIR = "workflows"
    
    def __init__(self, config: dict, core=None):
        """
        Initialize media service
        
        Args:
            config: Full application config dict
            core: PixelleVideoCore instance (for accessing shared ComfyKit)
        """
        super().__init__(config, service_name="image", core=core)  # Keep "image" for config compatibility
    
    def _scan_workflows(self):
        """
        Scan workflows for both image_ and video_ prefixes
        
        Override parent method to support multiple prefixes
        """
        from pixelle_video.utils.os_util import list_resource_dirs, list_resource_files, get_resource_path
        from pathlib import Path
        
        workflows = []
        
        # Get all workflow source directories
        source_dirs = list_resource_dirs("workflows")
        
        if not source_dirs:
            logger.warning("No workflow source directories found")
            return workflows
        
        # Scan each source directory for workflow files
        for source_name in source_dirs:
            # Get all JSON files for this source
            workflow_files = list_resource_files("workflows", source_name)
            
            # Filter to only files matching image_ or video_ prefix
            matching_files = [
                f for f in workflow_files 
                if (f.startswith("image_") or f.startswith("video_")) and f.endswith('.json')
            ]
            
            for filename in matching_files:
                try:
                    # Get actual file path
                    file_path = Path(get_resource_path("workflows", source_name, filename))
                    workflow_info = self._parse_workflow_file(file_path, source_name)
                    workflows.append(workflow_info)
                    logger.debug(f"Found workflow: {workflow_info['key']}")
                except Exception as e:
                    logger.error(f"Failed to parse workflow {source_name}/{filename}: {e}")
        
        # Sort by key (source/name)
        return sorted(workflows, key=lambda w: w["key"])
    
    async def __call__(
        self,
        prompt: str,
        workflow: Optional[str] = None,
        # Media type specification (required for proper handling)
        media_type: str = "image",  # "image" or "video"
        # ComfyUI connection (optional overrides)
        comfyui_url: Optional[str] = None,
        runninghub_api_key: Optional[str] = None,
        # Common workflow parameters
        width: Optional[int] = None,
        height: Optional[int] = None,
        duration: Optional[float] = None,  # Video duration in seconds (for video workflows)
        negative_prompt: Optional[str] = None,
        steps: Optional[int] = None,
        seed: Optional[int] = None,
        cfg: Optional[float] = None,
        sampler: Optional[str] = None,
        **params
    ) -> MediaResult:
        """
        Generate media (image or video) using workflow
        
        Media type must be specified explicitly via media_type parameter.
        Returns a MediaResult object containing media type and URL.
        
        Args:
            prompt: Media generation prompt
            workflow: Workflow filename (default: from config or "image_flux.json")
            media_type: Type of media to generate - "image" or "video" (default: "image")
            comfyui_url: ComfyUI URL (optional, overrides config)
            runninghub_api_key: RunningHub API key (optional, overrides config)
            width: Media width
            height: Media height
            duration: Target video duration in seconds (only for video workflows, typically from TTS audio duration)
            negative_prompt: Negative prompt
            steps: Sampling steps
            seed: Random seed
            cfg: CFG scale
            sampler: Sampler name
            **params: Additional workflow parameters
        
        Returns:
            MediaResult object with media_type ("image" or "video") and url
        
        Examples:
            # Simplest: use default workflow (workflows/image_flux.json)
            media = await pixelle_video.media(prompt="a beautiful cat")
            if media.is_image:
                print(f"Image: {media.url}")
            
            # Use specific workflow
            media = await pixelle_video.media(
                prompt="a cat",
                workflow="image_flux.json"
            )
            
            # Video workflow
            media = await pixelle_video.media(
                prompt="a cat running",
                workflow="image_video.json"
            )
            if media.is_video:
                print(f"Video: {media.url}, duration: {media.duration}s")
            
            # With additional parameters
            media = await pixelle_video.media(
                prompt="a cat",
                workflow="image_flux.json",
                width=1024,
                height=1024,
                steps=20,
                seed=42
            )
            
            # With absolute path
            media = await pixelle_video.media(
                prompt="a cat",
                workflow="/path/to/custom.json"
            )
            
            # With custom ComfyUI server
            media = await pixelle_video.media(
                prompt="a cat",
                comfyui_url="http://192.168.1.100:8188"
            )
        """
        # 1. Resolve workflow (returns structured info)
        workflow_info = self._resolve_workflow(workflow=workflow)
        
        # 2. Build workflow parameters (ComfyKit config is now managed by core)
        workflow_params = {"prompt": prompt}
        
        # Add optional parameters
        if width is not None:
            workflow_params["width"] = width
        if height is not None:
            workflow_params["height"] = height
        if duration is not None:
            workflow_params["duration"] = duration
            if media_type == "video":
                logger.info(f"📏 Target video duration: {duration:.2f}s (from TTS audio)")
        if negative_prompt is not None:
            workflow_params["negative_prompt"] = negative_prompt
        if steps is not None:
            workflow_params["steps"] = steps
        if seed is not None:
            workflow_params["seed"] = seed
        if cfg is not None:
            workflow_params["cfg"] = cfg
        if sampler is not None:
            workflow_params["sampler"] = sampler
        
        # Add any additional parameters
        workflow_params.update(params)
        
        logger.debug(f"Workflow parameters: {workflow_params}")
        
        # Determine what to pass to ComfyKit based on source (resolved once; not retried)
        if workflow_info["source"] == "runninghub" and "workflow_id" in workflow_info:
            # RunningHub: pass workflow_id (ComfyKit will use runninghub backend)
            workflow_input = workflow_info["workflow_id"]
            logger.info(f"Executing RunningHub workflow: {workflow_input}")
        else:
            # Selfhost: pass file path (ComfyKit will use local ComfyUI)
            workflow_input = workflow_info["path"]
            logger.info(f"Executing selfhost workflow: {workflow_input}")

        # 4. Execute workflow with retry. RunningHub task failures are frequently
        #    transient (queue eviction, GPU hiccup), so re-run the whole execute —
        #    each attempt creates a fresh task with a freshly randomized seed. When
        #    retries are exhausted we abort with a clear, actionable message.
        from pixelle_video.config import config_manager
        max_retries = config_manager.config.comfyui.media_max_retries
        attempts_total = max_retries + 1
        last_error: Optional[Exception] = None

        for attempt in range(attempts_total):
            try:
                # Get shared ComfyKit instance (lazy initialization + config hot-reload)
                kit = await self.core._get_or_create_comfykit()

                result = await kit.execute(workflow_input, workflow_params)

                # 5. Handle result based on specified media_type
                if result.status != "completed":
                    raise Exception(_format_media_error(result))

                # Extract media based on specified type
                if media_type == "video":
                    # Video workflow - get video from result
                    if not result.videos:
                        raise Exception("No video generated (workflow returned no videos)")

                    video_url = result.videos[0]
                    logger.info(f"✅ Generated video: {video_url}")

                    # Try to extract duration from result (if available)
                    duration = None
                    if hasattr(result, 'duration') and result.duration:
                        duration = result.duration

                    return MediaResult(
                        media_type="video",
                        url=video_url,
                        duration=duration
                    )
                else:  # image
                    # Image workflow - get image from result
                    if not result.images:
                        raise Exception("No image generated (workflow returned no images)")

                    image_url = result.images[0]
                    logger.info(f"✅ Generated image: {image_url}")

                    return MediaResult(
                        media_type="image",
                        url=image_url
                    )

            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    wait = 2 ** attempt
                    logger.warning(
                        f"Media generation attempt {attempt + 1}/{attempts_total} failed: {e}. "
                        f"Retrying in {wait}s..."
                    )
                    await asyncio.sleep(wait)
                else:
                    logger.error(
                        f"Media generation failed after {attempts_total} attempt(s): {e}"
                    )

        # All attempts failed — propagate the clear message (aborts the whole video).
        raise Exception(f"Media generation failed: {last_error}")
