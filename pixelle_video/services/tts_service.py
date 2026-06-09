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
TTS (Text-to-Speech) Service - Supports both local and ComfyUI inference
"""

import os
import uuid
from pathlib import Path
from typing import Optional

from comfykit import ComfyKit
from loguru import logger

from pixelle_video.services.comfy_base_service import ComfyBaseService
from pixelle_video.utils.tts_util import edge_tts
from pixelle_video.tts_voices import speed_to_rate, resolve_custom_voice, resolve_vieneu_voice


class TTSService(ComfyBaseService):
    """
    TTS (Text-to-Speech) service - Workflow-based
    
    Uses ComfyKit to execute TTS workflows.
    
    Usage:
        # Use default workflow
        audio_path = await pixelle_video.tts(text="Hello, world!")
        
        # Use specific workflow
        audio_path = await pixelle_video.tts(
            text="你好，世界！",
            workflow="tts_edge.json"
        )
        
        # List available workflows
        workflows = pixelle_video.tts.list_workflows()
    """
    
    WORKFLOW_PREFIX = "tts_"
    DEFAULT_WORKFLOW = None  # No hardcoded default, must be configured
    WORKFLOWS_DIR = "workflows"
    
    def __init__(self, config: dict, core=None):
        """
        Initialize TTS service
        
        Args:
            config: Full application config dict
            core: PixelleVideoCore instance (for accessing shared ComfyKit)
        """
        super().__init__(config, service_name="tts", core=core)
    
    
    async def __call__(
        self,
        text: str,
        workflow: Optional[str] = None,
        # ComfyUI connection (optional overrides)
        comfyui_url: Optional[str] = None,
        runninghub_api_key: Optional[str] = None,
        # TTS parameters
        voice: Optional[str] = None,
        speed: Optional[float] = None,
        # Inference mode override
        inference_mode: Optional[str] = None,
        # Output path
        output_path: Optional[str] = None,
        **params
    ) -> str:
        """
        Generate speech using local Edge TTS or ComfyUI workflow
        
        Args:
            text: Text to convert to speech
            workflow: Workflow filename (for ComfyUI mode, default: from config)
            comfyui_url: ComfyUI URL (optional, overrides config)
            runninghub_api_key: RunningHub API key (optional, overrides config)
            voice: Voice ID (for local mode: Edge TTS voice ID; for ComfyUI: workflow-specific)
            speed: Speech speed multiplier (1.0 = normal, >1.0 = faster, <1.0 = slower)
            inference_mode: Override inference mode ("local" or "comfyui", default: from config)
            output_path: Custom output path (auto-generated if None)
            **params: Additional workflow parameters
        
        Returns:
            Generated audio file path
        
        Examples:
            # Local inference (Edge TTS)
            audio_path = await pixelle_video.tts(
                text="Hello, world!",
                inference_mode="local",
                voice="zh-CN-YunjianNeural",
                speed=1.2
            )
            
            # ComfyUI inference
            audio_path = await pixelle_video.tts(
                text="你好，世界！",
                inference_mode="comfyui",
                workflow="runninghub/tts_edge.json"
            )
        """
        # Determine inference mode (param > config)
        mode = inference_mode or self.config.get("inference_mode", "local")
        
        # Route to appropriate implementation
        if mode == "local":
            result_path = await self._call_local_tts(
                text=text,
                voice=voice,
                speed=speed,
                output_path=output_path
            )
            # Normalize loudness so every voice (clones + Edge) is at the same
            # standard level (F5 otherwise tracks each reference clip's loudness).
            target_lufs = self.config.get("local", {}).get("target_lufs", -16.0)
            return await self._normalize_loudness(result_path, target_lufs)
        else:  # comfyui
            # 1. Resolve workflow (returns structured info)
            workflow_info = self._resolve_workflow(workflow=workflow)
            
            # 2. Execute ComfyUI workflow
            return await self._call_comfyui_workflow(
                workflow_info=workflow_info,
                text=text,
                comfyui_url=comfyui_url,
                runninghub_api_key=runninghub_api_key,
                voice=voice,
                speed=speed,
                output_path=output_path,
                **params
            )
    
    async def _normalize_loudness(self, path: str, target_lufs: float = -16.0) -> str:
        """
        Normalize a narration audio file to a standard integrated loudness (EBU R128),
        so every voice comes out at the same level regardless of the source/reference.

        Two-pass ffmpeg ``loudnorm`` (measure, then correct with linear=true) for
        accurate, consistent targeting. On any failure (or if disabled via a falsy
        target_lufs) the original file is left untouched. Returns the path.
        """
        if not target_lufs:
            return path

        import asyncio
        return await asyncio.to_thread(self._level_to_target, path, float(target_lufs))

    def _measure_lufs(self, path: str) -> Optional[float]:
        """Measure integrated loudness (LUFS) of a file via ffmpeg loudnorm."""
        import json
        import re
        import subprocess
        r = subprocess.run(
            ["ffmpeg", "-i", path, "-af", "loudnorm=I=-16:print_format=json",
             "-f", "null", "-"],
            capture_output=True, text=True,
        )
        m = re.search(r"\{[\s\S]*?\}", r.stderr)
        if not m:
            return None
        try:
            val = float(json.loads(m.group(0))["input_i"])
        except (ValueError, KeyError, json.JSONDecodeError):
            return None
        return val if val > -70 else None  # -inf/near-silence -> skip

    def _level_to_target(self, path: str, target_lufs: float) -> str:
        """
        Bring a clip to an exact integrated loudness, in place.

        Two stages because plain ``loudnorm`` won't boost peaky speech past its
        true-peak ceiling:
          1) ``loudnorm`` (TP-safe) does the bulk normalization,
          2) a corrective ``volume`` gain + ``alimiter`` hits the exact target.
        Falls back gracefully (keeps the best available result) on any error.
        """
        import subprocess

        s1 = f"{path}.s1.mp3"
        s2 = f"{path}.s2.mp3"
        try:
            # Stage 1: TP-safe bulk loudness normalization
            subprocess.run(
                ["ffmpeg", "-y", "-i", path,
                 "-af", f"loudnorm=I={target_lufs}:TP=-1.5:LRA=11",
                 "-ar", "44100", "-loglevel", "error", s1],
                check=True, capture_output=True, text=True,
            )

            # Stage 2: measure residual and apply exact corrective gain + limiter
            measured = self._measure_lufs(s1)
            if measured is None:
                os.replace(s1, path)
                return path
            gain = target_lufs - measured
            subprocess.run(
                ["ffmpeg", "-y", "-i", s1,
                 "-af", f"volume={gain:.2f}dB,alimiter=limit=0.95",
                 "-ar", "44100", "-loglevel", "error", s2],
                check=True, capture_output=True, text=True,
            )
            os.replace(s2, path)
        except Exception as e:
            logger.warning(f"Loudness normalization skipped ({e}); keeping original")
        finally:
            for tmp in (s1, s2):
                try:
                    if os.path.exists(tmp):
                        os.unlink(tmp)
                except OSError:
                    pass
        return path

    async def _call_local_tts(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: Optional[float] = None,
        output_path: Optional[str] = None,
    ) -> str:
        """
        Generate speech using local Edge TTS
        
        Args:
            text: Text to convert to speech
            voice: Edge TTS voice ID (default: from config)
            speed: Speech speed multiplier (default: from config)
            output_path: Custom output path (auto-generated if None)
        
        Returns:
            Generated audio file path
        """
        # Get config defaults
        local_config = self.config.get("local", {})
        
        # Determine voice and speed (param > config)
        final_voice = voice or local_config.get("voice", "vi-VN-NamMinhNeural")
        final_speed = speed if speed is not None else local_config.get("speed", 1.2)

        # Convert speed to rate parameter
        rate = speed_to_rate(final_speed)

        # Generate output path if not provided
        if not output_path:
            # Generate unique filename
            unique_id = uuid.uuid4().hex
            output_path = f"output/{unique_id}.mp3"

            # Ensure output directory exists
            Path("output").mkdir(parents=True, exist_ok=True)

        # VieNeu preset voice: self-contained Vietnamese named-voice engine (no ref clip).
        vieneu = resolve_vieneu_voice(final_voice)
        if vieneu:
            return await self._call_vieneu(text, vieneu, final_speed, output_path)

        # Cloned voice: route to the configured clone engine.
        #   vixtts   -> true zero-shot cloning (timbre + intonation), Vietnamese
        #   openvoice -> Edge TTS base + OpenVoice tone-color conversion (fallback)
        custom = resolve_custom_voice(final_voice)
        if custom:
            engine = local_config.get("clone_engine", "f5tts")
            if engine == "f5tts":
                return await self._call_f5tts(text, custom, final_speed, output_path)
            if engine == "knnvc":
                return await self._call_knnvc(text, custom, final_speed, output_path)
            if engine == "vixtts":
                return await self._call_vixtts(text, custom, final_speed, output_path)
            return await self._call_cloned_tts(text, custom, rate, output_path)

        logger.info(f"🎙️  Using local Edge TTS: voice={final_voice}, speed={final_speed}x (rate={rate})")

        # Call Edge TTS
        try:
            audio_bytes = await edge_tts(
                text=text,
                voice=final_voice,
                rate=rate,
                output_path=output_path
            )
            
            logger.info(f"✅ Generated audio (local Edge TTS): {output_path}")
            return output_path
        
        except Exception as e:
            logger.error(f"Local TTS generation error: {e}")
            raise

    async def _call_f5tts(
        self,
        text: str,
        custom: dict,
        speed: float,
        output_path: str,
    ) -> str:
        """
        Generate a cloned voice with F5-TTS Vietnamese (in-context cloning).

        F5-TTS clones the reference's timbre AND Northern accent directly from
        ``custom["ref"]`` and speaks Vietnamese natively. Returns output_path (mp3).
        """
        import asyncio

        ref_path = custom["ref"]
        logger.info(f"🧬 Cloned voice '{custom['id']}' via F5-TTS -> ref={ref_path}")

        from pixelle_video.services.f5tts_service import F5TTSEngine
        engine = F5TTSEngine()
        # Speed/quality knobs (config): nfe_step (diffusion steps), ref_sec (reference
        # length — the dominant cost on CPU), quantize (int8).
        local_cfg = self.config.get("local", {})
        nfe = int(local_cfg.get("clone_nfe_step", 16))
        ref_sec = float(local_cfg.get("clone_ref_sec", 7.0))
        quantize = bool(local_cfg.get("clone_quantize", False))
        out_wav = f"{output_path}.f5.wav"
        try:
            await asyncio.to_thread(
                engine.synthesize, text, ref_path, out_wav, speed, nfe, ref_sec, quantize
            )

            import ffmpeg
            (
                ffmpeg
                .input(out_wav)
                .output(output_path, loglevel="error")
                .overwrite_output()
                .run()
            )
        finally:
            try:
                if os.path.exists(out_wav):
                    os.unlink(out_wav)
            except OSError:
                pass

        logger.info(f"✅ Generated cloned audio (F5-TTS): {output_path}")
        return output_path

    async def _call_knnvc(
        self,
        text: str,
        custom: dict,
        speed: float,
        output_path: str,
    ) -> str:
        """
        Generate a cloned voice with a Northern Vietnamese base TTS + kNN-VC.

        The base (gTTS Vietnamese, which is Northern/Hanoi — the Edge vi-VN voices
        are Southern) provides the Vietnamese pronunciation/accent; kNN-VC then
        converts only the timbre to the reference clip. Returns output_path (mp3).
        """
        import asyncio

        ref_path = custom["ref"]
        logger.info(f"🧬 Cloned voice '{custom['id']}' via kNN-VC -> timbre={ref_path}")

        # 1. Northern Vietnamese base speech
        base_path = f"{output_path}.base.mp3"
        await self._synthesize_clone_base(text, custom, speed, base_path)

        # 2. kNN-VC timbre conversion (blocking torch -> run in thread)
        from pixelle_video.services.knnvc_service import KNNVCConverter
        converter = KNNVCConverter()
        converted_wav = f"{output_path}.knn.wav"
        try:
            await asyncio.to_thread(
                converter.convert, base_path, ref_path, converted_wav
            )

            # 3. Transcode converted wav to the requested mp3 output path
            import ffmpeg
            (
                ffmpeg
                .input(converted_wav)
                .output(output_path, loglevel="error")
                .overwrite_output()
                .run()
            )
        finally:
            for tmp in (base_path, converted_wav):
                try:
                    if os.path.exists(tmp):
                        os.unlink(tmp)
                except OSError:
                    pass

        logger.info(f"✅ Generated cloned audio (kNN-VC): {output_path}")
        return output_path

    async def _synthesize_clone_base(
        self,
        text: str,
        custom: dict,
        speed: float,
        output_mp3: str,
    ) -> str:
        """
        Synthesize the clone's base speech (pronunciation/accent), before timbre
        conversion. Defaults to gTTS Vietnamese (Northern); falls back to the Edge
        base voice if ``base_engine == "edge"``.
        """
        import asyncio

        base_engine = custom.get("base_engine", "gtts")

        if base_engine == "gtts":
            from gtts import gTTS

            lang = custom.get("base_lang", "vi")

            def _gen():
                gTTS(text, lang=lang).save(output_mp3)

            await asyncio.to_thread(_gen)

            # gTTS has no speed control -> adjust tempo with ffmpeg if needed
            if speed and abs(speed - 1.0) > 0.01:
                import ffmpeg
                tempo = max(0.5, min(2.0, speed))
                adjusted = f"{output_mp3}.spd.mp3"
                (
                    ffmpeg
                    .input(output_mp3)
                    .output(adjusted, **{"filter:a": f"atempo={tempo}"}, loglevel="error")
                    .overwrite_output()
                    .run()
                )
                os.replace(adjusted, output_mp3)
        else:  # edge fallback (note: Edge vi-VN is Southern)
            rate = speed_to_rate(speed)
            await edge_tts(
                text=text, voice=custom["base"], rate=rate, output_path=output_mp3
            )

        return output_mp3

    async def _call_vixtts(
        self,
        text: str,
        custom: dict,
        speed: float,
        output_path: str,
    ) -> str:
        """
        Generate a cloned voice with viXTTS (true Vietnamese zero-shot cloning).

        Args:
            text: Text to synthesize
            custom: Custom-voice config (keys: id, name, ref, ...)
            speed: Speech speed multiplier
            output_path: Final audio path (mp3)

        Returns:
            Generated audio file path (output_path)
        """
        import asyncio

        ref_path = custom["ref"]
        logger.info(f"🧬 Cloned voice '{custom['id']}' via viXTTS -> timbre={ref_path}")

        from pixelle_video.services.vixtts_service import ViXTTSEngine
        engine = ViXTTSEngine()
        out_wav = f"{output_path}.vixtts.wav"
        try:
            # Blocking torch inference -> run in a thread
            await asyncio.to_thread(
                engine.synthesize, text, ref_path, out_wav, speed, "vi"
            )

            # Transcode the 24kHz wav to the requested mp3 output path
            import ffmpeg
            (
                ffmpeg
                .input(out_wav)
                .output(output_path, loglevel="error")
                .overwrite_output()
                .run()
            )
        finally:
            try:
                if os.path.exists(out_wav):
                    os.unlink(out_wav)
            except OSError:
                pass

        logger.info(f"✅ Generated cloned audio (viXTTS): {output_path}")
        return output_path

    async def _call_vieneu(
        self,
        text: str,
        vieneu: dict,
        speed: float,
        output_path: str,
    ) -> str:
        """
        Generate speech with a VieNeu-TTS preset voice (Vietnamese named-voice engine).

        VieNeu speaks the text directly in the named preset voice (no reference clip).
        It has no speed control, so tempo is applied here via ffmpeg ``atempo`` while
        transcoding the 48 kHz wav to the requested mp3.
        """
        import asyncio

        voice_id = vieneu["voice_id"]
        logger.info(f"🗣️  VieNeu preset voice '{voice_id}' (v3 Turbo)")

        from pixelle_video.services.vieneu_service import VieNeuEngine
        engine = VieNeuEngine()
        out_wav = f"{output_path}.vieneu.wav"
        try:
            # Blocking ONNX inference -> run in a thread
            await asyncio.to_thread(engine.synthesize, text, voice_id, out_wav)

            # Transcode 48kHz wav -> mp3, adjusting tempo if speed != 1.0
            import ffmpeg
            stream = ffmpeg.input(out_wav)
            if speed and abs(speed - 1.0) > 0.01:
                tempo = max(0.5, min(2.0, speed))
                stream = stream.output(
                    output_path, **{"filter:a": f"atempo={tempo}"}, loglevel="error"
                )
            else:
                stream = stream.output(output_path, loglevel="error")
            stream.overwrite_output().run()
        finally:
            try:
                if os.path.exists(out_wav):
                    os.unlink(out_wav)
            except OSError:
                pass

        logger.info(f"✅ Generated audio (VieNeu): {output_path}")
        return output_path

    async def _call_cloned_tts(
        self,
        text: str,
        custom: dict,
        rate: str,
        output_path: str,
    ) -> str:
        """
        Generate a cloned voice: Edge TTS produces the speech with a base voice,
        then OpenVoice converts the timbre to the reference clip (custom["ref"]).

        Args:
            text: Text to synthesize
            custom: Custom-voice config (keys: id, name, ref, base, ...)
            rate: Edge TTS rate string (from speed)
            output_path: Final audio path (mp3)

        Returns:
            Generated audio file path (output_path)
        """
        import asyncio

        base_voice = custom["base"]
        ref_path = custom["ref"]
        logger.info(
            f"🧬 Cloned voice '{custom['id']}': Edge base={base_voice} -> timbre={ref_path}"
        )

        # 1. Synthesize base speech with Edge TTS (good Vietnamese pronunciation)
        base_path = f"{output_path}.base.mp3"
        await edge_tts(text=text, voice=base_voice, rate=rate, output_path=base_path)

        # 2. Convert timbre to the reference clip (blocking torch -> run in thread)
        from pixelle_video.services.voice_conversion import OpenVoiceConverter
        converter = OpenVoiceConverter()
        # Lower tau preserves the Edge base's Northern pronunciation/clarity.
        tau = float(self.config.get("local", {}).get("clone_tau", 0.25))
        converted_wav = f"{output_path}.conv.wav"
        try:
            await asyncio.to_thread(
                converter.convert, base_path, base_voice, ref_path, converted_wav, tau
            )

            # 3. Transcode the converted wav to the requested mp3 output path
            import ffmpeg
            (
                ffmpeg
                .input(converted_wav)
                .output(output_path, loglevel="error")
                .overwrite_output()
                .run()
            )
        finally:
            for tmp in (base_path, converted_wav):
                try:
                    if os.path.exists(tmp):
                        os.unlink(tmp)
                except OSError:
                    pass

        logger.info(f"✅ Generated cloned audio: {output_path}")
        return output_path

    async def _call_comfyui_workflow(
        self,
        workflow_info: dict,
        text: str,
        comfyui_url: Optional[str] = None,
        runninghub_api_key: Optional[str] = None,
        voice: Optional[str] = None,
        speed: float = 1.0,
        output_path: Optional[str] = None,
        **params
    ) -> str:
        """
        Generate speech using ComfyUI workflow
        
        Args:
            workflow_info: Workflow info dict from _resolve_workflow()
            text: Text to convert to speech
            comfyui_url: ComfyUI URL
            runninghub_api_key: RunningHub API key
            voice: Voice ID (workflow-specific)
            speed: Speech speed multiplier (workflow-specific)
            output_path: Custom output path (downloads if URL returned)
            **params: Additional workflow parameters
        
        Returns:
            Generated audio file path (local if output_path provided, otherwise URL)
        """
        logger.info(f"🎙️  Using workflow: {workflow_info['key']}")
        
        # 1. Build workflow parameters (ComfyKit config is now managed by core)
        workflow_params = {"text": text}
        
        # Add optional TTS parameters (only if explicitly provided and not None)
        if voice is not None:
            workflow_params["voice"] = voice
        if speed is not None and speed != 1.0:
            workflow_params["speed"] = speed
        
        # Add any additional parameters
        workflow_params.update(params)
        
        logger.debug(f"Workflow parameters: {workflow_params}")
        
        # 3. Execute workflow using shared ComfyKit instance from core
        try:
            # Get shared ComfyKit instance (lazy initialization + config hot-reload)
            kit = await self.core._get_or_create_comfykit()
            
            # Determine what to pass to ComfyKit based on source
            if workflow_info["source"] == "runninghub" and "workflow_id" in workflow_info:
                # RunningHub: pass workflow_id
                workflow_input = workflow_info["workflow_id"]
                logger.info(f"Executing RunningHub TTS workflow: {workflow_input}")
            else:
                # Selfhost: pass file path
                workflow_input = workflow_info["path"]
                logger.info(f"Executing selfhost TTS workflow: {workflow_input}")
            
            result = await kit.execute(workflow_input, workflow_params)
            
            # 4. Handle result
            if result.status != "completed":
                error_msg = result.msg or "Unknown error"
                logger.error(f"TTS generation failed: {error_msg}")
                raise Exception(f"TTS generation failed: {error_msg}")
            
            # ComfyKit result can have audio files in different output types
            # Try to get audio file path from result
            audio_path = None
            
            # Check for audio files in result.audios (if available)
            if hasattr(result, 'audios') and result.audios:
                audio_path = result.audios[0]
                logger.debug(f"✅ Found audio in result.audios: {audio_path}")
            # Check for files in result.files
            elif hasattr(result, 'files') and result.files:
                audio_path = result.files[0]
                logger.debug(f"✅ Found audio in result.files: {audio_path}")
            # Check in outputs dictionary
            elif hasattr(result, 'outputs') and result.outputs:
                logger.debug(f"Searching for audio file in result.outputs: {result.outputs}")
                # Try to find audio file in outputs
                for key, value in result.outputs.items():
                    if isinstance(value, str) and any(value.endswith(ext) for ext in ['.mp3', '.wav', '.flac']):
                        audio_path = value
                        logger.debug(f"✅ Found audio in result.outputs[{key}]: {audio_path}")
                        break
            
            if not audio_path:
                logger.error("No audio file generated")
                logger.error(f"❌ Result analysis:")
                logger.error(f"   - result.audios: {getattr(result, 'audios', 'NOT_FOUND')}")
                logger.error(f"   - result.files: {getattr(result, 'files', 'NOT_FOUND')}")
                logger.error(f"   - result.outputs: {getattr(result, 'outputs', 'NOT_FOUND')}")
                logger.error(f"   - Full __dict__: {result.__dict__}")
                raise Exception("No audio file generated by workflow")
            
            # If output_path provided and audio_path is URL, download to local
            if output_path and audio_path.startswith(('http://', 'https://')):
                import httpx
                import os
                
                # Ensure parent directory exists
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                logger.info(f"Downloading audio from {audio_path} to {output_path}")
                async with httpx.AsyncClient() as client:
                    response = await client.get(audio_path)
                    response.raise_for_status()
                    
                    with open(output_path, 'wb') as f:
                        f.write(response.content)
                
                logger.info(f"✅ Generated audio (ComfyUI): {output_path}")
                return output_path
            
            logger.info(f"✅ Generated audio (ComfyUI): {audio_path}")
            return audio_path
        
        except Exception as e:
            logger.error(f"TTS generation error: {e}")
            raise
