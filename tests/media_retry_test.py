"""Tests for media-generation retry + clear RunningHub failure messages.

Covers pixelle_video/services/media.py: the per-frame retry loop that lets
transient RunningHub task failures self-heal, and the `_format_media_error`
helper that rewrites comfykit's misleading "... failed: success" into an
actionable message. See the plan in
.claude/plans/web-components-output-preview-render-sin-fluttering-gadget.md

NOTE: named *_test.py (not test_*.py) because .gitignore ignores test_*.py;
pytest's default discovery (python_files = test_*.py *_test.py) still finds it.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from comfykit.comfyui.models import ExecuteResult

from pixelle_video.config import config_manager
from pixelle_video.services.media import MediaService, _format_media_error


# --- helpers ---------------------------------------------------------------

def _make_service(execute_mock) -> MediaService:
    """Build a MediaService whose ComfyKit.execute is the given mock.

    Workflow resolution is stubbed so the test never touches the filesystem.
    """
    core = MagicMock()
    kit = MagicMock()
    kit.execute = execute_mock
    core._get_or_create_comfykit = AsyncMock(return_value=kit)

    service = MediaService(config={"comfyui": {"image": {}}}, core=core)
    service._resolve_workflow = MagicMock(
        return_value={"source": "runninghub", "workflow_id": "wf123", "path": "x.json"}
    )
    return service


def _failed_result() -> ExecuteResult:
    # Mirrors the exact shape comfykit produces for a server-side RunningHub
    # failure: the trailing "success" is the HTTP envelope msg, not the reason.
    return ExecuteResult(status="error", prompt_id="123", msg="RunningHub task 123 failed: success")


def _ok_image_result() -> ExecuteResult:
    return ExecuteResult(status="completed", prompt_id="123", images=["http://example/img.png"])


# --- retry behavior --------------------------------------------------------

async def test_media_retries_then_succeeds(monkeypatch):
    """A transient failure followed by success returns the media (no raise)."""
    monkeypatch.setattr(config_manager.config.comfyui, "media_max_retries", 2)
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())  # don't actually wait

    execute = AsyncMock(side_effect=[_failed_result(), _ok_image_result()])
    service = _make_service(execute)

    result = await service(prompt="a cat", media_type="image")

    assert result.media_type == "image"
    assert result.url == "http://example/img.png"
    assert execute.await_count == 2


async def test_media_aborts_with_clear_message(monkeypatch):
    """When every attempt fails, it raises after max_retries+1 tries with a
    clear, actionable message instead of the bare 'failed: success'."""
    monkeypatch.setattr(config_manager.config.comfyui, "media_max_retries", 2)
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())

    execute = AsyncMock(side_effect=[_failed_result() for _ in range(3)])
    service = _make_service(execute)

    with pytest.raises(Exception) as exc_info:
        await service(prompt="a cat", media_type="image")

    msg = str(exc_info.value)
    assert "RunningHub task 123 failed server-side" in msg
    assert "dashboard" in msg
    assert execute.await_count == 3  # max_retries (2) + 1


async def test_media_no_retry_when_disabled(monkeypatch):
    """media_max_retries=0 means a single attempt and no sleep."""
    monkeypatch.setattr(config_manager.config.comfyui, "media_max_retries", 0)
    sleep = AsyncMock()
    monkeypatch.setattr(asyncio, "sleep", sleep)

    execute = AsyncMock(side_effect=[_failed_result()])
    service = _make_service(execute)

    with pytest.raises(Exception):
        await service(prompt="a cat", media_type="image")

    assert execute.await_count == 1
    sleep.assert_not_awaited()


# --- message formatting ----------------------------------------------------

def test_format_media_error_runninghub():
    r = ExecuteResult(status="error", prompt_id="999", msg="RunningHub task 999 failed: success")
    out = _format_media_error(r)
    assert "RunningHub task 999 failed server-side" in out
    assert "dashboard" in out
    assert "[raw: RunningHub task 999 failed: success]" in out  # raw preserved for debugging


def test_format_media_error_without_task_id_passes_through():
    r = ExecuteResult(status="error", msg="some local ComfyUI error")
    assert _format_media_error(r) == "some local ComfyUI error"
