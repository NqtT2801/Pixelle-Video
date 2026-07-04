"""Tests for the TTS edge-silence trim (pixelle_video/services/tts_service.py).

Trimming the leading/trailing dead air from each per-paragraph voice clip is what
lets the concatenated segments play as one continuous narration (no focus-breaking
pause between segments). See the plan in
.claude/plans/b-y-gi-nh-vi-c-steady-sunset.md

Covered here (pure logic — no ffmpeg or audio files needed):
- _build_trim_filter: the ffmpeg filter string (lead/trail pad, the areverse
  sandwich, threshold, and ms clamping).
- _trim_silence_if_enabled: config gating + the knobs it forwards to the trim.

NOTE: named *_test.py (not test_*.py) because .gitignore ignores test_*.py;
pytest's default discovery (python_files = test_*.py *_test.py) still finds it.
"""

from unittest.mock import MagicMock

from pixelle_video.services.tts_service import TTSService


def _bare_service(config: dict) -> TTSService:
    """A TTSService with only ``.config`` set, skipping ComfyBaseService.__init__
    (no workflow scan / filesystem). The trim helpers just read ``self.config`` and
    call each other, so a bare instance is enough.
    """
    svc = TTSService.__new__(TTSService)
    svc.config = config
    return svc


# --- _build_trim_filter ----------------------------------------------------

def test_build_trim_filter_structure():
    """Head trim, areverse, tail trim, areverse back — with the expected pads."""
    f = TTSService._build_trim_filter(-45, 50, 120)
    assert f == (
        "silenceremove=start_periods=1:start_silence=0.050"
        ":start_threshold=-45dB:detection=peak,"
        "areverse,"
        "silenceremove=start_periods=1:start_silence=0.120"
        ":start_threshold=-45dB:detection=peak,"
        "areverse"
    )


def test_build_trim_filter_threshold_passthrough():
    """Threshold is forwarded; two silenceremove stages sit in an areverse sandwich."""
    f = TTSService._build_trim_filter(-30, 0, 0)
    assert "start_threshold=-30dB" in f
    assert f.count("silenceremove") == 2
    assert f.count("areverse") == 2
    assert "start_silence=0.000" in f


def test_build_trim_filter_clamps_negative_ms():
    """Negative pads clamp to 0 (never a malformed/negative start_silence)."""
    f = TTSService._build_trim_filter(-45, -10, -1)
    assert "start_silence=0.000" in f
    assert "-0.0" not in f


# --- _trim_silence_if_enabled ----------------------------------------------

async def test_trim_disabled_returns_path_untouched():
    """trim_silence: false -> original path returned, ffmpeg never invoked."""
    svc = _bare_service({"local": {"trim_silence": False}})
    svc._trim_edge_silence = MagicMock()
    out = await svc._trim_silence_if_enabled("a.mp3")
    assert out == "a.mp3"
    svc._trim_edge_silence.assert_not_called()


async def test_trim_enabled_by_default_forwards_defaults():
    """Absent config -> trim runs with the documented defaults (-45 / 50 / 120)."""
    svc = _bare_service({"local": {}})
    svc._trim_edge_silence = MagicMock(return_value="a.mp3")
    out = await svc._trim_silence_if_enabled("a.mp3")
    assert out == "a.mp3"
    svc._trim_edge_silence.assert_called_once_with(
        "a.mp3", threshold_db=-45, lead_ms=50, trail_ms=120
    )


async def test_trim_forwards_config_overrides():
    """Config knobs flow through to _trim_edge_silence."""
    svc = _bare_service({"local": {
        "trim_silence": True,
        "trim_silence_threshold_db": -38,
        "trim_lead_ms": 30,
        "trim_trail_ms": 200,
    }})
    svc._trim_edge_silence = MagicMock(return_value="b.mp3")
    await svc._trim_silence_if_enabled("b.mp3")
    svc._trim_edge_silence.assert_called_once_with(
        "b.mp3", threshold_db=-38, lead_ms=30, trail_ms=200
    )
