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
Story Shortener Pipeline UI

A standalone text tool: paste a person's first-person life narrative and the
Claude Code subscription condenses it into a title plus a coherent 18-paragraph
summary, where each paragraph is sized to be read in ~5s by the VieNeu TTS voice
(18 × 5s ≈ a 90s video) and sanitized for natural reading. It preserves the
narrator's original voice.
"""

from typing import Any

import streamlit as st

from web.i18n import tr
from web.pipelines.base import PipelineUI, register_pipeline_ui
from web.utils.claude_cli import is_claude_available, summarize_story
from web.utils.streamlit_helpers import render_copy_button


class StoryShortenerPipelineUI(PipelineUI):
    """
    UI for the Story Shortener tool.

    Independent of the video pipelines: it does not use config_manager, the
    project LLM, or ComfyUI, so it works even when those are unconfigured.
    """
    name = "story_shortener"
    icon = "✂️"

    @property
    def display_name(self):
        return tr("pipeline.story_shortener.name")

    @property
    def description(self):
        return tr("pipeline.story_shortener.description")

    def render(self, pixelle_video: Any):
        # pixelle_video is unused — this tab is a standalone text tool.
        if not is_claude_available():
            st.warning(tr("story_shortener.no_claude"))

        left_col, right_col = st.columns([1, 1])

        # ====================================================================
        # Left Column: Original story input
        # ====================================================================
        with left_col:
            with st.container(border=True):
                narrative = st.text_area(
                    tr("story_shortener.input_label"),
                    height=380,
                    placeholder=tr("story_shortener.input_placeholder"),
                    key="story_shortener_input",
                )

                if st.button(
                    tr("story_shortener.shorten_btn"),
                    type="primary",
                    use_container_width=True,
                ):
                    if not narrative or not narrative.strip():
                        st.warning(tr("story_shortener.empty_input"))
                    else:
                        with st.spinner(tr("story_shortener.spinner")):
                            try:
                                result = summarize_story(narrative.strip())
                                st.session_state["story_shortener_result"] = result
                                # Push result into Quick Create's content input on the
                                # next run, before its widgets instantiate.
                                st.session_state["_pending_quick_create_text"] = result.summary
                                st.session_state["_pending_quick_create_title"] = result.title or ""
                            except Exception as e:
                                st.session_state.pop("story_shortener_result", None)
                                st.error(tr("story_shortener.error", error=str(e)))
                            else:
                                st.rerun()

        # ====================================================================
        # Right Column: Title + shortened 10-paragraph result (editable)
        # ====================================================================
        with right_col:
            with st.container(border=True):
                result = st.session_state.get("story_shortener_result")
                if result:
                    # AI-suggested title for the narrative.
                    if result.title:
                        st.caption(tr("story_shortener.title_label"))
                        st.markdown(f"### {result.title}")

                    # No `key` here so the box always shows the latest result
                    # while staying editable within the current run.
                    edited = st.text_area(
                        tr("story_shortener.result_label"),
                        value=result.summary,
                        height=320,
                    )
                    st.caption(
                        tr("story_shortener.word_count", count=len(edited.split()))
                    )
                    # Copy button copies the (edited) 10-paragraph summary only.
                    render_copy_button(
                        edited,
                        tr("story_shortener.copy_btn"),
                        tr("story_shortener.copied"),
                    )
                else:
                    st.caption(tr("story_shortener.result_placeholder"))


# Register self
register_pipeline_ui(StoryShortenerPipelineUI)
