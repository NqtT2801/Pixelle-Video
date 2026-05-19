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
Streamlit helper functions
"""

import json

import streamlit as st
import streamlit.components.v1 as components

from web.i18n import tr
from pixelle_video.config import config_manager


def safe_rerun():
    """Safe rerun that works with both old and new Streamlit versions"""
    if hasattr(st, 'rerun'):
        st.rerun()
    else:
        st.experimental_rerun()


# ============================================================================
# SelfHost Workflow Warning - Using Native JavaScript Alert
# ============================================================================
# Uses native browser alert() to avoid Streamlit's dialog limitations.
# This is simple, reliable, and works across all browsers.

def check_and_warn_selfhost_workflow(workflow_path: str):
    """
    Check if user just switched to a selfhost workflow and show JS alert.
    
    Uses native JavaScript alert() which bypasses all Streamlit dialog limitations.
    The alert is shown immediately when user switches to a selfhost workflow.
    
    Args:
        workflow_path: The workflow path (e.g., "selfhost/image_flux.json")
    """
    if not workflow_path:
        return
    
    # Check if this is a transition TO selfhost
    is_selfhost = workflow_path.startswith("selfhost/")
    
    # Only show alert when transitioning TO selfhost
    if is_selfhost:
        _show_js_alert(workflow_path)


def _show_js_alert(workflow_path: str):
    """
    Show a native JavaScript alert with selfhost workflow warning.
    
    Args:
        workflow_path: The workflow path to display in the alert
    """
    # Get ComfyUI URL from config
    comfyui_config = config_manager.get_comfyui_config()
    comfyui_url = comfyui_config.get("comfyui_url", "http://localhost:8188")
    
    # Build alert message
    title = tr("selfhost.warning.title")
    message = tr("selfhost.warning.message", 
                 comfyui_url=comfyui_url, 
                 workflow_path=f"workflows/{workflow_path}")
    hint = tr("selfhost.warning.hint")
    
    # Clean up markdown formatting for plain text alert
    # Remove ** (bold markers) and other markdown
    message = message.replace("**", "").replace("*", "")
    hint = hint.replace("**", "").replace("*", "")
    
    # Combine into single alert message
    full_message = f"{title}\\n\\n{message}\\n\\n{hint}"
    
    # Escape for JavaScript string
    full_message = full_message.replace("'", "\\'").replace('"', '\\"')
    full_message = full_message.replace("\n", "\\n")
    
    # Inject JavaScript alert
    js_code = f"""
    <script>
        alert("{full_message}");
    </script>
    """
    
    components.html(js_code, height=0, width=0)


# ============================================================================
# Copy to Clipboard Button - Native HTML/JS button
# ============================================================================
# Streamlit has no native copy-to-clipboard button, so we render a small HTML
# button inside a component iframe (same approach as the JS alert above).

def render_copy_button(text: str, label: str, copied_label: str, height: int = 48):
    """
    Render an HTML "copy to clipboard" button.

    Uses navigator.clipboard with a document.execCommand fallback so it works
    reliably inside Streamlit's component iframe. The button briefly shows
    `copied_label` after a successful copy.

    Args:
        text: The text placed on the clipboard when the button is clicked.
        label: The button's normal label.
        copied_label: The label shown briefly after a successful copy.
        height: Height of the component iframe, in pixels.
    """
    # JSON-encode for safe JS embedding; neutralize "</script>" breakout.
    text_js = json.dumps(text).replace("<", "\\u003c")
    label_js = json.dumps(label).replace("<", "\\u003c")
    copied_js = json.dumps(copied_label).replace("<", "\\u003c")

    html = """
    <style>
      html, body { margin: 0; padding: 0; }
      .pv-copy-btn {
        width: 100%; padding: 0.5rem 1rem; margin: 0;
        font-size: 0.95rem; font-weight: 400;
        font-family: "Source Sans Pro", sans-serif;
        border: 1px solid rgba(49, 51, 63, 0.2); border-radius: 0.5rem;
        background: #ffffff; color: rgb(49, 51, 63); cursor: pointer;
        transition: border-color 0.15s, color 0.15s;
      }
      .pv-copy-btn:hover { border-color: #ff4b4b; color: #ff4b4b; }
      .pv-copy-btn:active { background: #f0f2f6; }
    </style>
    <button class="pv-copy-btn" id="pvCopyBtn"></button>
    <script>
      (function () {
        var btn = document.getElementById("pvCopyBtn");
        var label = __LABEL__;
        var copiedLabel = __COPIED__;
        var text = __TEXT__;
        btn.textContent = label;
        function fallbackCopy(t) {
          var ta = document.createElement("textarea");
          ta.value = t;
          ta.style.position = "fixed";
          ta.style.opacity = "0";
          document.body.appendChild(ta);
          ta.focus();
          ta.select();
          var ok = false;
          try { ok = document.execCommand("copy"); } catch (e) { ok = false; }
          document.body.removeChild(ta);
          return ok;
        }
        function flash() {
          btn.textContent = copiedLabel;
          setTimeout(function () { btn.textContent = label; }, 1800);
        }
        btn.addEventListener("click", function () {
          if (navigator.clipboard && navigator.clipboard.writeText) {
            navigator.clipboard.writeText(text).then(flash, function () {
              if (fallbackCopy(text)) flash();
            });
          } else {
            if (fallbackCopy(text)) flash();
          }
        });
      })();
    </script>
    """
    # Replace __TEXT__ last so user-controlled content cannot inject placeholders.
    html = (
        html
        .replace("__LABEL__", label_js)
        .replace("__COPIED__", copied_js)
        .replace("__TEXT__", text_js)
    )
    components.html(html, height=height)

