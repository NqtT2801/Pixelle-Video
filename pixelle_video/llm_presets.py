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
LLM Presets - Predefined configurations for popular LLM providers

All providers support OpenAI SDK protocol.
"""

from typing import Dict, Any, List


LLM_PRESETS: List[Dict[str, Any]] = [
    {
        "name": "OpenAI",
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o",
        "api_key_url": "https://platform.openai.com/api-keys",
    },
]


def get_preset_names() -> List[str]:
    """Get list of preset names"""
    return [preset["name"] for preset in LLM_PRESETS]


def get_preset(name: str) -> Dict[str, Any]:
    """Get preset configuration by name"""
    for preset in LLM_PRESETS:
        if preset["name"] == name:
            return preset
    return {}


def find_preset_by_base_url_and_model(base_url: str, model: str) -> str | None:
    """
    Find preset name by base_url and model
    
    Returns:
        Preset name if found, None otherwise
    """
    for preset in LLM_PRESETS:
        if preset["base_url"] == base_url and preset["model"] == model:
            return preset["name"]
    return None

