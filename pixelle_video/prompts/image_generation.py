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
Image prompt generation template

For generating image prompts from narrations.
"""

import json
from typing import List, Optional


# ==================== PRESET IMAGE STYLES ====================
# Predefined visual styles for different use cases

IMAGE_STYLE_PRESETS = {
    "stick_figure": {
        "name": "Stick Figure Sketch",
        "description": "stick figure style sketch, black and white lines, pure white background, minimalist hand-drawn feel",
        "use_case": "General scenes, simple and intuitive"
    },
    
    "minimal": {
        "name": "Minimalist Abstract",
        "description": "minimalist abstract art, geometric shapes, clean composition, modern design, soft pastel colors",
        "use_case": "Modern, artistic feel"
    },
    
    "concept": {
        "name": "Conceptual Visual",
        "description": "conceptual visual metaphors, symbolic elements, thought-provoking imagery, artistic interpretation",
        "use_case": "Deep content, philosophical thinking"
    },
}

# Default preset
DEFAULT_IMAGE_STYLE = "stick_figure"


IMAGE_PROMPT_GENERATION_PROMPT = """# Role Definition
You are a professional visual creative designer, skilled at creating expressive and symbolic image prompts for video scripts, transforming abstract concepts into concrete visual scenes.

# Core Task
Based on the existing video script, create corresponding **English** image prompts for each storyboard's "narration content", ensuring visual scenes perfectly match the narrative content and enhance audience understanding and memory.

**Important: The input contains {narrations_count} narrations. You must generate one corresponding image prompt for each narration, totaling {narrations_count} image prompts.**

# Input Content
{narrations_json}

# Output Requirements

## Image Prompt Specifications
- Language: **Must use English** (for AI image generation models)
- Description structure: scene + one clear main subject + character action + facial expression + emotion + simple setting
- Description length: Ensure clear, complete, and creative descriptions (recommended 50-100 English words)

## Visual Creative Requirements
- Each image must accurately reflect the specific content and emotion of the corresponding narration
- Depict the literal, concrete subject of the narration: show clear characters with readable facial expressions, body language, and a simple, real-world setting
- Any symbolism must stay subtle and environmental (lighting, color, weather, posture, or a single small flat icon) and must be physically plausible — never an object that connects or tethers people together, and never a disembodied object floating in mid-air
- Scenes should express rich emotions through the characters themselves to enhance visual impact
- Prefer grounded, physically-plausible compositions; the main subject of the narration must be clearly recognizable at a glance

## Avoid (these render badly and break the clean cartoon style)
- No thin threads, strings, wires, ropes, or cords connecting, tethering, or linking people or body parts (image models render these as tangled string or as a person walking a tightrope)
- No tightrope, wire-walking, or balancing-on-a-line imagery unless the narration is literally about that
- No disembodied body parts or symbolic objects (hearts, rings, padlocks) floating or suspended in the air; if a symbol is truly needed, draw it as a small flat sticker/icon resting within the scene
- Keep one clear focal subject — avoid crowded, surreal, multi-element compositions that fight the cute minimalist cartoon style
- Do not render text, letters, words, or numbers inside the image

## Key English Vocabulary Reference
- Symbolic elements: symbolic elements
- Expression: expression / facial expression
- Action: action / gesture / movement
- Scene: scene / setting
- Atmosphere: atmosphere / mood

## Visual and Copy Coordination Principles
- Images should serve the copy, becoming a visual extension of the copy content
- Avoid visual elements unrelated to or contradicting the copy content
- Choose visual presentation methods that best enhance the persuasiveness of the copy
- Ensure the audience can quickly understand the core viewpoint of the copy through images

## Creative Guidance
1. **Phenomenon Description Copy**: Use intuitive scenes to represent social phenomena
2. **Cause Analysis Copy**: Use visual metaphors of cause-and-effect relationships to represent internal logic
3. **Impact Argumentation Copy**: Use consequence scenes or contrast techniques to represent the degree of impact
4. **In-depth Discussion Copy**: Use concretization of abstract concepts to represent deep thinking
5. **Conclusion Inspiration Copy**: Use open-ended scenes or guiding elements to represent inspiration

# Output Format
Strictly output in the following JSON format, **image prompts must be in English**:

```json
{{
  "image_prompts": [
    "[detailed English image prompt following the style requirements]",
    "[detailed English image prompt following the style requirements]"
  ]
}}
```

# Important Reminders
1. Only output JSON format content, do not add any explanations
2. Ensure JSON format is strictly correct and can be directly parsed by the program
3. Input is {{"narrations": [narration array]}} format, output is {{"image_prompts": [image prompt array]}} format
4. **The output image_prompts array must contain exactly {narrations_count} elements, corresponding one-to-one with the input narrations array**
5. **Image prompts must use English** (for AI image generation models)
6. Image prompts must accurately reflect the specific content and emotion of the corresponding narration
7. Each image must be creative and visually impactful, avoid being monotonous
8. Ensure visual scenes can enhance the persuasiveness of the copy and audience understanding

Now, please create {narrations_count} corresponding **English** image prompts for the above {narrations_count} narrations. Only output JSON, no other content.
"""


def build_image_prompt_prompt(
    narrations: List[str],
    min_words: int,
    max_words: int
) -> str:
    """
    Build image prompt generation prompt
    
    Note: Style/prefix will be applied later via prompt_prefix in config.
    
    Args:
        narrations: List of narrations
        min_words: Minimum word count
        max_words: Maximum word count
    
    Returns:
        Formatted prompt for LLM
    
    Example:
        >>> build_image_prompt_prompt(narrations, 50, 100)
    """
    narrations_json = json.dumps(
        {"narrations": narrations},
        ensure_ascii=False,
        indent=2
    )
    
    return IMAGE_PROMPT_GENERATION_PROMPT.format(
        narrations_json=narrations_json,
        narrations_count=len(narrations),
        min_words=min_words,
        max_words=max_words
    )

