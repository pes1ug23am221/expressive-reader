"""
Character Voice Profiler (character_profiler.py)
------------------------------------------------
Reads 'cleaned_story.txt' (from story_cleaner.py) and generates consistent,
non-hallucinated character voice profiles with TTS performance descriptions
and exact reference dialogue quotes.

Output: 'characters.json' (feeds into dialogue_tagger.py and ttts.ipynb)
"""

import os
import re
import json
import argparse
import requests
from typing import Dict, Any

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

LM_STUDIO_URL = "http://localhost:1234/v1"

def get_active_model(default: str = "qwen/qwen3-4b") -> str:
    """Detects active text generation model in LM Studio."""
    try:
        res = requests.get(f"{LM_STUDIO_URL}/models", timeout=3).json()
        for m in res.get("data", []):
            mid = m.get("id", "")
            if "embed" not in mid.lower():
                return mid
    except Exception:
        pass
    return default

def get_llm(temperature: float = 0.2) -> ChatOpenAI:
    model_name = get_active_model()
    return ChatOpenAI(
        base_url=LM_STUDIO_URL,
        api_key="lm-studio",
        model=model_name,
        temperature=temperature,
    )

# Strict character profiling prompt
character_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a STRICT character and voice profiling system for audiobook generation.

Your role is to read the FULL story and extract ALL relevant speaking characters,
then generate CONSISTENT, NON-HALLUCINATED voice embeddings suitable for TTS systems.

This output will be directly used for voice synthesis. Accuracy and consistency are critical.

==================================================
PRIMARY OBJECTIVE
==================================================

For each important character:
1. Identify the character
2. Understand how they GENERALLY speak across the story
3. Generate a stable voice description (NOT situation-based)
4. Provide one real dialogue line as reference

Also include a NARRATOR entry.

==================================================
CHARACTER SELECTION RULES (STRICT)
==================================================

Include ONLY characters that meet at least ONE:
1. Have dialogue in the story
2. Appear multiple times
3. Influence the story progression

Exclude:
- unnamed background characters
- one-time mentions without importance
- symbolic or abstract entities

==================================================
CORE EXTRACTION RULES
==================================================

1. ZERO HALLUCINATION
- DO NOT invent traits
- DO NOT invent age, accent, pitch, tone, personality, or speaking style
- ONLY use information inferable from:
  - dialogue
  - repeated behavior
  - explicit descriptions

If uncertain -> OMIT that detail

--------------------------------------------------

2. GENERAL VOICE ONLY (CRITICAL)
- Describe how the character speaks OVERALL across the story
- DO NOT include temporary or situation-based delivery such as:
  - whispering to stay hidden
  - shouting during battle
  - crying in one emotional scene

Instead capture the stable baseline voice:
- likely age category if inferable
- likely gender if clear
- pitch tendency
- pace tendency
- overall tone
- speaking style
- accent ONLY if clearly supported by the text

--------------------------------------------------

3. AGE RULE
- Use GENERAL categories ONLY:
  "young boy", "young girl", "teenage boy", "teenage girl",
  "young adult male", "young adult female", "adult male", "adult female",
  "middle-aged man", "middle-aged woman", "elderly man", "elderly woman"
- NEVER use exact ages like "19 years old", "around 45", "70+"

--------------------------------------------------

4. DESCRIPTION RULE
Each character description should be a concise TTS-oriented voice embedding.
Include ONLY supported traits: age category, gender if clear, pitch (high/medium/deep), pace (slow/medium/fast), tone/personality, speaking style, accent ONLY if clearly indicated.

Do NOT include scene-specific delivery or poetic phrasing.

--------------------------------------------------

5. REFERENCE TEXT (MANDATORY)
- Must be an EXACT line spoken by the character
- Extract directly from the story
- Do NOT paraphrase, do NOT generate new dialogue

--------------------------------------------------

6. NARRATOR RULE (IMPORTANT)
- ALWAYS include a "narrator" entry
- Neutral, clear storytelling voice used for non-dialogue text
- narrator ref_text must be a real narration line from the story, NOT dialogue

==================================================
OUTPUT FORMAT (STRICT JSON ONLY)
==================================================

{{
  "characters": {{
    "narrator": {{
      "name": "Narrator",
      "description": "Neutral voice, adult, clear and steady tone, medium pace, balanced narration style",
      "ref_text": "Exact narration line from the story"
    }},
    "character_id": {{
      "name": "Character name exactly as in text",
      "description": "TTS-oriented general voice embedding based only on supported evidence",
      "ref_text": "Exact dialogue line from the story"
    }}
  }}
}}

Return ONLY the raw JSON object, without markdown quotes or explanation."""),
    ("human", """STORY TO ANALYZE:
{story_text}""")
])

def extract_characters(story_text: str, output_file: str = "characters.json") -> Dict[str, Any]:
    """Analyzes story and extracts character voice profiles via LangChain."""
    print(f"[CharacterProfiler] Profiling characters using local LLM ({get_active_model()})...")
    llm = get_llm()
    chain = character_prompt | llm | StrOutputParser()
    raw = chain.invoke({"story_text": story_text}).strip()

    cleaned = re.sub(r"^```json\s*", "", raw, flags=re.IGNORECASE)
    cleaned = re.sub(r"^```\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    m = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if m:
        cleaned = m.group(0)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse LLM output as JSON:\n{raw}") from e

    if "characters" not in data or not isinstance(data["characters"], dict):
        raise ValueError("JSON does not contain a valid 'characters' dictionary.")

    chars = {}
    for cid, cinfo in data["characters"].items():
        if not isinstance(cinfo, dict):
            continue
        nid = str(cid).strip().lower().replace(" ", "_")
        name = str(cinfo.get("name", "")).strip() or nid.replace("_", " ").title()
        desc = str(cinfo.get("description", "")).strip() or "Neutral voice, adult, clear tone, medium pace"
        ref = str(cinfo.get("ref_text", "")).strip()
        chars[nid] = {
            "name": name,
            "description": desc,
            "ref_text": ref
        }

    if "narrator" not in chars:
        first_line = story_text.split(".")[0].strip() + "."
        chars["narrator"] = {
            "name": "Narrator",
            "description": "Neutral voice, adult, clear and steady tone, medium pace, balanced narration style",
            "ref_text": first_line
        }

    result = {
        "characters": chars,
        "total_characters": len(chars)
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"[CharacterProfiler] Extracted {result['total_characters']} character profiles -> saved to '{output_file}'")
    return result

SAMPLE_STORY = """
Arjun stepped into the dark forest. The wind whispered through the trees.

"I have a bad feeling about this," Arjun said.

A shadow moved behind him.

"Who's there?" he shouted.

From somewhere deeper in the forest, a calm voice replied,
"You should not have come here."

Arjun tightened his grip on the lantern and looked around.
The trees seemed to lean closer as the silence thickened.

A woman emerged from the mist, her eyes fixed on him.
"If you value your life, turn back now," she said.

But Arjun stood his ground.
"I didn't come this far to run."
"""

def main():
    parser = argparse.ArgumentParser(description="Character Voice Profiler")
    parser.add_argument("--input", "-i", type=str, default="cleaned_story.txt", help="Input story file path (default: cleaned_story.txt)")
    parser.add_argument("--output", "-o", type=str, default="characters.json", help="Output characters JSON path (default: characters.json)")
    parser.add_argument("--test", "-t", action="store_true", help="Run with built-in test sample")
    args = parser.parse_args()

    print("=" * 60)
    print("  CHARACTER VOICE PROFILER (LangChain + LM Studio)")
    print("=" * 60)

    if args.test:
        print("[CharacterProfiler] Running test mode with sample story...")
        res = extract_characters(SAMPLE_STORY, args.output)
    else:
        if not os.path.exists(args.input):
            print(f"[CharacterProfiler] '{args.input}' not found. Run 'python story_cleaner.py' first or use --test.")
            return

        with open(args.input, "r", encoding="utf-8") as f:
            story = f.read()

        print(f"[CharacterProfiler] Loaded '{args.input}' ({len(story)} characters).")
        res = extract_characters(story, args.output)

    print("\nCharacter Voice Profiles:")
    print("-" * 50)
    for cid, cinfo in res["characters"].items():
        print(f"[{cid}] {cinfo['name']}")
        print(f"  Description: {cinfo['description']}")
        print(f"  Ref Quote:   {cinfo['ref_text']}\n")

    print(f"[CharacterProfiler] Done! Next step: run 'python dialogue_tagger.py' to tag dialogue and narration.")

if __name__ == "__main__":
    main()
