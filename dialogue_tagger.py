"""
Dialogue & Narration Tagger (dialogue_tagger.py)
-----------------------------------------------
Reads 'cleaned_story.txt' (from story_cleaner.py) and 'characters.json' (from character_profiler.py),
chunks the text, tags narration and dialogue with speakers, and generates
detailed TTS performance instructions (speaking_description).

Outputs:
  - 'dialogue_segments_full.json' (feeds directly into ttts.ipynb)
  - 'dialogue_script.txt' (human-readable script for preview)
"""

import os
import re
import json
import argparse
import requests
from typing import Dict, Any, List

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

# Strict dialogue & narration tagging prompt
tagging_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a STRICT dialogue and narration tagging system for audiobook generation.

Your output will be used by a TTS model that takes:
1. text
2. a description of how that text should be spoken (speaking_description)

So for each segment, you must produce:
- the exact text
- whether it is narration or dialogue
- who speaks it
- a speaking_description that tells the TTS model how to perform that exact segment

==================================================
MAIN GOAL
==================================================

Read the story in order and split it into segments.
Known characters in this story:
{characters}

Each segment must contain:
- index
- type: "narration" or "dialogue"
- speaker: "narrator" (for narration) or character name (for dialogue, matching known characters, else "unknown")
- speaking_description: natural performance instruction describing HOW it should sound (e.g., emotional progression, pacing, intensity, tone). Do NOT make it a one-word label like 'sad' or 'angry'.
- text: exact original text (do NOT rewrite or paraphrase)

==================================================
OUTPUT FORMAT (STRICT JSON ONLY)
==================================================

{{
  "segments": [
    {{
      "index": 1,
      "type": "narration",
      "speaker": "narrator",
      "speaking_description": "Neutral narration with steady pacing and clear delivery.",
      "text": "Exact text from story"
    }},
    {{
      "index": 2,
      "type": "dialogue",
      "speaker": "character_name",
      "speaking_description": "Starts angry and accusatory, then becomes more controlled.",
      "text": "Exact spoken text from story"
    }}
  ]
}}

Return ONLY valid JSON. No markdown fences, no explanatory text."""),
    ("human", """STORY TO TAG:
{chunk_text}""")
])

def parse_segments_json(raw_text: str) -> List[Dict[str, Any]]:
    cleaned = raw_text.strip()
    cleaned = re.sub(r"^```json\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^```\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    m = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if m:
        cleaned = m.group(0)
    data = json.loads(cleaned)
    return data.get("segments", [])

def save_and_format_segments(raw_segments: list, output_file: str = "dialogue_segments_full.json", script_file: str = "dialogue_script.txt") -> Dict[str, Any]:
    cleaned = []
    for i, seg in enumerate(raw_segments, 1):
        if not isinstance(seg, dict):
            continue
        stype = str(seg.get("type", "narration")).strip().lower()
        if stype not in ["narration", "dialogue"]:
            stype = "narration"

        spk = str(seg.get("speaker", "narrator" if stype == "narration" else "unknown")).strip().lower().replace(" ", "_")
        if stype == "narration":
            spk = "narrator"

        desc = str(seg.get("speaking_description", "")).strip()
        if not desc:
            desc = "Neutral narration with steady pacing." if stype == "narration" else "Clear delivery with natural emotion."

        txt = str(seg.get("text", "")).strip()
        if not txt:
            continue

        cleaned.append({
            "index": i,
            "type": stype,
            "speaker": spk,
            "speaking_description": desc,
            "text": txt,
            "chunk": seg.get("chunk", 1)
        })

    result = {
        "segments": cleaned,
        "total_segments": len(cleaned)
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    if script_file:
        with open(script_file, "w", encoding="utf-8") as f:
            for s in cleaned:
                if s["type"] == "narration":
                    f.write(f"[NARRATION] {s['text']}\n({s['speaking_description']})\n\n")
                else:
                    f.write(f"[{s['speaker'].upper()}] {s['text']}\n({s['speaking_description']})\n\n")

    print(f"[DialogueTagger] Saved {len(cleaned)} total segments to '{output_file}' and '{script_file}'")
    return result

def process_dialogue_and_narration(story_text: str, characters_data: dict, output_file: str = "dialogue_segments_full.json", script_file: str = "dialogue_script.txt") -> Dict[str, Any]:
    """Processes story text in a single pass."""
    print(f"[DialogueTagger] Tagging story segments with local LLM ({get_active_model()})...")
    llm = get_llm()
    chain = tagging_prompt | llm | StrOutputParser()
    chars_str = json.dumps(characters_data.get("characters", {}), indent=2)
    raw = chain.invoke({"characters": chars_str, "chunk_text": story_text})
    segs = parse_segments_json(raw)
    return save_and_format_segments(segs, output_file, script_file)

def process_in_chunks(story_text: str, characters_data: dict, chunk_size: int = 8000, overlap: int = 500, output_file: str = "dialogue_segments_full.json", script_file: str = "dialogue_script.txt") -> Dict[str, Any]:
    """Processes long story texts using chunks with overlap."""
    if len(story_text) <= chunk_size:
        return process_dialogue_and_narration(story_text, characters_data, output_file, script_file)

    print(f"[DialogueTagger] Processing long story ({len(story_text)} chars) in chunks of {chunk_size} chars...")
    llm = get_llm()
    chain = tagging_prompt | llm | StrOutputParser()
    chars_str = json.dumps(characters_data.get("characters", {}), indent=2)

    all_segments = []
    start = 0
    chunk_num = 1
    while start < len(story_text):
        end = min(start + chunk_size, len(story_text))
        chunk = story_text[start:end]
        print(f"[DialogueTagger] Processing Chunk {chunk_num} ({len(chunk)} chars)...")
        try:
            raw = chain.invoke({"characters": chars_str, "chunk_text": chunk})
            segs = parse_segments_json(raw)
            for s in segs:
                s["chunk"] = chunk_num
                all_segments.append(s)
            print(f"[DialogueTagger] Chunk {chunk_num}: Extracted {len(segs)} segments")
        except Exception as e:
            print(f"[DialogueTagger] Error in Chunk {chunk_num}: {e}")

        if end >= len(story_text):
            break
        start += chunk_size - overlap
        chunk_num += 1

    return save_and_format_segments(all_segments, output_file, script_file)

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

SAMPLE_CHARS = {
    "characters": {
        "narrator": {
            "name": "Narrator",
            "description": "Neutral voice, adult, clear and steady tone, medium pace, balanced narration style",
            "ref_text": "Arjun stepped into the dark forest. The wind whispered through the trees."
        },
        "arjun": {
            "name": "Arjun",
            "description": "Young adult male, firm and determined tone, moderate pace",
            "ref_text": "I didn't come this far to run."
        },
        "woman": {
            "name": "Woman",
            "description": "Adult female, calm and authoritative tone, slow deliberate pace",
            "ref_text": "You should not have come here."
        }
    },
    "total_characters": 3
}

def main():
    parser = argparse.ArgumentParser(description="Dialogue & Narration Tagger")
    parser.add_argument("--story", "-s", type=str, default="cleaned_story.txt", help="Input story file path (default: cleaned_story.txt)")
    parser.add_argument("--characters", "-c", type=str, default="characters.json", help="Input characters JSON path (default: characters.json)")
    parser.add_argument("--output", "-o", type=str, default="dialogue_segments_full.json", help="Output segments JSON path (default: dialogue_segments_full.json)")
    parser.add_argument("--script", type=str, default="dialogue_script.txt", help="Output text script path (default: dialogue_script.txt)")
    parser.add_argument("--chunk-size", type=int, default=8000, help="Chunk size for long stories (default: 8000)")
    parser.add_argument("--test", "-t", action="store_true", help="Run with built-in test sample")
    args = parser.parse_args()

    print("=" * 60)
    print("  DIALOGUE & NARRATION TAGGER (LangChain + LM Studio)")
    print("=" * 60)

    if args.test:
        print("[DialogueTagger] Running test mode with sample story and characters...")
        res = process_in_chunks(SAMPLE_STORY, SAMPLE_CHARS, chunk_size=args.chunk_size, output_file=args.output, script_file=args.script)
    else:
        if not os.path.exists(args.story):
            print(f"[DialogueTagger] Story file '{args.story}' not found. Run 'python story_cleaner.py' first or use --test.")
            return

        if not os.path.exists(args.characters):
            print(f"[DialogueTagger] Characters file '{args.characters}' not found. Run 'python character_profiler.py' first or use --test.")
            return

        with open(args.story, "r", encoding="utf-8") as f:
            story = f.read()

        with open(args.characters, "r", encoding="utf-8") as f:
            chars_data = json.load(f)

        print(f"[DialogueTagger] Loaded story ({len(story)} chars) and {len(chars_data.get('characters', {}))} characters.")
        res = process_in_chunks(story, chars_data, chunk_size=args.chunk_size, output_file=args.output, script_file=args.script)

    print("\nPreview of First 3 Segments:")
    print("-" * 50)
    for seg in res["segments"][:3]:
        print(f"[{seg['index']}] {seg['type'].upper()} ({seg['speaker']})")
        print(f"  Style: {seg['speaking_description']}")
        print(f"  Text:  {seg['text'][:80]}...\n")

    print(f"[DialogueTagger] Done! All outputs are ready for 'ttts.ipynb'.")

if __name__ == "__main__":
    main()
