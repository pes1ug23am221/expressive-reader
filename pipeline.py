"""
Audiobook Preprocessing Pipeline Runner
---------------------------------------
Orchestrates the three pipeline stages end-to-end:
  1. story_cleaner.py      -> cleaned_story.txt
  2. character_profiler.py -> characters.json
  3. dialogue_tagger.py    -> dialogue_segments_full.json & dialogue_script.txt

Directly feeds into ttts.ipynb.
"""

import os
import argparse
from story_cleaner import clean_story_text, extract_story_from_pdf, SAMPLE_TEXT
from character_profiler import extract_characters
from dialogue_tagger import process_in_chunks

def run_pipeline(source_path: str = None, is_test: bool = False, chunk_size: int = 8000):
    print("=" * 65)
    print("  AUDIOBOOK PREPROCESSING PIPELINE")
    print("  (Story Cleaner -> Character Profiler -> Dialogue Tagger)")
    print("=" * 65)

    # 1. Step 1: Story Cleaner
    if is_test or not source_path:
        print("\n>>> STEP 1: Running Story Cleaner (from Sample Text)...")
        story_text = clean_story_text(SAMPLE_TEXT, "cleaned_story.txt")
    elif source_path.lower().endswith(".pdf"):
        print(f"\n>>> STEP 1: Running Story Cleaner (Extract & Clean '{source_path}')...")
        story_text = extract_story_from_pdf(source_path, "cleaned_story.txt")
    else:
        print(f"\n>>> STEP 1: Running Story Cleaner (Clean '{source_path}')...")
        with open(source_path, "r", encoding="utf-8") as f:
            raw = f.read()
        story_text = clean_story_text(raw, "cleaned_story.txt")

    # 2. Step 2: Character Profiler
    print("\n>>> STEP 2: Running Character Profiler...")
    chars_data = extract_characters(story_text, "characters.json")

    # 3. Step 3: Dialogue Tagger
    print("\n>>> STEP 3: Running Dialogue Tagger...")
    segments_data = process_in_chunks(
        story_text,
        chars_data,
        chunk_size=chunk_size,
        output_file="dialogue_segments_full.json",
        script_file="dialogue_script.txt"
    )

    print("\n" + "=" * 65)
    print("  [SUCCESS] PIPELINE EXECUTION COMPLETE!")
    print("=" * 65)
    print("Output artifacts generated:")
    print("  1. cleaned_story.txt            -> (Story Cleaner)")
    print("  2. characters.json              -> (Character Profiler)")
    print("  3. dialogue_segments_full.json  -> (Dialogue Tagger)")
    print("  4. dialogue_script.txt          -> (Dialogue Tagger)")
    print("\nAll outputs are directly compatible with 'ttts.ipynb'!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audiobook Preprocessing Pipeline Runner")
    parser.add_argument("--input", "-i", type=str, help="Path to input PDF or text file")
    parser.add_argument("--chunk-size", type=int, default=8000, help="Chunk size for long stories (default: 8000)")
    parser.add_argument("--test", "-t", action="store_true", help="Run with built-in test sample")
    args = parser.parse_args()

    if args.test or not args.input:
        run_pipeline(is_test=True, chunk_size=args.chunk_size)
    else:
        run_pipeline(source_path=args.input, chunk_size=args.chunk_size)
