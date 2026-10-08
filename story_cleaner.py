"""
Story Cleaner & Extractor (story_cleaner.py)
--------------------------------------------
Extracts and cleans raw narrative text from a PDF or raw book text,
removing front matter, TOC, headers, footers, appendices, and metadata.

Output: 'cleaned_story.txt' (feeds into character_profiler.py and dialogue_tagger.py)
"""

import os
import glob
import argparse
import requests
import pypdf
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

# Story extraction prompt template
story_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert story extractor and editor. Your job is to extract only the pure narrative story content from book text."),
    ("human", """Extract ONLY the main story content from this book text.

REMOVE these completely:
- Table of contents
- Foreword, preface, introduction
- Copyright, publisher info, ISBN
- Page numbers, headers, footers
- Appendix, glossary, index
- About the author, ads, reviews

KEEP only:
- The actual narrative/story from Chapter 1 to The End
- All dialogue, descriptions, and narration
- Original paragraph structure and indentation

Return ONLY the cleaned story text, nothing else.

TEXT TO CLEAN:
{raw_text}""")
])

def clean_story_text(raw_text: str, output_path: str = "cleaned_story.txt") -> str:
    """Cleans raw text using LangChain and local LM Studio."""
    print(f"[StoryCleaner] Invoking local LLM ({get_active_model()}) to clean narrative...")
    llm = get_llm()
    chain = story_prompt | llm | StrOutputParser()
    cleaned = chain.invoke({"raw_text": raw_text}).strip()

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(cleaned)

    print(f"[StoryCleaner] Saved {len(cleaned)} characters ({len(cleaned.split())} words) to '{output_path}'")
    return cleaned

def extract_story_from_pdf(pdf_path: str, output_path: str = "cleaned_story.txt") -> str:
    """Extracts raw text from PDF and passes it to the cleaning chain."""
    print(f"[StoryCleaner] Reading PDF: '{pdf_path}'...")
    raw_text = ""
    with open(pdf_path, "rb") as f:
        reader = pypdf.PdfReader(f)
        for idx, page in enumerate(reader.pages, 1):
            text = page.extract_text()
            if text:
                raw_text += text + "\n\n"

    print(f"[StoryCleaner] Extracted {len(raw_text)} raw characters across {len(reader.pages)} pages.")
    return clean_story_text(raw_text, output_path)

SAMPLE_TEXT = """
FOREWORD

This book is dedicated to all dreamers.

---

Chapter 1: The Beginning

Arjun stepped into the dark forest. The wind whispered through the trees.
"I have a bad feeling about this," he muttered.

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

---

APPENDIX

Extra notes about the world.

Afterword

Thanks for reading.
"""

def main():
    parser = argparse.ArgumentParser(description="Story Cleaner & Extractor")
    parser.add_argument("--input", "-i", type=str, help="Input PDF or text file path")
    parser.add_argument("--output", "-o", type=str, default="cleaned_story.txt", help="Output story file path (default: cleaned_story.txt)")
    parser.add_argument("--test", "-t", action="store_true", help="Run with built-in sample story text")
    args = parser.parse_args()

    print("=" * 60)
    print("  STORY CLEANER & EXTRACTOR (LangChain + LM Studio)")
    print("=" * 60)

    if args.test:
        print("[StoryCleaner] Running test mode with sample text...")
        clean_story_text(SAMPLE_TEXT, args.output)
        return

    input_file = args.input
    if not input_file:
        pdfs = glob.glob("*.pdf")
        if pdfs:
            input_file = pdfs[0]
            print(f"[StoryCleaner] Auto-detected PDF: '{input_file}'")
        else:
            print("[StoryCleaner] No input file specified and no PDF found in directory.")
            print("[StoryCleaner] Falling back to sample text. Use --input <file> to specify a file.")
            clean_story_text(SAMPLE_TEXT, args.output)
            return

    if not os.path.exists(input_file):
        print(f"[Error] File not found: '{input_file}'")
        return

    if input_file.lower().endswith(".pdf"):
        extract_story_from_pdf(input_file, args.output)
    else:
        with open(input_file, "r", encoding="utf-8") as f:
            raw = f.read()
        clean_story_text(raw, args.output)

    print(f"\n[StoryCleaner] Done! Next step: run 'python character_profiler.py' to extract character voice profiles.")

if __name__ == "__main__":
    main()
