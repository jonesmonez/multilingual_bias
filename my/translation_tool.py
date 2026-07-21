#!/usr/bin/env python3
"""
translate_and_filter_groups.py

Interactively review bias‑attribute word groups, translate each word,
and keep or discard whole groups.

Requirements:
    pip install googletrans==4.0.0rc1
"""

import json
import sys
from pathlib import Path

try:
    from googletrans import Translator
except ImportError:
    print(
        "Error: googletrans is not installed. Install it with:\n"
        "    pip install googletrans==4.0.0rc1"
    )
    sys.exit(1)

DATA_DIR = Path("../data")
TRANSLATOR = Translator()


def _src_lang_from_code(code: str) -> str:
    """Extract the language part before the first '_' (e.g. 'ca_ES' -> 'ca')."""
    return code.split("_")[0]


def load_data(lang: str):
    """
    Load the JSON file for a given language.
    Returns a dict like:
        {
            "gender": [["ell", "ella"], ["dones", "homes"], ...],
            "race-color": [...],
            ...
        }
    """
    path = DATA_DIR / f"bias_attribute_own_{lang}.json"
    if not path.is_file():
        print(f"⚠️  {path} not found.")
        return {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    # Ensure every value is a list of lists of strings
    cleaned = {}
    for key, val in data.items():
        if isinstance(val, list):
            # Keep only inner lists that contain strings
            groups = []
            for item in val:
                if isinstance(item, list):
                    strs = [s for s in item if isinstance(s, str) and s.strip()]
                    if strs:
                        groups.append(strs)
                elif isinstance(item, str):
                    # stray string – treat as a single‑item group
                    if item.strip():
                        groups.append([item.strip()])
            if groups:
                cleaned[key] = groups
        else:
            print(f"⚠️  Unexpected top‑level value for key '{key}' in {path}")
    return cleaned


def save_filtered(lang: str, filtered_dict):
    """Write the filtered dict to <lang>_filtered.json preserving structure."""
    out_path = DATA_DIR / "filtered" / f"bias_attribute_own_{lang}.json"
    if len(filtered_dict) == 0:
        print(f"No entries to save, skipping file")
        return
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(filtered_dict, f, ensure_ascii=False, indent=2)
    print(f"💾 Saved filtered file to {out_path}")


def translate_word(word: str, src: str, dest: str) -> str:
    """Translate a single word; return the translation or an error message."""
    if not isinstance(word, str):
        return "[not a string]"
    try:
        translation = TRANSLATOR.translate(word, src=src, dest=dest)
        return translation.text
    except Exception as exc:
        return f"[translation error: {exc}]"


def main():
    # Discover language files
    pattern = "bias_attribute_own_*.json"
    files = list(DATA_DIR.glob(pattern))
    if not files:
        print(f"❌ No files matching '{pattern}' found in {DATA_DIR}")
        return

    # Extract language codes from filenames
    langs = []
    for f in files:
        stem = f.stem  # e.g., bias_attribute_own_es
        if stem.startswith("bias_attribute_own_"):
            langs.append(stem[len("bias_attribute_own_") :])
        else:
            print(f"⚠️  Skipping unexpected file: {f.name}")

    if not langs:
        print("❌ No valid language files detected.")
        return

    print(f"🔎 Found languages: {', '.join(langs)}")
    target_lang = input("Enter target language for translation (default: en): ").strip()
    translate_all_words = target_lang[:2] == "xx"
    print("Translate all:", translate_all_words)
    if target_lang[:2] == "xx":
        target_lang = target_lang[2:4]
    if not target_lang:
        target_lang = "en"
    
    print("Target lang:", target_lang)

    # Process each language file
    for lang in langs:
        print("\n" + "=" * 60)
        print(f"🔤 Processing language: {lang}")
        src_lang = _src_lang_from_code(lang)
        if target_lang == src_lang:
            continue
        data = load_data(lang)
        if not data:
            print(f"📭 No data found for {lang}.")
            continue

        filtered = {}  # will hold only the kept groups

        for attr_type, groups in data.items():
            print(f"\n--- Attribute type: {attr_type} ---")
            kept_groups = []

            for idx, group in enumerate(groups, start=1):
                if not translate_all_words:
                    print(f"\n[{idx}/{len(groups)}] Group (type: {attr_type}):")
                translations = []
                for word in group:
                    translation = translate_word(word, src=src_lang, dest=target_lang)
                    translations.append(translation)
                    print(f"  • {word}  →  {translation}")

                if not translate_all_words:
                    while True:
                        choice = input("Keep (k), Delete (d), Skip ([s])? ").strip().lower()
                        if choice in ("k", "kept", "y", "yes"):
                            kept_groups.append(group)
                            break
                        if choice in ("d", "delete", "n", "no"):
                            break
                        if choice in ("s", "skip", ""):
                            break
                        print("Please enter 'k' to keep, 'd' to delete, or 's' to skip.")

            if kept_groups:
                filtered[attr_type] = kept_groups

        # Write the filtered result (empty dict means nothing kept → empty file)
        save_filtered(lang, filtered)


if __name__ == "__main__":
    main()