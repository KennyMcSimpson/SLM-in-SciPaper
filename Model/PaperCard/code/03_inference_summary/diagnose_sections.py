from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_DIR / "src"))

from ske.structure.schema import CANONICAL_SECTIONS
from ske.structure.sectioning import section_document


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect local txt section recovery.")
    parser.add_argument("files", nargs="+")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for file_name in args.files:
        path = Path(file_name)
        text = path.read_text(encoding="utf-8", errors="ignore")
        sentences = section_document(text)
        counts = Counter(sentence.section for sentence in sentences)
        print(f"# {path.name}")
        print({section: counts.get(section, 0) for section in CANONICAL_SECTIONS})
        for section in CANONICAL_SECTIONS:
            section_sentences = [sentence for sentence in sentences if sentence.section == section]
            examples = section_sentences[:2]
            if not examples:
                continue
            print(f"## {section}")
            for sentence in examples:
                words = sentence.text.split()
                print(f"- [{sentence.sentence_index}] {' '.join(words[:42])}")
            if len(section_sentences) > 2:
                print("  tail:")
                for sentence in section_sentences[-2:]:
                    words = sentence.text.split()
                    print(f"  - [{sentence.sentence_index}] {' '.join(words[:42])}")
        print()


if __name__ == "__main__":
    main()

