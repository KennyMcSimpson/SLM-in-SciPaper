from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_DIR / "src"))

from ske.structure.schema import CANONICAL_SECTIONS
from ske.structure.sectioning import section_document


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect local txt section recovery.")
    parser.add_argument("files", nargs="+", help="Txt files or directories containing txt files.")
    parser.add_argument("--summary", action="store_true", help="Print only batch-level section coverage statistics.")
    parser.add_argument("--worst", type=int, default=12, help="How many low-coverage files to show in summary mode.")
    return parser.parse_args()


def expand_paths(items: list[str]) -> list[Path]:
    paths: list[Path] = []
    for item in items:
        path = Path(item)
        if path.is_dir():
            paths.extend(sorted(path.glob("*.txt")))
        else:
            paths.append(path)
    return paths


def count_sections(path: Path) -> tuple[int, Counter[str]]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    sentences = section_document(text)
    return len(sentences), Counter(sentence.section for sentence in sentences)


def print_summary(paths: list[Path], worst: int) -> None:
    present_counter: Counter[int] = Counter()
    missing_counter: Counter[str] = Counter()
    low_coverage: list[tuple[int, int, str, dict[str, int]]] = []
    total_sentences = 0
    for path in paths:
        sentence_count, counts = count_sections(path)
        total_sentences += sentence_count
        section_counts = {section: counts.get(section, 0) for section in CANONICAL_SECTIONS}
        present = sum(1 for count in section_counts.values() if count > 0)
        present_counter[present] += 1
        for section, count in section_counts.items():
            if count == 0:
                missing_counter[section] += 1
        if present <= 2 or section_counts.get("method", 0) == 0 or section_counts.get("experiment", 0) == 0:
            low_coverage.append((present, sentence_count, path.name, section_counts))

    print("files", len(paths))
    print("sentences", total_sentences)
    print("present_sections", dict(sorted(present_counter.items())))
    print("missing", {section: missing_counter.get(section, 0) for section in CANONICAL_SECTIONS})
    print("low_coverage", len(low_coverage))
    for present, sentence_count, name, section_counts in sorted(low_coverage)[:worst]:
        print(f"- present={present} sentences={sentence_count} {name} {section_counts}")


def main() -> None:
    args = parse_args()
    paths = expand_paths(args.files)
    if args.summary:
        print_summary(paths, args.worst)
        return
    for path in paths:
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

