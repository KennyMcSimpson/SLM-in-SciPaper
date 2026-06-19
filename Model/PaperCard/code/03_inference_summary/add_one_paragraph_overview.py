from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_DIR / "src"))

from ske.structure.schema import CANONICAL_SECTIONS
from ske.structure.verbalize import SECTION_TITLES, build_one_paragraph_overview


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add a one-paragraph overview to an existing paper-card Markdown file.")
    parser.add_argument("input_md", help="Existing paper-card Markdown file.")
    parser.add_argument("--output_md", default=None, help="Optional output Markdown path. If omitted, prints the overview only.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_md)
    markdown = input_path.read_text(encoding="utf-8")
    summary = extract_five_part_summary(markdown)
    overview = build_one_paragraph_overview(summary)

    if args.output_md:
        output_path = Path(args.output_md)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(insert_or_replace_overview(markdown, overview), encoding="utf-8")
        print(str(output_path.resolve()))
    else:
        print(overview)


def extract_five_part_summary(markdown: str) -> dict[str, str]:
    marker = "## Five-part Structured Summary"
    if marker not in markdown:
        raise ValueError(f"Cannot find section: {marker}")
    section_text = markdown.split(marker, 1)[1]
    summary: dict[str, str] = {}
    for idx, section in enumerate(CANONICAL_SECTIONS):
        title = SECTION_TITLES[section]
        start_marker = f"### {title}"
        start = section_text.find(start_marker)
        if start < 0:
            summary[section] = ""
            continue
        start += len(start_marker)
        next_starts = [
            section_text.find(f"### {SECTION_TITLES[next_section]}", start)
            for next_section in CANONICAL_SECTIONS[idx + 1 :]
        ]
        next_starts = [value for value in next_starts if value >= 0]
        end = min(next_starts) if next_starts else len(section_text)
        summary[section] = section_text[start:end].strip()
    return summary


def insert_or_replace_overview(markdown: str, overview: str) -> str:
    block = f"## One-paragraph Paper Overview\n\n{overview}\n\n"
    marker = "## One-paragraph Paper Overview"
    next_marker = "## Five-part Structured Summary"
    if marker in markdown:
        start = markdown.find(marker)
        next_start = markdown.find(next_marker, start)
        if next_start < 0:
            return markdown[:start].rstrip() + "\n\n" + block.rstrip() + "\n"
        return markdown[:start].rstrip() + "\n\n" + block + markdown[next_start:].lstrip()
    if next_marker not in markdown:
        return markdown.rstrip() + "\n\n" + block.rstrip() + "\n"
    start = markdown.find(next_marker)
    return markdown[:start].rstrip() + "\n\n" + block + markdown[start:].lstrip()


if __name__ == "__main__":
    main()

