from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_DIR / "src"))

from ske.structure.card import build_paper_card
from ske.structure.schema import CANONICAL_SECTIONS, PaperCard
from ske.structure.verbalize import render_card_markdown


GENERIC_PHRASES = {
    "base model",
    "large model",
    "small model",
    "beam size",
    "beam search",
    "learning rate",
    "dropout rate",
    "tagging task",
    "downstream task",
    "feature-based approach",
}

VALID_SHORT_PHRASES = {
    "qa",
    "nq",
    "em",
    "f1",
    "rnn",
    "cnn",
    "gan",
    "bert",
    "bart",
    "dpr",
    "gpt",
    "bm25",
    "squad",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch paper-card inference with lightweight quality diagnostics.")
    parser.add_argument("--input_dir", default=str(PROJECT_DIR / "datasets" / "03_demo_txt" / "full_library"))
    parser.add_argument("--output_dir", default=str(PROJECT_DIR / "outputs" / "paper_cards_batch_eval"))
    parser.add_argument("--keyword_checkpoint", default=str(PROJECT_DIR / "models" / "checkpoints" / "keyword_scibert_semeval2010_finetune_nobow"))
    parser.add_argument("--structured_checkpoint", default=str(PROJECT_DIR / "models" / "checkpoints" / "structure_v2_scibert_evidencefix"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--top_k_keyphrases", type=int, default=28)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--files", nargs="*", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = collect_paths(input_dir, args.files, args.limit)
    rows: list[dict[str, Any]] = []
    for idx, path in enumerate(paths, start=1):
        print(f"[{idx}/{len(paths)}] {path.name}", flush=True)
        text = path.read_text(encoding="utf-8", errors="ignore")
        card = build_paper_card(
            text=text,
            keyword_checkpoint=args.keyword_checkpoint,
            structured_checkpoint=args.structured_checkpoint,
            title=path.stem,
            top_k_keyphrases=args.top_k_keyphrases,
            device_name=args.device,
        )
        safe_stem = safe_name(path.stem)
        (output_dir / f"{safe_stem}.json").write_text(json.dumps(card.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        (output_dir / f"{safe_stem}.md").write_text(render_card_markdown(card), encoding="utf-8")
        rows.append(analyze_card(path, card))

    write_report(output_dir, rows)
    print(json.dumps({"files": len(paths), "output_dir": str(output_dir.resolve())}, ensure_ascii=False, indent=2))


def collect_paths(input_dir: Path, requested: list[str] | None, limit: int | None) -> list[Path]:
    if requested:
        paths = [(input_dir / name) if not Path(name).is_absolute() else Path(name) for name in requested]
    else:
        paths = sorted(input_dir.glob("*.txt"))
    if limit is not None:
        paths = paths[:limit]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing input files: " + "; ".join(missing))
    return paths


def analyze_card(path: Path, card: PaperCard) -> dict[str, Any]:
    units_by_section = Counter(unit.section for unit in card.units)
    roles_by_section: dict[str, Counter[str]] = {section: Counter() for section in CANONICAL_SECTIONS}
    for unit in card.units:
        roles_by_section.setdefault(unit.section, Counter())[unit.role] += 1

    note_counts = {section: len(card.section_notes.get(section, [])) for section in CANONICAL_SECTIONS}
    empty_notes = [section for section, count in note_counts.items() if count == 0]
    no_units = [section for section in CANONICAL_SECTIONS if units_by_section.get(section, 0) == 0]
    noisy_phrases = [unit.phrase for unit in card.units if looks_noisy_phrase(unit.phrase)]
    generic_phrases = [unit.phrase for unit in card.units if unit.phrase.lower().strip() in GENERIC_PHRASES]
    long_evidence = [unit.phrase for unit in card.units if len(unit.evidence_sentence.split()) > 75]
    appendix_like = [
        unit.phrase
        for unit in card.units
        if re.search(r"\b(reference|appendix|best viewed|visualization|figure|table)\b", unit.evidence_sentence, re.I)
    ]
    coverage_score = sum(1 for section in CANONICAL_SECTIONS if units_by_section.get(section, 0) and note_counts.get(section, 0)) / len(
        CANONICAL_SECTIONS
    )
    risk_score = (
        2.0 * len(empty_notes)
        + 1.5 * len(no_units)
        + 0.5 * len(noisy_phrases)
        + 0.6 * len(generic_phrases)
        + 0.4 * len(long_evidence)
        + 0.8 * len(appendix_like)
    )
    return {
        "file": path.name,
        "units": len(card.units),
        "units_by_section": {section: units_by_section.get(section, 0) for section in CANONICAL_SECTIONS},
        "top_roles_by_section": {
            section: [role for role, _ in roles_by_section.get(section, Counter()).most_common(3)] for section in CANONICAL_SECTIONS
        },
        "note_counts": note_counts,
        "empty_note_sections": empty_notes,
        "no_unit_sections": no_units,
        "noisy_phrases": noisy_phrases[:8],
        "generic_phrases": generic_phrases[:8],
        "long_evidence_phrases": long_evidence[:8],
        "appendix_like_phrases": appendix_like[:8],
        "coverage_score": round(coverage_score, 3),
        "risk_score": round(risk_score, 3),
    }


def looks_noisy_phrase(phrase: str) -> bool:
    normalized = phrase.lower().strip()
    if not normalized:
        return True
    if normalized in VALID_SHORT_PHRASES:
        return False
    tokens = normalized.split()
    if len(tokens) == 1 and len(normalized) <= 3:
        return True
    if any(token in {"eos", "pad", "cls", "sep", "tok"} for token in tokens):
        return True
    if sum(token.isdigit() for token in tokens) >= 2:
        return True
    if re.search(r"\b[a-z]\s+\d+\b", normalized):
        return True
    return False


def write_report(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    rows = sorted(rows, key=lambda row: row["risk_score"], reverse=True)
    (output_dir / "batch_report.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Paper Card Batch Diagnostics", ""]
    lines.append("| file | risk | coverage | units | empty notes | no units | generic phrases | noisy phrases |")
    lines.append("|---|---:|---:|---:|---|---|---|---|")
    for row in rows:
        lines.append(
            "| {file} | {risk_score:.1f} | {coverage_score:.2f} | {units} | {empty} | {no_units} | {generic} | {noisy} |".format(
                file=row["file"],
                risk_score=row["risk_score"],
                coverage_score=row["coverage_score"],
                units=row["units"],
                empty=", ".join(row["empty_note_sections"]) or "-",
                no_units=", ".join(row["no_unit_sections"]) or "-",
                generic=", ".join(row["generic_phrases"]) or "-",
                noisy=", ".join(row["noisy_phrases"]) or "-",
            )
        )
    lines.append("")
    lines.append("## Section Distribution")
    lines.append("")
    for row in rows:
        lines.append(f"### {row['file']}")
        lines.append("")
        lines.append(json.dumps(row["units_by_section"], ensure_ascii=False))
        lines.append("")
    (output_dir / "batch_report.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "paper"


if __name__ == "__main__":
    main()


