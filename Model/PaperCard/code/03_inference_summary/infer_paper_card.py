from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_DIR / "src"))

from ske.structure.card import build_paper_card
from ske.structure.evidence_units import build_evidence_units_payload
from ske.structure.verbalize import render_card_markdown


def parse_args() -> argparse.Namespace:
    lexicon_dir = PROJECT_DIR / "datasets" / "lexicon"
    default_section_bow = lexicon_dir / "section_bow_vocabulary.csv"
    default_frequency_matrix = lexicon_dir / "section_document_term_matrix_frequency.csv"
    default_tfidf_matrix = lexicon_dir / "section_document_term_matrix_tfidf.csv"
    default_evidence_cues = lexicon_dir / "evidence_cue_lexicon.csv"
    default_sentence_evidence = lexicon_dir / "sentence_evidence_candidates.csv"
    parser = argparse.ArgumentParser(description="Build evidence-grounded concept units JSON from a txt paper.")
    parser.add_argument("--input_txt", required=True)
    parser.add_argument("--keyword_checkpoint", default=str(PROJECT_DIR / "models" / "checkpoints" / "keyword_scibert_semeval2010_finetune_nobow"))
    parser.add_argument("--structured_checkpoint", default=str(PROJECT_DIR / "models" / "checkpoints" / "structure_v4_partial_role_balanced_fulldev"))
    parser.add_argument("--bow_csv", default=None)
    parser.add_argument("--section_bow_csv", default=str(default_section_bow) if default_section_bow.exists() else None)
    parser.add_argument("--term_frequency_matrix_csv", default=str(default_frequency_matrix) if default_frequency_matrix.exists() else None)
    parser.add_argument("--term_tfidf_matrix_csv", default=str(default_tfidf_matrix) if default_tfidf_matrix.exists() else None)
    parser.add_argument("--evidence_cue_csv", default=str(default_evidence_cues) if default_evidence_cues.exists() else None)
    parser.add_argument("--sentence_evidence_csv", default=str(default_sentence_evidence) if default_sentence_evidence.exists() else None)
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--output_md", default=None)
    parser.add_argument("--top_k_keyphrases", type=int, default=28)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--include_legacy_card", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_txt)
    text = input_path.read_text(encoding="utf-8", errors="ignore")
    card = build_paper_card(
        text=text,
        keyword_checkpoint=args.keyword_checkpoint,
        structured_checkpoint=args.structured_checkpoint,
        title=input_path.stem,
        bow_csv=args.bow_csv,
        top_k_keyphrases=args.top_k_keyphrases,
        device_name=args.device,
    )
    payload = build_evidence_units_payload(
        card=card,
        text=text,
        source_path=input_path,
        keyword_checkpoint=args.keyword_checkpoint,
        structured_checkpoint=args.structured_checkpoint,
        bow_csv=args.bow_csv,
        section_bow_csv=args.section_bow_csv,
        term_frequency_matrix_csv=args.term_frequency_matrix_csv,
        term_tfidf_matrix_csv=args.term_tfidf_matrix_csv,
        evidence_cue_csv=args.evidence_cue_csv,
        sentence_evidence_csv=args.sentence_evidence_csv,
        include_legacy_card=args.include_legacy_card,
    )
    rendered_json = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output_json:
        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(rendered_json, encoding="utf-8")
    if args.output_md:
        output_md = Path(args.output_md)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(render_card_markdown(card), encoding="utf-8")
    print(rendered_json)


if __name__ == "__main__":
    main()


