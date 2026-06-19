from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_DIR / "src"))

from ske.data.bow_vocab import BowVocabulary
from ske.data.text_utils import split_sentences
from ske.infer import build_inference_windows, coverage_rerank, extract_candidates_from_bio
from ske.modeling import ScientificKeyphraseExtractor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch keyphrase inference for a folder of txt papers.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--bow_csv", default=None)
    parser.add_argument("--top_k_keyphrases", type=int, default=12)
    parser.add_argument("--top_k_sentences", type=int, default=5)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--max_sentences", type=int, default=160)
    parser.add_argument("--coverage_lambda", type=float, default=0.72)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu")
    if args.device != "auto":
        device = torch.device(args.device)

    checkpoint = Path(args.checkpoint)
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    per_file_dir = output_dir / "json"
    per_file_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True)
    model = ScientificKeyphraseExtractor.load(checkpoint, map_location=device).to(device)
    model.eval()
    bow_vocab = BowVocabulary.load(args.bow_csv)

    files = sorted(input_dir.glob("*.txt"))
    if args.limit is not None:
        files = files[: args.limit]

    summary_rows: list[dict[str, Any]] = []
    for path in tqdm(files, desc="infer_txt"):
        result = infer_one_file(
            path=path,
            tokenizer=tokenizer,
            model=model,
            device=device,
            bow_vocab=bow_vocab,
            top_k_keyphrases=args.top_k_keyphrases,
            top_k_sentences=args.top_k_sentences,
            max_seq_length=args.max_seq_length,
            max_sentences=args.max_sentences,
            coverage_lambda=args.coverage_lambda,
        )
        out_path = per_file_dir / f"{safe_name(path.stem)}.json"
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        summary_rows.append(
            {
                "file": path.name,
                "json": str(out_path),
                "chars": result["chars"],
                "sentences_used": result["sentences_used"],
                "keyphrases": "; ".join(item["keyphrase"] for item in result["keyphrases"]),
                "top_sentence": result["key_sentences"][0]["sentence"] if result["key_sentences"] else "",
            }
        )

    write_summary(output_dir, summary_rows)
    print(json.dumps({"files": len(files), "output_dir": str(output_dir.resolve())}, ensure_ascii=False, indent=2))


def infer_one_file(
    path: Path,
    tokenizer: Any,
    model: ScientificKeyphraseExtractor,
    device: torch.device,
    bow_vocab: BowVocabulary | None,
    top_k_keyphrases: int,
    top_k_sentences: int,
    max_seq_length: int,
    max_sentences: int,
    coverage_lambda: float,
) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    sentences = split_sentences(text)[:max_sentences]
    candidates: list[dict[str, Any]] = []
    sentence_scores: dict[int, float] = {}

    for features in build_inference_windows(sentences, tokenizer, max_seq_length):
        batch = {key: value.to(device) for key, value in features["tensors"].items()}
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch["token_type_ids"],
            cls_positions=batch["cls_positions"],
            sentence_mask=batch["sentence_mask"],
        )
        sentence_probs = outputs["sentence_probs"][0].detach().cpu().tolist()
        for local_idx, global_idx in enumerate(features["sentence_indices"]):
            sentence_scores[global_idx] = max(sentence_scores.get(global_idx, 0.0), float(sentence_probs[local_idx]))

        boundary_probs = outputs["boundary_probs"][0].detach().cpu()
        boundary_preds = outputs["boundary_logits"][0].argmax(dim=-1).detach().cpu().tolist()
        window_candidates = extract_candidates_from_bio(
            sentences=features["sentences"],
            token_meta=features["token_meta"],
            boundary_probs=boundary_probs,
            boundary_preds=boundary_preds,
            sentence_probs=sentence_probs,
            bow_vocab=bow_vocab,
        )
        for item in window_candidates:
            item["sentence_index"] = features["sentence_indices"][item["sentence_index"]]
            candidates.append(item)

    ranked = coverage_rerank(candidates, top_k_keyphrases, coverage_lambda)
    key_sentences = [
        {
            "rank": rank + 1,
            "sentence_index": idx,
            "score": float(score),
            "sentence": sentences[idx],
        }
        for rank, (idx, score) in enumerate(sorted(sentence_scores.items(), key=lambda item: item[1], reverse=True)[:top_k_sentences])
    ]
    return {
        "input": str(path.resolve()),
        "chars": len(text),
        "sentences_used": len(sentences),
        "key_sentences": key_sentences,
        "keyphrases": [
            {
                "rank": idx + 1,
                "keyphrase": item["canonical"],
                "surface": item["surface"],
                "score": item["score"],
                "sentence_index": item["sentence_index"],
                "selector_score": item["selector_score"],
                "boundary_score": item["boundary_score"],
                "bow_confidence": item["bow_confidence"],
                "evidence_sentence": sentences[item["sentence_index"]],
            }
            for idx, item in enumerate(ranked)
        ],
    }


def write_summary(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    (output_dir / "summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output_dir / "summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["file", "json", "chars", "sentences_used", "keyphrases", "top_sentence"])
        writer.writeheader()
        writer.writerows(rows)


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "paper"


if __name__ == "__main__":
    main()

