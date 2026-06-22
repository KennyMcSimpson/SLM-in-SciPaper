from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
ROOT_DIR = PROJECT_DIR.parent
PYTHON = sys.executable


def main() -> None:
    run_dir = PROJECT_DIR / "runs" / "structure_v2_smoke"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    data_dir = run_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    train_jsonl = data_dir / "train.jsonl"
    dev_jsonl = data_dir / "dev.jsonl"
    sample_txt = data_dir / "sample_paper.txt"
    write_smoke_dataset(train_jsonl)
    write_smoke_dataset(dev_jsonl)
    sample_txt.write_text(SAMPLE_PAPER, encoding="utf-8")

    checkpoint_dir = run_dir / "model"
    train_cmd = [
        PYTHON,
        str(PROJECT_DIR / "src" / "ske" / "structure" / "train.py"),
        "--train_jsonl",
        str(train_jsonl),
        "--dev_jsonl",
        str(dev_jsonl),
        "--output_dir",
        str(checkpoint_dir),
        "--model_name",
        "hf-internal-testing/tiny-random-bert",
        "--epochs",
        "1",
        "--batch_size",
        "2",
        "--max_seq_length",
        "192",
        "--device",
        "cpu",
    ]
    run(train_cmd)

    keyword_checkpoint = PROJECT_DIR / "runs" / "tiny_train_smoke"
    if keyword_checkpoint.exists():
        infer_cmd = [
            PYTHON,
            str(PROJECT_DIR / "code" / "03_inference_summary" / "infer_paper_card.py"),
            "--input_txt",
            str(sample_txt),
            "--keyword_checkpoint",
            str(keyword_checkpoint),
            "--structured_checkpoint",
            str(checkpoint_dir),
            "--output_json",
            str(run_dir / "paper_card.json"),
            "--output_md",
            str(run_dir / "paper_card.md"),
            "--top_k_keyphrases",
            "8",
            "--device",
            "cpu",
        ]
        run(infer_cmd)
    result = {
        "ok": True,
        "run_dir": str(run_dir),
        "trained_checkpoint": str(checkpoint_dir),
        "paper_card_json": str(run_dir / "paper_card.json") if (run_dir / "paper_card.json").exists() else None,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


def write_smoke_dataset(path: Path) -> None:
    records = [
        {
            "doc_id": "smoke-1",
            "title": "A tiny paper about attention",
            "source": "smoke",
            "sentences": [
                sentence("Neural machine translation remains difficult for long sequences.", "intro", "problem", None, 0.7, 0),
                sentence("We propose a self attention encoder for sequence transduction.", "method", "core_method", 1.0, 0.95, 1),
                sentence("The model is trained on WMT datasets and evaluated with BLEU.", "experiment", "dataset", 0.0, 0.65, 2),
                sentence("Results show improved translation quality over recurrent baselines.", "experiment", "result", 1.0, 0.9, 3),
                sentence("The conclusion is that attention reduces sequential computation.", "conclusion", "finding", None, 0.8, 4),
            ],
        },
        {
            "doc_id": "smoke-2",
            "title": "A tiny paper about pretraining",
            "source": "smoke",
            "sentences": [
                sentence("Language understanding needs transferable representations.", "intro", "background", None, 0.5, 0),
                sentence("Prior feature based approaches are limited.", "related_work", "limitation", 0.0, 0.6, 1),
                sentence("We introduce bidirectional pre training with masked tokens.", "method", "core_method", 1.0, 0.9, 2),
                sentence("Experiments on question answering and natural language inference improve accuracy.", "experiment", "result", 1.0, 0.9, 3),
                sentence("Future work should study larger corpora.", "conclusion", "future_work", None, 0.55, 4),
            ],
        },
    ]
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def sentence(text: str, section: str, role: str, evidence: float | None, importance: float, idx: int) -> dict:
    return {
        "text": text,
        "section": section,
        "role": role,
        "evidence_label": evidence,
        "importance_label": importance,
        "section_title": section,
        "source": "smoke",
        "sentence_index": idx,
    }


def run(cmd: list[str]) -> None:
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=str(ROOT_DIR), check=True)


SAMPLE_PAPER = """Introduction
Neural machine translation remains difficult for long sequences. We study a sequence transduction model for this problem.

Method
We propose a self attention encoder that computes contextual representations without recurrence. The architecture uses attention heads and positional encoding.

Experiments
The model is trained on WMT datasets and evaluated with BLEU. Results show improved translation quality over recurrent baselines.

Conclusion
The conclusion is that attention reduces sequential computation while preserving translation quality.
"""


if __name__ == "__main__":
    main()


