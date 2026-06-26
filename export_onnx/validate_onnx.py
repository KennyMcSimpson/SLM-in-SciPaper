"""
ONNX Model Validation Script
==============================
Validates exported ONNX models against the original PyTorch outputs to ensure
numerical correctness before deploying to the browser.

Usage:
  python code/export_onnx/validate_onnx.py \
    --keyword_checkpoint  models/checkpoints/keyword_scibert_semeval2010_finetune_nobow \
    --structure_checkpoint models/checkpoints/structure_v2_scibert_evidencefix \
    --onnx_dir             onnx_exports \
    --input_txt            datasets/03_demo_txt/full_library/2017_transformer_and_large_language_models_attention_is_all_you_need.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_DIR / "Model" / "PaperCard" / "src"))

from transformers import AutoTokenizer
from ske.modeling import ScientificKeyphraseExtractor
from ske.structure.modeling import StructuredPaperModel
from ske.structure.schema import CANONICAL_SECTIONS
from ske.data.text_utils import split_sentences
from ske.infer import build_inference_windows

# Local imports from export script
sys.path.insert(0, str(Path(__file__).parent))
from export_models import (
    KeywordOnnxWrapper,
    StructureOnnxWrapper,
    _make_dummy_keyword_inputs,
    _make_dummy_structure_inputs,
    MAX_SEQ_LEN,
    MAX_SENTENCES_KW,
    MAX_SENTENCES_ST,
)

ATOL = 1e-4  # Acceptable absolute tolerance between PyTorch and ONNX outputs


def validate_keyword_model(
    checkpoint_dir: Path,
    onnx_dir: Path,
    input_text: str,
    device: torch.device,
) -> bool:
    print("\n[Keyword Validation]")
    try:
        import onnxruntime as ort
    except ImportError:
        print(
            "  [SKIP] onnxruntime not installed. Install with: pip install onnxruntime"
        )
        return True

    # Load models
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, use_fast=True)
    pt_model = ScientificKeyphraseExtractor.load(
        checkpoint_dir, map_location=device
    ).to(device)
    pt_model.eval()
    wrapper = KeywordOnnxWrapper(pt_model).to(device)
    wrapper.eval()

    # Build real inference windows from the input text
    sentences = split_sentences(input_text)[:MAX_SENTENCES_KW]
    windows = build_inference_windows(sentences, tokenizer, MAX_SEQ_LEN)
    if not windows:
        print("  [WARN] No windows built from input text; using dummy inputs.")
        dummy = _make_dummy_keyword_inputs(tokenizer, num_sentences=4)
        tensors = {
            "input_ids": dummy[0],
            "attention_mask": dummy[1],
            "token_type_ids": dummy[2],
            "cls_positions": dummy[3],
            "sentence_mask": dummy[4],
        }
        windows = [
            {
                "tensors": tensors,
                "sentence_indices": list(range(4)),
                "sentences": [""] * 4,
                "token_meta": [],
            }
        ]

    # Use first window for comparison
    batch_tensors = {k: v.to(device) for k, v in windows[0]["tensors"].items()}

    # PyTorch forward pass
    with torch.no_grad():
        sent_probs_pt, boundary_probs_pt = wrapper(
            batch_tensors["input_ids"],
            batch_tensors["attention_mask"],
            batch_tensors["token_type_ids"],
            batch_tensors["cls_positions"],
            batch_tensors["sentence_mask"],
        )

    # ONNX Runtime forward pass
    onnx_path = onnx_dir / "keyword_extractor.onnx"
    if not onnx_path.exists():
        print(f"  [ERROR] ONNX not found: {onnx_path}")
        return False

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_inputs = {
        "input_ids": batch_tensors["input_ids"].cpu().numpy(),
        "attention_mask": batch_tensors["attention_mask"].cpu().numpy(),
        "token_type_ids": batch_tensors["token_type_ids"].cpu().numpy(),
        "cls_positions": batch_tensors["cls_positions"].cpu().numpy(),
        "sentence_mask": batch_tensors["sentence_mask"].cpu().numpy(),
    }
    sent_probs_ort, boundary_probs_ort = sess.run(None, ort_inputs)

    # Compare
    max_diff_sent = float(np.abs(sent_probs_pt.cpu().numpy() - sent_probs_ort).max())
    max_diff_boundary = float(
        np.abs(boundary_probs_pt.cpu().numpy() - boundary_probs_ort).max()
    )

    print(f"  Max diff sentence_probs  : {max_diff_sent:.2e}  (atol={ATOL:.0e})")
    print(f"  Max diff boundary_probs  : {max_diff_boundary:.2e}  (atol={ATOL:.0e})")

    ok = max_diff_sent < ATOL and max_diff_boundary < ATOL
    print(f"  Result: {'PASS' if ok else 'FAIL'}")

    # Also validate INT8 if it exists
    int8_path = onnx_dir / "keyword_extractor_int8.onnx"
    if int8_path.exists():
        sess8 = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])
        sent_probs_int8, boundary_probs_int8 = sess8.run(None, ort_inputs)
        max_diff_int8 = float(
            np.abs(sent_probs_pt.cpu().numpy() - sent_probs_int8).max()
        )
        print(f"  INT8 max diff sent_probs : {max_diff_int8:.2e}  (acceptable < 0.01)")
        print(
            f"  INT8 result: {'PASS' if max_diff_int8 < 0.01 else ' WARN (higher tolerance expected)'}"
        )

    return ok


def validate_structure_model(
    checkpoint_dir: Path,
    onnx_dir: Path,
    device: torch.device,
) -> bool:
    print("\n[Structure Validation]")
    try:
        import onnxruntime as ort
    except ImportError:
        print("  [SKIP] onnxruntime not installed.")
        return True

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, use_fast=True)
    pt_model = StructuredPaperModel.load(checkpoint_dir, map_location=device).to(device)
    pt_model.eval()
    wrapper = StructureOnnxWrapper(pt_model).to(device)
    wrapper.eval()

    dummy = _make_dummy_structure_inputs(tokenizer, num_sentences=8)
    batch_tensors = {
        "input_ids": dummy[0].to(device),
        "attention_mask": dummy[1].to(device),
        "token_type_ids": dummy[2].to(device),
        "section_token_ids": dummy[3].to(device),
        "cls_positions": dummy[4].to(device),
        "sentence_mask": dummy[5].to(device),
    }

    with torch.no_grad():
        role_probs_pt, evidence_probs_pt, importance_pt = wrapper(
            batch_tensors["input_ids"],
            batch_tensors["attention_mask"],
            batch_tensors["token_type_ids"],
            batch_tensors["section_token_ids"],
            batch_tensors["cls_positions"],
            batch_tensors["sentence_mask"],
        )

    onnx_path = onnx_dir / "structure_model.onnx"
    if not onnx_path.exists():
        print(f"  [ERROR] ONNX not found: {onnx_path}")
        return False

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_inputs = {k: v.cpu().numpy() for k, v in batch_tensors.items()}
    role_probs_ort, evidence_probs_ort, importance_ort = sess.run(None, ort_inputs)

    max_diff_role = float(np.abs(role_probs_pt.cpu().numpy() - role_probs_ort).max())
    max_diff_ev = float(
        np.abs(evidence_probs_pt.cpu().numpy() - evidence_probs_ort).max()
    )
    max_diff_imp = float(np.abs(importance_pt.cpu().numpy() - importance_ort).max())

    print(f"  Max diff role_probs      : {max_diff_role:.2e}  (atol={ATOL:.0e})")
    print(f"  Max diff evidence_probs  : {max_diff_ev:.2e}  (atol={ATOL:.0e})")
    print(f"  Max diff importance_sc   : {max_diff_imp:.2e}  (atol={ATOL:.0e})")

    ok = max(max_diff_role, max_diff_ev, max_diff_imp) < ATOL
    print(f"  Result: {'PASS' if ok else 'FAIL'}")

    int8_path = onnx_dir / "structure_model_int8.onnx"
    if int8_path.exists():
        sess8 = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])
        role8, ev8, imp8 = sess8.run(None, ort_inputs)
        max_diff_role8 = float(np.abs(role_probs_pt.cpu().numpy() - role8).max())
        print(f"  INT8 max diff role_probs : {max_diff_role8:.2e}")
        print(f"  INT8 result: {'PASS' if max_diff_role8 < 0.02 else ' WARN'}")

    return ok


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate exported ONNX models against PyTorch."
    )
    parser.add_argument(
        "--keyword_checkpoint",
        default=str(
            PROJECT_DIR
            / "models"
            / "checkpoints"
            / "keyword_scibert_semeval2010_finetune_nobow"
        ),
    )
    parser.add_argument(
        "--structure_checkpoint",
        default=str(
            PROJECT_DIR / "Model" / "PaperCard" / "models" / "checkpoints" / "structure_v2_scibert_evidencefix"
        ),
    )
    parser.add_argument("--onnx_dir", default=str(PROJECT_DIR / "onnx_exports"))
    parser.add_argument(
        "--input_txt",
        default=None,
        help="Optional: path to a paper txt for real-data validation.",
    )
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    onnx_dir = Path(args.onnx_dir)

    input_text = ""
    if args.input_txt:
        input_text = Path(args.input_txt).read_text(encoding="utf-8", errors="ignore")

    print("=" * 60)
    print("SciPaper ONNX Validation")
    print("=" * 60)

    kw_ok = validate_keyword_model(
        Path(args.keyword_checkpoint), onnx_dir, input_text, device
    )
    st_ok = validate_structure_model(Path(args.structure_checkpoint), onnx_dir, device)

    print("\n" + "=" * 60)
    all_ok = kw_ok and st_ok
    print(
        f"Overall result: {'All validations PASSED' if all_ok else 'Some validations FAILED'}"
    )
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
