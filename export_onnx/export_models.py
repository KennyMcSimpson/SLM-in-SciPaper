"""
ONNX Export & INT8 Quantization Script
========================================
Step 1 of the Client-Edge-Cloud Migration (see PROJECT_ARCHITECTURE.md).

Exports two PyTorch checkpoints to ONNX format and quantizes them to INT8,
reducing each model from ~400 MB → ~100 MB for WebAssembly / ONNX Runtime Web distribution.

Models exported:
  1. keyword_scibert  → keyword_extractor.onnx  (INT8: keyword_extractor_int8.onnx)
  2. structure_v2_scibert_evidencefix → structure_model.onnx (INT8: structure_model_int8.onnx)

Usage (run from the PaperCard project root):
  python code/export_onnx/export_models.py \
    --keyword_checkpoint  models/checkpoints/keyword_scibert_semeval2010_finetune_nobow \
    --structure_checkpoint models/checkpoints/structure_v2_scibert_evidencefix \
    --output_dir           onnx_exports

Requirements:
  pip install torch onnx onnxruntime transformers
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ─── Path bootstrap ────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_DIR / "Model" / "PaperCard" / "src"))

import torch
from transformers import AutoTokenizer

from ske.modeling import ScientificKeyphraseExtractor
from ske.structure.modeling import StructuredPaperModel
from ske.structure.schema import CANONICAL_SECTIONS


# ─── ONNX Export Config ────────────────────────────────────────────────────────

# Maximum token sequence length used during training (keep consistent with StructureFeatureConfig)
MAX_SEQ_LEN = 512
# Maximum number of sentences per window (keyword model)
MAX_SENTENCES_KW = 16
# Maximum number of sentences per window (structure model)
MAX_SENTENCES_ST = 48


# ─── Keyword Model Wrapper ─────────────────────────────────────────────────────


class KeywordOnnxWrapper(torch.nn.Module):
    """
    Wraps ScientificKeyphraseExtractor for ONNX export.

    ONNX does not allow dynamic conditional branches on traced tensors, so we
    pre-run the sentence_context layer inside forward() and always assume it
    exists (it does for the trained checkpoint).

    Inputs:
        input_ids       : (1, seq_len)  long
        attention_mask  : (1, seq_len)  long
        token_type_ids  : (1, seq_len)  long
        cls_positions   : (1, num_sent) long
        sentence_mask   : (1, num_sent) bool

    Outputs:
        sentence_probs  : (1, num_sent) float  – per-sentence relevance probability
        boundary_probs  : (1, seq_len, 3) float – B/I/O token boundary probs
    """

    def __init__(self, model: ScientificKeyphraseExtractor) -> None:
        super().__init__()
        self.encoder = model.encoder
        self.dropout = model.dropout
        self.sentence_context = model.sentence_context
        self.selector = model.selector
        self.boundary_classifier = model.boundary_classifier

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
        cls_positions: torch.Tensor,
        sentence_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        ).last_hidden_state  # (B, seq, H)

        # Gather CLS representations for each sentence
        B, _, H = hidden_states.shape
        safe_pos = cls_positions.clamp(min=0)  # (B, S)
        expanded = safe_pos.unsqueeze(-1).expand(B, safe_pos.size(1), H)  # (B, S, H)
        sentence_reps = hidden_states.gather(1, expanded)  # (B, S, H)
        sentence_reps = sentence_reps * sentence_mask.unsqueeze(-1).float()

        if self.sentence_context is not None:
            sentence_reps = self.sentence_context(
                sentence_reps, src_key_padding_mask=~sentence_mask
            )
            sentence_reps = sentence_reps * sentence_mask.unsqueeze(-1).float()

        sentence_logits = self.selector(self.dropout(sentence_reps)).squeeze(
            -1
        )  # (B, S)
        sentence_probs = torch.sigmoid(sentence_logits)

        boundary_logits = self.boundary_classifier(
            self.dropout(hidden_states)
        )  # (B, seq, 3)
        boundary_probs = torch.softmax(boundary_logits, dim=-1)

        return sentence_probs, boundary_probs


# ─── Structure Model Wrapper ───────────────────────────────────────────────────


class StructureOnnxWrapper(torch.nn.Module):
    """
    Wraps StructuredPaperModel for ONNX export.

    Inputs:
        input_ids        : (1, seq_len)  long
        attention_mask   : (1, seq_len)  long
        token_type_ids   : (1, seq_len)  long
        section_token_ids: (1, seq_len)  long   – per-token section index [0..4]
        cls_positions    : (1, num_sent) long
        sentence_mask    : (1, num_sent) bool

    Outputs:
        role_probs       : (1, num_sent, num_roles) float
        evidence_probs   : (1, num_sent) float
        importance_scores: (1, num_sent) float
    """

    def __init__(self, model: StructuredPaperModel) -> None:
        super().__init__()
        self.encoder = model.encoder
        self.section_embeddings = model.section_embeddings
        self.dropout = model.dropout
        self.sentence_context = model.sentence_context
        self.role_classifier = model.role_classifier
        self.evidence_classifier = model.evidence_classifier
        self.importance_regressor = model.importance_regressor
        self.num_sections = len(CANONICAL_SECTIONS)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
        section_token_ids: torch.Tensor,
        cls_positions: torch.Tensor,
        sentence_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        token_embeddings = self.encoder.get_input_embeddings()(input_ids)
        section_embeddings = self.section_embeddings(
            section_token_ids.clamp(min=0, max=self.num_sections - 1)
        )
        inputs_embeds = token_embeddings + section_embeddings

        hidden_states = self.encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        ).last_hidden_state  # (B, seq, H)

        # Gather sentence CLS reps
        B, _, H = hidden_states.shape
        safe_pos = cls_positions.clamp(min=0)
        expanded = safe_pos.unsqueeze(-1).expand(B, safe_pos.size(1), H)
        sentence_reps = hidden_states.gather(1, expanded)
        sentence_reps = sentence_reps * sentence_mask.unsqueeze(-1).float()

        if self.sentence_context is not None:
            sentence_reps = self.sentence_context(
                sentence_reps, src_key_padding_mask=~sentence_mask
            )
            sentence_reps = sentence_reps * sentence_mask.unsqueeze(-1).float()

        sentence_reps = self.dropout(sentence_reps)
        role_logits = self.role_classifier(sentence_reps)  # (B, S, R)
        role_probs = torch.softmax(role_logits, dim=-1)

        evidence_logits = self.evidence_classifier(sentence_reps).squeeze(-1)  # (B, S)
        evidence_probs = torch.sigmoid(evidence_logits)

        importance_logits = self.importance_regressor(sentence_reps).squeeze(
            -1
        )  # (B, S)
        importance_scores = torch.sigmoid(importance_logits)

        return role_probs, evidence_probs, importance_scores


# ─── Export Helpers ────────────────────────────────────────────────────────────


def _make_dummy_keyword_inputs(
    tokenizer: any, num_sentences: int = MAX_SENTENCES_KW
) -> tuple:
    """Build dummy ONNX-trace inputs for the keyword model."""
    seq_len = MAX_SEQ_LEN
    input_ids = torch.zeros(1, seq_len, dtype=torch.long)
    attention_mask = torch.ones(1, seq_len, dtype=torch.long)
    token_type_ids = torch.zeros(1, seq_len, dtype=torch.long)
    cls_positions = torch.zeros(1, num_sentences, dtype=torch.long)
    sentence_mask = torch.ones(1, num_sentences, dtype=torch.bool)
    # Place [CLS] token id
    input_ids[0, 0] = tokenizer.cls_token_id or 101
    return input_ids, attention_mask, token_type_ids, cls_positions, sentence_mask


def _make_dummy_structure_inputs(
    tokenizer: any, num_sentences: int = MAX_SENTENCES_ST
) -> tuple:
    """Build dummy ONNX-trace inputs for the structure model."""
    seq_len = MAX_SEQ_LEN
    input_ids = torch.zeros(1, seq_len, dtype=torch.long)
    attention_mask = torch.ones(1, seq_len, dtype=torch.long)
    token_type_ids = torch.zeros(1, seq_len, dtype=torch.long)
    section_token_ids = torch.zeros(1, seq_len, dtype=torch.long)
    cls_positions = torch.zeros(1, num_sentences, dtype=torch.long)
    sentence_mask = torch.ones(1, num_sentences, dtype=torch.bool)
    input_ids[0, 0] = tokenizer.cls_token_id or 101
    return (
        input_ids,
        attention_mask,
        token_type_ids,
        section_token_ids,
        cls_positions,
        sentence_mask,
    )


def export_keyword_model(
    checkpoint_dir: Path,
    output_dir: Path,
    device: torch.device,
) -> Path:
    """Load, wrap, and export the keyword extractor to ONNX."""
    print(f"\n[Keyword] Loading checkpoint from: {checkpoint_dir}")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, use_fast=True)
    model = ScientificKeyphraseExtractor.load(checkpoint_dir, map_location=device).to(
        device
    )
    model.eval()

    wrapper = KeywordOnnxWrapper(model).to(device)
    wrapper.eval()

    dummy_inputs = tuple(t.to(device) for t in _make_dummy_keyword_inputs(tokenizer))

    onnx_path = output_dir / "keyword_extractor.onnx"
    print(f"[Keyword] Exporting to ONNX: {onnx_path}")

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy_inputs,
            str(onnx_path),
            opset_version=17,
            input_names=[
                "input_ids",
                "attention_mask",
                "token_type_ids",
                "cls_positions",
                "sentence_mask",
            ],
            output_names=["sentence_probs", "boundary_probs"],
            dynamic_axes={
                # seq dimension is dynamic
                "input_ids": {0: "batch", 1: "seq_len"},
                "attention_mask": {0: "batch", 1: "seq_len"},
                "token_type_ids": {0: "batch", 1: "seq_len"},
                # sentence dimension is dynamic
                "cls_positions": {0: "batch", 1: "num_sentences"},
                "sentence_mask": {0: "batch", 1: "num_sentences"},
                "sentence_probs": {0: "batch", 1: "num_sentences"},
                "boundary_probs": {0: "batch", 1: "seq_len"},
            },
            do_constant_folding=True,
        )

    print(
        f"[Keyword] ONNX export complete: {onnx_path} ({onnx_path.stat().st_size // 1024 // 1024} MB)"
    )

    # Save tokenizer vocab alongside the model for Web inference
    _save_tokenizer_vocab(tokenizer, output_dir, "keyword_vocab.json")

    return onnx_path


def export_structure_model(
    checkpoint_dir: Path,
    output_dir: Path,
    device: torch.device,
    tokenizer_fallback_dir: Path | None = None,
) -> Path:
    """Load, wrap, and export the structure model to ONNX."""
    print(f"\n[Structure] Loading checkpoint from: {checkpoint_dir}")
    # Structure checkpoint may lack tokenizer files (it shares SciBERT vocab with keyword model)
    tokenizer_dir = checkpoint_dir
    if not (checkpoint_dir / "tokenizer.json").exists() and tokenizer_fallback_dir is not None:
        tokenizer_dir = tokenizer_fallback_dir
        print(f"  [Tokenizer] Using fallback tokenizer from: {tokenizer_fallback_dir}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, use_fast=True)
    model = StructuredPaperModel.load(checkpoint_dir, map_location=device).to(device)
    model.eval()

    wrapper = StructureOnnxWrapper(model).to(device)
    wrapper.eval()

    dummy_inputs = tuple(t.to(device) for t in _make_dummy_structure_inputs(tokenizer))

    onnx_path = output_dir / "structure_model.onnx"
    print(f"[Structure] Exporting to ONNX: {onnx_path}")

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy_inputs,
            str(onnx_path),
            opset_version=17,
            input_names=[
                "input_ids",
                "attention_mask",
                "token_type_ids",
                "section_token_ids",
                "cls_positions",
                "sentence_mask",
            ],
            output_names=["role_probs", "evidence_probs", "importance_scores"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "seq_len"},
                "attention_mask": {0: "batch", 1: "seq_len"},
                "token_type_ids": {0: "batch", 1: "seq_len"},
                "section_token_ids": {0: "batch", 1: "seq_len"},
                "cls_positions": {0: "batch", 1: "num_sentences"},
                "sentence_mask": {0: "batch", 1: "num_sentences"},
                "role_probs": {0: "batch", 1: "num_sentences"},
                "evidence_probs": {0: "batch", 1: "num_sentences"},
                "importance_scores": {0: "batch", 1: "num_sentences"},
            },
            do_constant_folding=True,
        )

    print(
        f"[Structure] ONNX export complete: {onnx_path} ({onnx_path.stat().st_size // 1024 // 1024} MB)"
    )

    # Save tokenizer vocab alongside the model for Web inference
    _save_tokenizer_vocab(tokenizer, output_dir, "structure_vocab.json")

    return onnx_path


def quantize_to_int8(onnx_path: Path) -> Path:
    """
    Apply dynamic INT8 quantization using onnxruntime.quantization.

    Dynamic quantization calibrates weight ranges at quantize-time (no
    calibration dataset needed), which is perfect for an offline offline step.
    It typically reduces model size by 3–4× with minimal accuracy loss for
    Transformer encoder models.
    """
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
    except ImportError:
        print("  [WARN] onnxruntime-tools not installed. Skipping INT8 quantization.")
        print("         Install with: pip install onnxruntime onnxruntime-tools")
        return onnx_path

    int8_path = onnx_path.with_name(onnx_path.stem + "_int8.onnx")
    print(f"  [INT8] Quantizing → {int8_path}")

    quantize_dynamic(
        model_input=str(onnx_path),
        model_output=str(int8_path),
        weight_type=QuantType.QInt8,
    )

    original_mb = onnx_path.stat().st_size // 1024 // 1024
    quantized_mb = int8_path.stat().st_size // 1024 // 1024
    ratio = original_mb / max(quantized_mb, 1)
    print(
        f"  [INT8] {original_mb} MB → {quantized_mb} MB  (compression ratio: {ratio:.1f}x)"
    )
    return int8_path


def _save_tokenizer_vocab(tokenizer: any, output_dir: Path, filename: str) -> None:
    """
    Save the tokenizer vocabulary as a JSON mapping {token: id} for use in
    JavaScript (the Web Worker tokenizer implements BertWordPieceTokenizer
    logic from the vocab file).
    """
    vocab = tokenizer.get_vocab()
    vocab_path = output_dir / filename
    vocab_path.write_text(
        json.dumps(vocab, ensure_ascii=False, indent=None, separators=(",", ":")),
        encoding="utf-8",
    )
    print(f"  [Vocab] Saved {len(vocab)} tokens → {vocab_path}")


def _save_export_manifest(output_dir: Path, results: dict) -> None:
    """Save a JSON manifest describing the exported models for the frontend to consume."""
    manifest = {
        "version": "1.0",
        "description": "ONNX models for SciPaper RAG client-side inference",
        "models": results,
        "onnx_opset": 17,
        "max_seq_length": MAX_SEQ_LEN,
        "sections": ["intro", "related_work", "method", "experiment", "conclusion"],
        "role_labels": [
            "none",
            "background",
            "problem",
            "motivation",
            "objective",
            "prior_work",
            "limitation",
            "gap",
            "comparison",
            "core_method",
            "component",
            "mechanism",
            "process",
            "dataset",
            "metric",
            "baseline",
            "result",
            "ablation",
            "contribution",
            "finding",
            "future_work",
        ],
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n[Manifest] Saved export manifest → {manifest_path}")


# ─── CLI ───────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export SciBERT keyword and structure models to INT8-quantized ONNX.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--keyword_checkpoint",
        default=str(
            PROJECT_DIR
            / "models"
            / "checkpoints"
            / "keyword_scibert_semeval2010_finetune_nobow"
        ),
        help="Path to the trained keyword extractor checkpoint directory.",
    )
    parser.add_argument(
        "--structure_checkpoint",
        default=str(
            PROJECT_DIR / "Model" / "PaperCard" / "models" / "checkpoints" / "structure_v2_scibert_evidencefix"
        ),
        help="Path to the trained structure model checkpoint directory.",
    )
    parser.add_argument(
        "--output_dir",
        default=str(PROJECT_DIR / "onnx_exports"),
        help="Directory to write ONNX and INT8 model files.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to load the PyTorch model on before export (use cpu for ONNX export).",
    )
    parser.add_argument(
        "--skip_quantization",
        action="store_true",
        help="Export FP32 ONNX only, skip INT8 quantization step.",
    )
    parser.add_argument(
        "--keyword_only",
        action="store_true",
        help="Export only the keyword extractor model.",
    )
    parser.add_argument(
        "--structure_only",
        action="store_true",
        help="Export only the structure model.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("SciPaper ONNX Export & INT8 Quantization")
    print("=" * 60)
    print(f"Output directory : {output_dir.resolve()}")
    print(f"Device           : {device}")
    print(
        f"Quantization     : {'disabled' if args.skip_quantization else 'INT8 dynamic'}"
    )
    print()

    results: dict = {}

    # ── Export keyword model ──────────────────────────────────────────────────
    if not args.structure_only:
        kw_checkpoint = Path(args.keyword_checkpoint)
        if not kw_checkpoint.exists():
            print(f"[ERROR] Keyword checkpoint not found: {kw_checkpoint}")
            sys.exit(1)

        kw_onnx = export_keyword_model(kw_checkpoint, output_dir, device)
        results["keyword_extractor"] = {
            "fp32_onnx": kw_onnx.name,
            "size_mb": kw_onnx.stat().st_size // 1024 // 1024,
        }

        if not args.skip_quantization:
            kw_int8 = quantize_to_int8(kw_onnx)
            results["keyword_extractor"]["int8_onnx"] = kw_int8.name
            results["keyword_extractor"]["int8_size_mb"] = (
                kw_int8.stat().st_size // 1024 // 1024
            )

    # ── Export structure model ────────────────────────────────────────────────
    if not args.keyword_only:
        st_checkpoint = Path(args.structure_checkpoint)
        if not st_checkpoint.exists():
            print(f"[ERROR] Structure checkpoint not found: {st_checkpoint}")
            sys.exit(1)

        # Pass keyword checkpoint as tokenizer fallback (structure checkpoint shares SciBERT vocab)
        kw_fallback = Path(args.keyword_checkpoint) if args.keyword_checkpoint else None
        st_onnx = export_structure_model(
            st_checkpoint, output_dir, device,
            tokenizer_fallback_dir=kw_fallback,
        )
        results["structure_model"] = {
            "fp32_onnx": st_onnx.name,
            "size_mb": st_onnx.stat().st_size // 1024 // 1024,
        }

        if not args.skip_quantization:
            st_int8 = quantize_to_int8(st_onnx)
            results["structure_model"]["int8_onnx"] = st_int8.name
            results["structure_model"]["int8_size_mb"] = (
                st_int8.stat().st_size // 1024 // 1024
            )

    _save_export_manifest(output_dir, results)

    print("\n" + "=" * 60)
    print("Export pipeline complete!")
    print(f"   Output: {output_dir.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
