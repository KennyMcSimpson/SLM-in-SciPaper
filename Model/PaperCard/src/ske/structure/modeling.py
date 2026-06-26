from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers import AutoModel

from .data import IGNORE_INDEX
from .schema import CANONICAL_SECTIONS, ROLE_LABELS


@dataclass
class StructuredModelConfig:
    model_name: str = "allenai/scibert_scivocab_uncased"
    num_sections: int = len(CANONICAL_SECTIONS)
    num_roles: int = len(ROLE_LABELS)
    loss_weighting: str = "fixed"
    role_loss_weight: float = 1.0
    evidence_loss_weight: float = 0.7
    evidence_pos_weight: float = 1.0
    importance_loss_weight: float = 0.35
    importance_pos_weight: float = 1.0
    sentence_context_layers: int = 1
    sentence_context_heads: int = 8
    dropout: float = 0.1


class StructuredPaperModel(nn.Module):
    """Section-aware SciBERT with role, evidence, and importance heads."""

    def __init__(self, config: StructuredModelConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = AutoModel.from_pretrained(config.model_name)
        hidden_size = self.encoder.config.hidden_size
        self.section_embeddings = nn.Embedding(config.num_sections, hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        if config.sentence_context_layers > 0:
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=config.sentence_context_heads,
                dim_feedforward=hidden_size * 4,
                dropout=config.dropout,
                activation="gelu",
                batch_first=True,
            )
            self.sentence_context = nn.TransformerEncoder(layer, num_layers=config.sentence_context_layers)
        else:
            self.sentence_context = None
        self.role_classifier = nn.Linear(hidden_size, config.num_roles)
        self.evidence_classifier = nn.Linear(hidden_size, 1)
        self.importance_regressor = nn.Linear(hidden_size, 1)
        if config.loss_weighting == "uncertainty":
            self.log_task_vars = nn.Parameter(torch.zeros(3))
        else:
            self.log_task_vars = None

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None,
        section_token_ids: torch.Tensor,
        cls_positions: torch.Tensor,
        sentence_mask: torch.Tensor,
        role_labels: torch.Tensor | None = None,
        role_candidate_masks: torch.Tensor | None = None,
        evidence_labels: torch.Tensor | None = None,
        importance_labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        token_embeddings = self.encoder.get_input_embeddings()(input_ids)
        section_embeddings = self.section_embeddings(section_token_ids.clamp(min=0, max=self.config.num_sections - 1))
        inputs_embeds = token_embeddings + section_embeddings
        encoder_outputs = self.encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        hidden_states = encoder_outputs.last_hidden_state
        sentence_reps = gather_sentence_reps(hidden_states, cls_positions, sentence_mask)
        if self.sentence_context is not None:
            sentence_reps = self.sentence_context(sentence_reps, src_key_padding_mask=~sentence_mask)
            sentence_reps = sentence_reps * sentence_mask.unsqueeze(-1)
        sentence_reps = self.dropout(sentence_reps)
        role_logits = self.role_classifier(sentence_reps)
        evidence_logits = self.evidence_classifier(sentence_reps).squeeze(-1)
        importance_logits = self.importance_regressor(sentence_reps).squeeze(-1)
        importance_scores = torch.sigmoid(importance_logits)
        outputs = {
            "role_logits": role_logits,
            "evidence_logits": evidence_logits,
            "evidence_probs": torch.sigmoid(evidence_logits),
            "importance_logits": importance_logits,
            "importance_scores": importance_scores,
        }
        task_losses: list[tuple[str, torch.Tensor, float]] = []
        if role_labels is not None or role_candidate_masks is not None:
            role_loss, role_loss_parts = mixed_role_loss(role_logits, sentence_mask, role_labels, role_candidate_masks)
            outputs["role_loss"] = role_loss
            outputs.update(role_loss_parts)
            if not torch.isnan(role_loss):
                task_losses.append(("role", role_loss, self.config.role_loss_weight))
        if evidence_labels is not None:
            evidence_mask = (evidence_labels >= 0.0) & sentence_mask
            if evidence_mask.any():
                evidence_loss = masked_bce_with_logits(evidence_logits, evidence_labels, evidence_mask, self.config.evidence_pos_weight)
                outputs["evidence_loss"] = evidence_loss
                task_losses.append(("evidence", evidence_loss, self.config.evidence_loss_weight))
        if importance_labels is not None:
            importance_mask = (importance_labels >= 0.0) & sentence_mask
            if importance_mask.any():
                importance_loss = masked_bce_with_logits(importance_logits, importance_labels, importance_mask, self.config.importance_pos_weight)
                outputs["importance_loss"] = importance_loss
                task_losses.append(("importance", importance_loss, self.config.importance_loss_weight))
        if task_losses:
            outputs["loss"] = combine_task_losses(task_losses, self.config.loss_weighting, self.log_task_vars, outputs)
        return outputs

    def save(self, output_dir: str | Path) -> None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), output / "model.pt")
        (output / "model_config.json").write_text(json.dumps(asdict(self.config), indent=2), encoding="utf-8")
        (output / "role_labels.json").write_text(json.dumps(ROLE_LABELS, indent=2), encoding="utf-8")
        (output / "section_labels.json").write_text(json.dumps(CANONICAL_SECTIONS, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, checkpoint_dir: str | Path, map_location: str | torch.device = "cpu") -> "StructuredPaperModel":
        checkpoint = Path(checkpoint_dir)
        config = StructuredModelConfig(**json.loads((checkpoint / "model_config.json").read_text(encoding="utf-8")))
        config.model_name = resolve_model_name(config.model_name, checkpoint)
        model = cls(config)
        state = torch.load(checkpoint / "model.pt", map_location=map_location)
        model.load_state_dict(state)
        return model

    def init_encoder_from_keyphrase_checkpoint(self, checkpoint_dir: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
        checkpoint = Path(checkpoint_dir)
        state = torch.load(checkpoint / "model.pt", map_location=map_location)
        encoder_state = {key: value for key, value in state.items() if key.startswith("encoder.")}
        missing, unexpected = self.load_state_dict(encoder_state, strict=False)
        return {
            "loaded_encoder_tensors": len(encoder_state),
            "missing": list(missing),
            "unexpected": list(unexpected),
        }


def gather_sentence_reps(hidden_states: torch.Tensor, cls_positions: torch.Tensor, sentence_mask: torch.Tensor) -> torch.Tensor:
    batch_size, _, hidden_size = hidden_states.shape
    safe_positions = cls_positions.clamp(min=0)
    expanded = safe_positions.unsqueeze(-1).expand(batch_size, safe_positions.size(1), hidden_size)
    reps = hidden_states.gather(1, expanded)
    return reps * sentence_mask.unsqueeze(-1)


def masked_bce_with_logits(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor, pos_weight: float = 1.0) -> torch.Tensor:
    weights = torch.ones_like(labels)
    if pos_weight > 1.0:
        weights = torch.where(labels >= 0.5, torch.full_like(labels, float(pos_weight)), weights)
    loss = nn.functional.binary_cross_entropy_with_logits(logits, labels.clamp(min=0.0), reduction="none")
    loss = loss * weights
    masked = loss * mask.float()
    return masked.sum() / mask.float().sum().clamp_min(1.0)


def mixed_role_loss(
    role_logits: torch.Tensor,
    sentence_mask: torch.Tensor,
    role_labels: torch.Tensor | None,
    role_candidate_masks: torch.Tensor | None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    losses: list[torch.Tensor] = []
    counts: list[torch.Tensor] = []
    parts: dict[str, torch.Tensor] = {}
    if role_labels is not None:
        hard_valid = (role_labels >= 0) & sentence_mask
        if hard_valid.any():
            hard_loss_all = nn.functional.cross_entropy(
                role_logits.view(-1, role_logits.size(-1)),
                role_labels.view(-1),
                ignore_index=IGNORE_INDEX,
                reduction="none",
            ).view_as(role_labels)
            hard_sum = (hard_loss_all * hard_valid.float()).sum()
            hard_count = hard_valid.float().sum()
            losses.append(hard_sum)
            counts.append(hard_count)
            parts["role_hard_loss"] = (hard_sum / hard_count.clamp_min(1.0)).detach()
            parts["role_hard_count"] = hard_count.detach()
    if role_candidate_masks is not None:
        candidate_counts = role_candidate_masks.sum(dim=-1)
        partial_valid = (candidate_counts > 0) & sentence_mask
        if partial_valid.any():
            log_probs = nn.functional.log_softmax(role_logits, dim=-1)
            negative_infinity = torch.finfo(log_probs.dtype).min
            candidate_log_probs = log_probs.masked_fill(role_candidate_masks <= 0, negative_infinity)
            candidate_log_mass = torch.logsumexp(candidate_log_probs, dim=-1)
            partial_loss_all = -candidate_log_mass
            partial_sum = (partial_loss_all * partial_valid.float()).sum()
            partial_count = partial_valid.float().sum()
            losses.append(partial_sum)
            counts.append(partial_count)
            parts["role_partial_loss"] = (partial_sum / partial_count.clamp_min(1.0)).detach()
            parts["role_partial_count"] = partial_count.detach()
    if not losses:
        return role_logits.sum() * 0.0, parts
    total_loss = torch.stack(losses).sum() / torch.stack(counts).sum().clamp_min(1.0)
    return total_loss, parts


def masked_mse(predictions: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    loss = (predictions - labels.clamp(min=0.0, max=1.0)).pow(2)
    masked = loss * mask.float()
    return masked.sum() / mask.float().sum().clamp_min(1.0)


def combine_task_losses(
    task_losses: list[tuple[str, torch.Tensor, float]],
    loss_weighting: str,
    log_task_vars: torch.nn.Parameter | None,
    outputs: dict[str, torch.Tensor],
) -> torch.Tensor:
    if loss_weighting == "uncertainty":
        if log_task_vars is None:
            raise ValueError("Uncertainty loss weighting requires log_task_vars.")
        task_to_idx = {"role": 0, "evidence": 1, "importance": 2}
        weighted: list[torch.Tensor] = []
        for name, loss, _ in task_losses:
            log_var = log_task_vars[task_to_idx[name]]
            precision = torch.exp(-log_var)
            outputs[f"{name}_loss_precision"] = precision.detach()
            outputs[f"{name}_loss_log_var"] = log_var.detach()
            weighted.append(precision * loss + log_var)
        return torch.stack(weighted).sum()
    return torch.stack([weight * loss for _, loss, weight in task_losses]).sum()


def move_structure_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def resolve_model_name(model_name: str, checkpoint: Path) -> str:
    model_path = Path(model_name).expanduser()
    if model_path.exists():
        return str(model_path)
    if any(separator in model_name for separator in ("/", "\\")):
        for parent in [checkpoint, *checkpoint.parents]:
            candidate = (parent / model_name).resolve()
            if candidate.exists():
                return str(candidate)
    return model_name
