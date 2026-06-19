from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers import AutoModel


@dataclass
class ModelConfig:
    model_name: str = "allenai/scibert_scivocab_uncased"
    selector_loss_weight: float = 0.4
    boundary_loss_weight: float = 1.0
    sentence_context_layers: int = 1
    sentence_context_heads: int = 8
    dropout: float = 0.1


class ScientificKeyphraseExtractor(nn.Module):
    """SciBERT encoder with keyword-evidence selector and BIO boundary head."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = AutoModel.from_pretrained(config.model_name)
        hidden_size = self.encoder.config.hidden_size
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
        self.selector = nn.Linear(hidden_size, 1)
        self.boundary_classifier = nn.Linear(hidden_size, 3)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None,
        cls_positions: torch.Tensor,
        sentence_mask: torch.Tensor,
        sentence_labels: torch.Tensor | None = None,
        bio_labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hidden_states = encoder_outputs.last_hidden_state
        sentence_reps = self._gather_sentence_reps(hidden_states, cls_positions, sentence_mask)
        if self.sentence_context is not None:
            sentence_reps = self.sentence_context(sentence_reps, src_key_padding_mask=~sentence_mask)
            sentence_reps = sentence_reps * sentence_mask.unsqueeze(-1)
        sentence_logits = self.selector(self.dropout(sentence_reps)).squeeze(-1)
        boundary_logits = self.boundary_classifier(self.dropout(hidden_states))
        outputs = {
            "sentence_logits": sentence_logits,
            "sentence_probs": torch.sigmoid(sentence_logits),
            "boundary_logits": boundary_logits,
            "boundary_probs": torch.softmax(boundary_logits, dim=-1),
        }
        losses: list[torch.Tensor] = []
        if sentence_labels is not None:
            selector_loss = masked_bce_with_logits(sentence_logits, sentence_labels, sentence_mask)
            outputs["selector_loss"] = selector_loss
            losses.append(self.config.selector_loss_weight * selector_loss)
        if bio_labels is not None:
            boundary_loss = nn.functional.cross_entropy(
                boundary_logits.view(-1, boundary_logits.size(-1)),
                bio_labels.view(-1),
                ignore_index=-100,
            )
            outputs["boundary_loss"] = boundary_loss
            losses.append(self.config.boundary_loss_weight * boundary_loss)
        if losses:
            outputs["loss"] = torch.stack(losses).sum()
        return outputs

    def save(self, output_dir: str | Path) -> None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), output / "model.pt")
        (output / "model_config.json").write_text(json.dumps(asdict(self.config), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, checkpoint_dir: str | Path, map_location: str | torch.device = "cpu") -> "ScientificKeyphraseExtractor":
        checkpoint = Path(checkpoint_dir)
        config = ModelConfig(**json.loads((checkpoint / "model_config.json").read_text(encoding="utf-8")))
        config.model_name = resolve_model_name(config.model_name, checkpoint)
        model = cls(config)
        state = torch.load(checkpoint / "model.pt", map_location=map_location)
        model.load_state_dict(state)
        return model

    def _gather_sentence_reps(self, hidden_states: torch.Tensor, cls_positions: torch.Tensor, sentence_mask: torch.Tensor) -> torch.Tensor:
        batch_size, _, hidden_size = hidden_states.shape
        safe_positions = cls_positions.clamp(min=0)
        expanded = safe_positions.unsqueeze(-1).expand(batch_size, safe_positions.size(1), hidden_size)
        reps = hidden_states.gather(1, expanded)
        return reps * sentence_mask.unsqueeze(-1)


def masked_bce_with_logits(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    loss = nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    masked = loss * mask.float()
    return masked.sum() / mask.float().sum().clamp_min(1.0)


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
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
