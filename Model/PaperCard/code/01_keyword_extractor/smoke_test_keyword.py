from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_DIR / "src"))

from transformers import AutoTokenizer

from ske.data.bow_vocab import BowVocabulary
from ske.data.dataset import FeatureConfig, build_features, collate_features
from ske.modeling import ModelConfig, ScientificKeyphraseExtractor


def main() -> None:
    model_name = "hf-internal-testing/tiny-random-bert"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    bow_path = PROJECT_DIR / "resources" / "vocab" / "final_bow_vocabulary.csv"
    bow_vocab = BowVocabulary.load(bow_path) if bow_path.exists() else None
    record = {
        "doc_id": "smoke",
        "title": "Graph attention networks for text classification",
        "abstract": "We propose a graph attention network for text classification. The model uses self-attention and training data augmentation.",
        "full_text": "",
        "keyphrases": ["graph attention network", "text classification", "self-attention"],
    }
    feature = build_features(record, tokenizer, FeatureConfig(max_seq_length=128, max_sentences=8), bow_vocab)
    batch = collate_features([feature], tokenizer.pad_token_id or 0)
    model = ScientificKeyphraseExtractor(ModelConfig(model_name=model_name, sentence_context_layers=0))
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        token_type_ids=batch["token_type_ids"],
        cls_positions=batch["cls_positions"],
        sentence_mask=batch["sentence_mask"],
        sentence_labels=batch["sentence_labels"],
        bio_labels=batch["bio_labels"],
    )
    print(json.dumps({"loss": float(outputs["loss"].item()), "ok": True}, indent=2))


if __name__ == "__main__":
    main()


