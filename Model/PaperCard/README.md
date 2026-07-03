# PaperCard Model Pipeline

This folder contains the Python implementation of the three-stage Evidence Unit extraction pipeline.

## Stages

```text
TXT paper
  -> section recovery
  -> Stage 1 SciBERT keyphrase extractor
  -> Stage 2 section-aware role / evidence / importance model
  -> Stage 3 Evidence-grounded Concept Units JSON
```

Stage 1 extracts present keyphrase candidates with a SciBERT encoder, sentence selector, BIO boundary head, and coverage-aware reranking.

Stage 2 reads section-aware sentence windows and predicts three sentence-level signals: rhetorical role, evidence probability, and sentence importance.

Stage 3 fuses Stage 1 candidates with Stage 2 sentence signals, applies the report-defined n-gram, BoW, and TF-IDF gates, and writes JSON concept units for a downstream local overview model. It does not generate Markdown cards or final natural-language overviews.

## Assets

Training datasets, restored checkpoints, lexicon files, and the TXT paper corpus are stored outside GitHub on Hugging Face:

[KennySimpson/SLM-in-SciPaper](https://huggingface.co/datasets/KennySimpson/SLM-in-SciPaper)

Expected local asset locations after restoration:

- `datasets/01_keyword_keyphrase/processed`: KP20k, LDKP10k, and SemEval2010 JSONL files for Stage 1.
- `datasets/02_structure_card/processed`: PubMed RCT, QASPER, and ACLSum JSONL files for Stage 2.
- `datasets/03_demo_txt/full_library`: TXT paper corpus for inference demos.
- `datasets/lexicon`: enriched section BoW, document-term frequency/TF-IDF matrices, evidence cue lexicon, and sentence evidence candidates.
- `models/base_scibert`: local SciBERT encoder files.
- `models/checkpoints`: final Stage 1 and Stage 2 PyTorch checkpoints.

## Main Entry Points

```powershell
python .\code\01_keyword_extractor\train_keyword.py --help
python .\code\02_structure_card\train_structure.py --help
python .\code\03_inference_summary\infer_paper_card.py --help
python .\code\03_inference_summary\batch_infer_evidence_units.py --help
```

Run a single TXT paper:

```powershell
python .\code\03_inference_summary\infer_paper_card.py `
  --input_txt .\datasets\03_demo_txt\full_library\2017_transformer_and_large_language_models_attention_is_all_you_need.txt `
  --output_json .\outputs\evidence_units_attention.json `
  --device cuda
```

Run a TXT folder:

```powershell
python .\code\03_inference_summary\batch_infer_evidence_units.py `
  --input_dir .\datasets\03_demo_txt\full_library `
  --output_dir .\outputs\evidence_units_batch `
  --device cuda
```

See `docs/RUN_COMMANDS.md` for full training and inference commands.
