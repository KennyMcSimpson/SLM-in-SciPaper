# Scientific Keyphrase Extractor

This folder is the new keyphrase-extraction line for the NLP project.

The design is based on three ideas:

- SEG-Net style long-document processing: select keyword-evidence sentences before extracting keyphrases.
- Scientific encoder: use SciBERT as the default encoder because the target documents are scientific / technical.
- Set-aware final selection: extract candidate phrases, canonicalize aliases, then rerank with coverage to avoid near-duplicate keywords.

This is not a continuation of the old SciTLDR sentence-ranker objective. SciTLDR is useful only as selector warm-up and long-document noise training. The main supervision should come from datasets with real keyphrases.

## Resource Plan

Core resources:

- `allenai/scibert_scivocab_uncased`: scientific encoder.
- `midas/kp20k`: large scientific abstract keyphrase warm-up.
- `midas/ldkp10k`: long scientific document keyphrase training.
- `midas/semeval2010`: scientific keyphrase evaluation / fine-tuning.
- Local SciTLDR: selector warm-up and long-document noise.
- Friend BoW vocabulary: alias-canonical dictionary, weak selector labels, hard negatives, canonicalized inference.

Optional resources:

- `midas/inspec`, `midas/krapivin`, `midas/nus`: smaller scientific keyphrase evaluation sets.
- `midas/openkp`: open-domain web keyphrases, useful only if we later want domain-general robustness.

## Quick Start

From `E:\class\2026.4\NLP`:

```powershell
.\.venv_keyword\Scripts\python.exe .\scientific_keyphrase_extractor\scripts\download_resources.py --core
.\.venv_keyword\Scripts\python.exe .\scientific_keyphrase_extractor\scripts\download_resources.py --core --normalize_only
.\.venv_keyword\Scripts\python.exe .\scientific_keyphrase_extractor\scripts\smoke_test.py
```

`--normalize_only` rebuilds processed JSONL from the already downloaded raw datasets. Use it after changing normalization; it does not re-download KP20k, LDKP10k, or SemEval2010.

Quick training-chain smoke test:

```powershell
.\.venv_keyword\Scripts\python.exe .\scientific_keyphrase_extractor\src\ske\train.py `
  --train_jsonl .\scientific_keyphrase_extractor\data\processed\kp20k\train.jsonl `
  --dev_jsonl .\scientific_keyphrase_extractor\data\processed\kp20k\validation.jsonl `
  --output_dir .\scientific_keyphrase_extractor\runs\tiny_train_smoke `
  --model_name hf-internal-testing/tiny-random-bert `
  --epochs 1 `
  --batch_size 2 `
  --max_train_records 8 `
  --max_dev_records 4 `
  --device cpu
```

Stage 1: KP20k warm-up with local SciBERT:

```powershell
.\.venv_keyword\Scripts\python.exe .\scientific_keyphrase_extractor\src\ske\train.py `
  --train_jsonl .\scientific_keyphrase_extractor\data\processed\kp20k\train.jsonl `
  --dev_jsonl .\scientific_keyphrase_extractor\data\processed\kp20k\validation.jsonl `
  --output_dir .\scientific_keyphrase_extractor\runs\scibert_kp20k_warmup `
  --model_name .\scientific_keyphrase_extractor\resources\models\allenai_scibert_scivocab_uncased `
  --epochs 3 `
  --batch_size 4 `
  --device cuda `
  --amp
```

Stage 2: LDKP10k long-document fine-tuning:

```powershell
.\.venv_keyword\Scripts\python.exe .\scientific_keyphrase_extractor\src\ske\train.py `
  --train_jsonl .\scientific_keyphrase_extractor\data\processed\ldkp10k\train.jsonl `
  --dev_jsonl .\scientific_keyphrase_extractor\data\processed\ldkp10k\validation.jsonl `
  --output_dir .\scientific_keyphrase_extractor\runs\scibert_ldkp10k_finetune `
  --init_checkpoint .\scientific_keyphrase_extractor\runs\scibert_kp20k_warmup `
  --bow_csv .\scientific_keyphrase_extractor\resources\vocab\final_bow_vocabulary.csv `
  --epochs 3 `
  --batch_size 2 `
  --device cuda `
  --amp
```

Stage 3: SemEval2010 small-data fine-tuning / evaluation:

```powershell
.\.venv_keyword\Scripts\python.exe .\scientific_keyphrase_extractor\src\ske\train.py `
  --train_jsonl .\scientific_keyphrase_extractor\data\processed\semeval2010\train.jsonl `
  --dev_jsonl .\scientific_keyphrase_extractor\data\processed\semeval2010\test.jsonl `
  --output_dir .\scientific_keyphrase_extractor\runs\scibert_semeval2010_finetune `
  --init_checkpoint .\scientific_keyphrase_extractor\runs\scibert_ldkp10k_finetune `
  --bow_csv .\scientific_keyphrase_extractor\resources\vocab\final_bow_vocabulary.csv `
  --epochs 10 `
  --batch_size 2 `
  --device cuda `
  --amp
```

## Why These Modules Exist

The sentence selector exists because long documents contain many irrelevant sentences. It should learn keyword evidence, not TLDR-style summary labels.

The BIO boundary head exists because keyword extraction needs explicit phrase boundaries. Sentence classification alone cannot tell which span is the keyword.

The dictionary / canonicalization layer exists because the friend's BoW vocabulary contains useful alias groups such as `training data`, `training dataset`, and `training-set` mapping to `training set`.

The coverage reranker exists because naive top-k extraction returns duplicates such as `attention`, `self-attention`, and `attention mechanism`.

The evidence-aware sentence packing exists because LDKP10k and SemEval2010 are long documents. If training only reads the first 512 tokens, many real keyphrase spans are never seen by the BIO boundary head.

## V2: Structure-aware Paper Card

The next line turns extracted keyphrases into an evidence-grounded paper card.
It does not replace the trained keyphrase extractor. It adds one structure-aware
model on top of the same SciBERT backbone:

```text
long paper -> section structure -> concept candidates -> evidence sentence
-> functional role -> paper card -> five-part summary
```

See `docs/STRUCTURED_PAPER_CARD.md` for the full design and commands.

Download and normalize the V2 structure datasets:

```powershell
.\.venv_keyword\Scripts\python.exe .\scientific_keyphrase_extractor\scripts\download_structure_resources.py --core
```

Run the V2 smoke test:

```powershell
.\.venv_keyword\Scripts\python.exe .\scientific_keyphrase_extractor\scripts\smoke_test_structure_v2.py
```

Train the V2 structure heads:

```powershell
.\.venv_keyword\Scripts\python.exe .\scientific_keyphrase_extractor\src\ske\structure\train.py `
  --train_jsonl `
    .\scientific_keyphrase_extractor\data\structure\pubmed_rct\train.jsonl `
    .\scientific_keyphrase_extractor\data\structure\qasper\train.jsonl `
    .\scientific_keyphrase_extractor\data\structure\aclsum\train.jsonl `
  --dev_jsonl `
    .\scientific_keyphrase_extractor\data\structure\pubmed_rct\validation.jsonl `
    .\scientific_keyphrase_extractor\data\structure\qasper\validation.jsonl `
    .\scientific_keyphrase_extractor\data\structure\aclsum\validation.jsonl `
  --output_dir .\scientific_keyphrase_extractor\runs\structure_v2_scibert `
  --model_name .\scientific_keyphrase_extractor\resources\models\allenai_scibert_scivocab_uncased `
  --init_encoder_checkpoint .\scientific_keyphrase_extractor\runs\scibert_semeval2010_finetune_nobow `
  --epochs 3 `
  --batch_size 4 `
  --device cuda `
  --amp
```

Build a paper card from a local txt:

```powershell
.\.venv_keyword\Scripts\python.exe .\scientific_keyphrase_extractor\scripts\infer_paper_card.py `
  --input_txt .\txt\2017_transformer_and_large_language_models_attention_is_all_you_need.txt `
  --keyword_checkpoint .\scientific_keyphrase_extractor\runs\scibert_semeval2010_finetune_nobow `
  --structured_checkpoint .\scientific_keyphrase_extractor\runs\structure_v2_scibert `
  --output_json .\scientific_keyphrase_extractor\runs\paper_cards\attention.json `
  --output_md .\scientific_keyphrase_extractor\runs\paper_cards\attention.md `
  --device cuda
```
