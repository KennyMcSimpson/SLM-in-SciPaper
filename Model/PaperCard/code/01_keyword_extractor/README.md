# 01 Keyword Extractor

This module contains Stage 1: present-keyphrase extraction for scientific papers.

It trains and runs a SciBERT-based model with:

- a shared SciBERT encoder;
- a sentence selector head for keyphrase-bearing sentences;
- a BIO boundary head for phrase spans;
- coverage-aware reranking for duplicate control.

Common entry points:

```powershell
python .\code\01_keyword_extractor\download_keyword_resources.py --help
python .\code\01_keyword_extractor\train_keyword.py --help
python .\code\01_keyword_extractor\infer_keywords.py --help
python .\code\01_keyword_extractor\smoke_test_keyword.py
```

Expected processed data location:

```text
datasets/01_keyword_keyphrase/processed
```

Default final checkpoint:

```text
models/checkpoints/keyword_scibert_semeval2010_finetune_nobow
```
