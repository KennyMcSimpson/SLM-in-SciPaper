# Run Commands

Run commands from `Model/PaperCard` unless noted otherwise.

## Stage 1: Keyword Extractor

Download and normalize the core keyphrase datasets:

```powershell
python .\code\01_keyword_extractor\download_keyword_resources.py --core
```

Train the KP20k warm-up model:

```powershell
python .\code\01_keyword_extractor\train_keyword.py `
  --train_jsonl .\datasets\01_keyword_keyphrase\processed\kp20k\train.jsonl `
  --dev_jsonl .\datasets\01_keyword_keyphrase\processed\kp20k\validation.jsonl `
  --output_dir .\runs\keyword_1_kp20k_warmup `
  --model_name .\models\base_scibert\allenai_scibert_scivocab_uncased `
  --epochs 3 `
  --batch_size 4 `
  --device cuda `
  --amp
```

Fine-tune on LDKP10k:

```powershell
python .\code\01_keyword_extractor\train_keyword.py `
  --train_jsonl .\datasets\01_keyword_keyphrase\processed\ldkp10k\train.jsonl `
  --dev_jsonl .\datasets\01_keyword_keyphrase\processed\ldkp10k\validation.jsonl `
  --output_dir .\runs\keyword_2_ldkp10k_finetune_nobow `
  --init_checkpoint .\runs\keyword_1_kp20k_warmup `
  --epochs 1 `
  --batch_size 4 `
  --device cuda `
  --amp
```

Fine-tune on SemEval2010:

```powershell
python .\code\01_keyword_extractor\train_keyword.py `
  --train_jsonl .\datasets\01_keyword_keyphrase\processed\semeval2010\train.jsonl `
  --dev_jsonl .\datasets\01_keyword_keyphrase\processed\semeval2010\test.jsonl `
  --output_dir .\models\checkpoints\keyword_scibert_semeval2010_finetune_nobow `
  --init_checkpoint .\runs\keyword_2_ldkp10k_finetune_nobow `
  --epochs 10 `
  --batch_size 4 `
  --device cuda `
  --amp
```

## Stage 2: Structure Model

Download and normalize PubMed RCT, QASPER, and ACLSum:

```powershell
python .\code\02_structure_card\download_structure_resources.py --core
```

Train the current v4 structure checkpoint:

```powershell
python .\code\02_structure_card\train_structure.py `
  --train_jsonl `
    .\datasets\02_structure_card\processed\pubmed_rct\train.jsonl `
    .\datasets\02_structure_card\processed\qasper\train.jsonl `
    .\datasets\02_structure_card\processed\aclsum\train.jsonl `
  --dev_jsonl `
    .\datasets\02_structure_card\processed\pubmed_rct\validation.jsonl `
    .\datasets\02_structure_card\processed\qasper\validation.jsonl `
    .\datasets\02_structure_card\processed\aclsum\validation.jsonl `
  --output_dir .\models\checkpoints\structure_v4_partial_role_balanced_fulldev `
  --model_name .\models\base_scibert\allenai_scibert_scivocab_uncased `
  --init_encoder_checkpoint .\models\checkpoints\keyword_scibert_semeval2010_finetune_nobow `
  --epochs 3 `
  --batch_size 4 `
  --device cuda `
  --amp `
  --loss_weighting uncertainty
```

## Stage 3: Evidence Unit JSON

Run one TXT file:

```powershell
python .\code\03_inference_summary\infer_paper_card.py `
  --input_txt .\datasets\03_demo_txt\full_library\2017_transformer_and_large_language_models_attention_is_all_you_need.txt `
  --output_json .\outputs\evidence_units_attention.json `
  --device cuda
```

Run a folder:

```powershell
python .\code\03_inference_summary\batch_infer_evidence_units.py `
  --input_dir .\datasets\03_demo_txt\full_library `
  --output_dir .\outputs\evidence_units_batch `
  --device cuda
```

Stage 3 automatically uses the lexicon resources in `datasets/lexicon` when they exist. Pass `--section_bow_csv`, `--term_frequency_matrix_csv`, `--term_tfidf_matrix_csv`, `--evidence_cue_csv`, or `--sentence_evidence_csv` only when overriding those defaults.
