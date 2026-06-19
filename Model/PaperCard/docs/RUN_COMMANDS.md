# Run Commands

以下命令默认在最终文件夹执行：

```powershell
cd E:\class\2026.4\NLP\scientific_paper_card_final
```

## 1. 关键词模型三阶段训练

### KP20k warm-up

```powershell
..\.venv_keyword\Scripts\python.exe .\code\01_keyword_extractor\train_keyword.py `
  --train_jsonl .\datasets\01_keyword_keyphrase\processed\kp20k\train.jsonl `
  --dev_jsonl .\datasets\01_keyword_keyphrase\processed\kp20k\validation.jsonl `
  --output_dir .\runs\keyword_1_kp20k_warmup `
  --model_name .\models\base_scibert\allenai_scibert_scivocab_uncased `
  --epochs 3 `
  --batch_size 4 `
  --device cuda `
  --amp
```

### LDKP10k fine-tune

```powershell
..\.venv_keyword\Scripts\python.exe .\code\01_keyword_extractor\train_keyword.py `
  --train_jsonl .\datasets\01_keyword_keyphrase\processed\ldkp10k\train.jsonl `
  --dev_jsonl .\datasets\01_keyword_keyphrase\processed\ldkp10k\validation.jsonl `
  --output_dir .\runs\keyword_2_ldkp10k_finetune_nobow_b4_e1 `
  --init_checkpoint .\runs\keyword_1_kp20k_warmup `
  --epochs 3 `
  --batch_size 4 `
  --device cuda `
  --amp
```

### SemEval2010 final fine-tune

```powershell
..\.venv_keyword\Scripts\python.exe .\code\01_keyword_extractor\train_keyword.py `
  --train_jsonl .\datasets\01_keyword_keyphrase\processed\semeval2010\train.jsonl `
  --dev_jsonl .\datasets\01_keyword_keyphrase\processed\semeval2010\test.jsonl `
  --output_dir .\runs\keyword_3_semeval2010_finetune_nobow `
  --init_checkpoint .\runs\keyword_2_ldkp10k_finetune_nobow_b4_e1 `
  --epochs 10 `
  --batch_size 4 `
  --device cuda `
  --amp
```

## 2. 结构模型训练

```powershell
..\.venv_keyword\Scripts\python.exe .\code\02_structure_card\train_structure.py `
  --train_jsonl `
    .\datasets\02_structure_card\processed\pubmed_rct\train.jsonl `
    .\datasets\02_structure_card\processed\qasper\train.jsonl `
    .\datasets\02_structure_card\processed\aclsum\train.jsonl `
  --dev_jsonl `
    .\datasets\02_structure_card\processed\pubmed_rct\validation.jsonl `
    .\datasets\02_structure_card\processed\qasper\validation.jsonl `
    .\datasets\02_structure_card\processed\aclsum\validation.jsonl `
  --output_dir .\runs\structure_v2_scibert_evidencefix `
  --model_name .\models\base_scibert\allenai_scibert_scivocab_uncased `
  --init_encoder_checkpoint .\models\checkpoints\keyword_scibert_semeval2010_finetune_nobow `
  --epochs 3 `
  --batch_size 4 `
  --device cuda `
  --amp
```

## 3. Evidence Units JSON 推理

```powershell
..\.venv_keyword\Scripts\python.exe .\code\03_inference_summary\infer_paper_card.py `
  --input_txt .\datasets\03_demo_txt\full_library\2017_transformer_and_large_language_models_attention_is_all_you_need.txt `
  --keyword_checkpoint .\models\checkpoints\keyword_scibert_semeval2010_finetune_nobow `
  --structured_checkpoint .\models\checkpoints\structure_v2_scibert_evidencefix `
  --output_json .\outputs\evidence_units_attention.json `
  --device cuda
```

第三阶段默认会加载 `datasets/lexicon` 里的 enriched section BoW、document-term frequency/TF-IDF matrix、evidence cue lexicon 和 sentence evidence candidate pool。需要替换资源时再手动传 `--section_bow_csv`、`--term_frequency_matrix_csv`、`--term_tfidf_matrix_csv`、`--evidence_cue_csv`、`--sentence_evidence_csv`。

旧版 Markdown 展示仍可通过额外添加 `--output_md .\outputs\demo_attention.md` 生成，但不再是第三阶段默认主线。

批量诊断：

```powershell
..\.venv_keyword\Scripts\python.exe .\code\03_inference_summary\evaluate_paper_cards.py `
  --input_dir .\datasets\03_demo_txt\full_library `
  --output_dir .\outputs\paper_cards_batch_new `
  --keyword_checkpoint .\models\checkpoints\keyword_scibert_semeval2010_finetune_nobow `
  --structured_checkpoint .\models\checkpoints\structure_v2_scibert_evidencefix `
  --files `
    2017_transformer_and_large_language_models_attention_is_all_you_need.txt `
    2018_natural_language_processing_bert_pre_training_of_deep_bidirectional_transformers_for_language_understanding.txt `
    2019_natural_language_processing_bart_denoising_sequence_to_sequence_pre_training_for_natural_language_generation.txt `
  --device cuda
```
