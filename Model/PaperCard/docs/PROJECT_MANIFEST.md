# Project Manifest

整理时间：2026-06-19

## Final Folder

`E:\class\2026.4\NLP\scientific_paper_card_final`

这是当前 NLP 科研项目的唯一最终工程文件夹。旧实验工程、旧 runs、旧数据缓存和旧 runner 已从 `E:\class\2026.4\NLP` 根目录清理。

## Preserved Assets

- Code: `src/ske` plus three functional blocks under `code/`.
- Keyword datasets: `datasets/01_keyword_keyphrase/processed`.
- Structure datasets: `datasets/02_structure_card/processed` plus PubMed RCT raw txt.
- Demo papers: `datasets/03_demo_txt/full_library`.
- Base encoder: `models/base_scibert/allenai_scibert_scivocab_uncased`.
- Final keyword checkpoint: `models/checkpoints/keyword_scibert_semeval2010_finetune_nobow`.
- Final structure checkpoint: `models/checkpoints/structure_v2_scibert_evidencefix`.
- Training history: `models/training_history` and `outputs/training_metrics`.
- Final outputs: `outputs/final_paper_cards_fixed11` and `outputs/showcase`.

## Verification

Completed after cleanup:

- Python AST syntax check passed for all `src` and `code` Python files.
- `train_keyword.py --help` passed.
- `train_structure.py --help` passed.
- `infer_paper_card.py --help` passed.
- Attention paper inference passed from the final folder.
- One-paragraph overview postprocessor passed from the final folder.

Verification outputs:

- `outputs/verification_attention.json`
- `outputs/verification_attention.md`
- `outputs/verification_attention_with_overview.md`

## Cleanup Performed

Deleted from old workspace after final-folder verification:

- `scientific_keyphrase_extractor`
- `document_ranker`
- `sentence_aware_keyphrase`
- `experiment_summary`
- `other`
- root `data`
- root `datasets`
- root `txt`
- root `__pycache__`
- `run_aic_comparison.py`
- `run_all_experiments.py`
- `run_chunk_aic_comparison.py`
- `run_pu_hard_aic_experiment.py`
- `nlp_classify.py`

Protected and preserved:

- `.venv_keyword`: needed to run the final project.
- `.codex-memory`: project memory and handoff state.
- course PDFs/DOCX/PPTX: course materials, not experiment residue.
