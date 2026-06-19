# Cleanup Decision

整理时间：2026-06-19

## Result

旧实验工程、旧数据缓存、旧 runner 脚本已经从 `E:\class\2026.4\NLP` 根目录删除。当前最终工程是：

`E:\class\2026.4\NLP\scientific_paper_card_final`

## Kept In Final Folder

- 代码主线：`src/ske`、三块 `code/*` 入口、`configs`、`docs`。
- 训练数据：keyword processed JSONL、structure processed JSONL、PubMed RCT 小型原始 txt。
- 模型：本地 SciBERT、最终 keyword checkpoint、最终 structure checkpoint、KP20k/LDKP10k 阶段记录。
- 输出：`paper_cards_batch_fixed11`、showcase 文件、verification Attention 输出。

## Deleted From Old Workspace

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

## Preserved Outside Final Folder

- `.venv_keyword`: 当前 Python 环境，最终 README 中的命令依赖它。
- `.codex-memory`: 项目记忆和交接记录。
- 课程 PDF/DOCX/PPTX：这些是课程资料，不属于模型试验残留。

## Verification

最终文件夹清理后已经验证：

- 三块代码入口的 `--help` 正常。
- 所有 `src` 和 `code` Python 文件 AST 解析正常。
- Attention txt 推理生成 `outputs/verification_attention.md/json`。
- overview 后处理生成 `outputs/verification_attention_with_overview.md`。
