# Final Project Layout

## Why This Folder Is Organized This Way

这个项目现在有一条主线：关键词不是最终答案，只是概念候选；结构模型把候选放回论文的 intro、related work、method、experiment、conclusion 中，再通过证据句生成 paper card。因此最终文件夹按功能分成三块，而不是按“哪个脚本先写出来”分。

## Dataset Blocks

- `datasets/01_keyword_keyphrase/processed`: 训练关键词边界和句子 selector。KP20k 负责大规模 warm-up，LDKP10k 负责长文档适配，SemEval2010 负责科学关键词小数据 fine-tune/evaluation。
- `datasets/02_structure_card/processed`: 训练结构模型。PubMed RCT 给 sentence role，QASPER 给 evidence sentence，ACLSum 用 facet summaries 构造 ROUGE-L extractive oracle 来监督 importance。
- `datasets/03_demo_txt/full_library`: 本地论文 txt，只用于推理展示，不参与训练。

## Code Blocks

- `code/01_keyword_extractor`: present-keyphrase extractor。
- `code/02_structure_card`: role/evidence/importance heads。
- `code/03_inference_summary`: paper card 和 overview 输出。

`src/ske` 是共享源码包。保留它是为了让三个功能块共用同一套 schema、tokenizer 处理、SciBERT 加载和论文卡片数据结构；强行复制三份会让模型接口更乱。

## Kept Model Artifacts

- `models/checkpoints/keyword_scibert_semeval2010_finetune_nobow`: 最终关键词模型。
- `models/checkpoints/structure_v4_partial_role_balanced_fulldev`: 当前结构模型。
- `models/training_history/keyword_1_kp20k_warmup`: 第一阶段训练记录和 checkpoint。
- `models/training_history/keyword_2_ldkp10k_finetune_nobow_b4_e1`: 第二阶段训练记录和 checkpoint。

## Not Included

没有把 HuggingFace `.arrow` 原始缓存作为主数据搬入最终工程。原因是训练实际读取的是 processed JSONL；原始缓存体积大，且不利于展示项目主线。需要重新下载时，用 `code/01_keyword_extractor/download_keyword_resources.py` 和 `code/02_structure_card/download_structure_resources.py`。
