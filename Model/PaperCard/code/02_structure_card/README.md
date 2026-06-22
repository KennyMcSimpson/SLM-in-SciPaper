# 02 Structure Card

这一块负责论文结构理解：section-aware SciBERT encoder 加 role/evidence/importance 三个句子级 head。它把关键词候选变成有证据位置、有论文角色的 concept unit。

当前训练分工：

- PubMed RCT 提供 hard role 监督；
- QASPER 监督 evidence 0/1，同时只提供 section-level role candidate mask；
- ACLSum 使用 facet summaries 构造 ROUGE-L extractive oracle，监督 importance；facet role 明确时才提供 hard role；
- 训练时使用 uncertainty weighting 自动平衡三个任务，避免手写任务 loss 权重。

常用入口：

```powershell
..\.venv_keyword\Scripts\python.exe .\code\02_structure_card\train_structure.py --help
```

数据位置：`datasets/02_structure_card/processed`。

当前 checkpoint：`models/checkpoints/structure_v4_partial_role_balanced_fulldev`。
