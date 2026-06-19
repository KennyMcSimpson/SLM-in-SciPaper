# 02 Structure Card

这一块负责论文结构理解：section-aware SciBERT encoder 加 role/evidence/importance 三个句子级 head。它把关键词候选变成有证据位置、有论文角色的 concept unit。

常用入口：

```powershell
..\.venv_keyword\Scripts\python.exe .\code\02_structure_card\train_structure.py --help
```

数据位置：`datasets/02_structure_card/processed`。

最终 checkpoint：`models/checkpoints/structure_v2_scibert_evidencefix`。
