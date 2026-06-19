# 01 Keyword Extractor

这一块只负责从科学论文中抽取 present keyphrases。它包含 SciBERT encoder、BIO boundary head、sentence selector，以及 coverage rerank。这里不负责生成论文概述。

常用入口：

```powershell
..\.venv_keyword\Scripts\python.exe .\code\01_keyword_extractor\train_keyword.py --help
..\.venv_keyword\Scripts\python.exe .\code\01_keyword_extractor\infer_keywords.py --help
```

数据位置：`datasets/01_keyword_keyphrase/processed`。

最终 checkpoint：`models/checkpoints/keyword_scibert_semeval2010_finetune_nobow`。
