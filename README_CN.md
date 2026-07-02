# SLM-in-SciPaper

## 数据和模型资源

处理后的训练数据集、词表资源、已上传 checkpoint，以及 178 篇 TXT 论文语料库放在 Hugging Face：

[KennySimpson/SLM-in-SciPaper](https://huggingface.co/datasets/KennySimpson/SLM-in-SciPaper)

这是一个面向科研论文的 NLP 分析系统，目标不是只抽几个关键词，而是从论文全文中生成 Evidence-grounded Concept Units，也就是“带证据句支撑的概念单元”。这些 JSON 单元可以继续交给本地开源模型，用来生成更完整的论文概述。

## 总体流程

```text
论文 TXT / PDF
  -> Stage 1: SciBERT 关键词候选抽取
  -> Stage 2: 带 section 信息的句子 role / evidence / importance 建模
  -> Stage 3: 生成 Evidence-grounded Concept Units
  -> JSON 交给下游本地 overview 模型
```

Stage 3 严格使用可解释公式：

```text
S_candidate(u) = 0.65 S_boundary(u) + 0.25 S_selector(u) + 0.10 S_BoW(u)
S_rerank(u)   = 0.75 S_candidate(u) + 0.25 S_coverage(u)
I_concept(u)  = 0.50 S_rerank(u) + 0.25 S_evidence(u) + 0.25 S_sentence(u)
```

最终筛选过程是：先保留 Top112 候选，再过三类阈值门，最后输出 Top28：

```text
phrase_word_number(u) >= 2
bow_support_score(u) = term_confidence(u) * match_quality(u) >= 0.70
tfidf_support_score(u) = matched_tfidf(u) / max_tfidf(d) >= 0.50
```

只要三个阈值里有一个通过，这个候选就可以进入最终排序。Stage 3 不再使用额外的 downstream relevance score 作为筛选分数。

## 代码结构

- `Model/PaperCard/code/01_keyword_extractor`：第一阶段关键词模型训练和关键词推理。
- `Model/PaperCard/code/02_structure_card`：第二阶段结构模型训练和 smoke test。
- `Model/PaperCard/code/03_inference_summary`：第三阶段 Evidence Units JSON 推理和评估脚本。
- `Model/PaperCard/src/ske`：共享 Python 源码，包括数据处理、模型、section 解析、推理和 schema。
- `frontend`：Vue 前端，浏览器里用 PDF.js 和 ONNX Runtime Web 跑推理。
- `backend`：轻量聊天代理和证据检索可视化支持。
- `frontend/public/models`：浏览器端模型和 Stage 3 资源文件。

## Python 推理

在 `Model/PaperCard` 下运行：

```powershell
python .\code\03_inference_summary\infer_paper_card.py `
  --input_txt .\datasets\03_demo_txt\full_library\2017_transformer_and_large_language_models_attention_is_all_you_need.txt `
  --output_json .\outputs\evidence_units_attention.json `
  --device cuda
```

如果本地有默认 checkpoint 和 `datasets/lexicon`，脚本会自动读取，不需要手动指定每个 CSV。

## 浏览器 Demo

在 `frontend` 下运行：

```powershell
npm install
npm run build
npm run dev
```

浏览器 worker 会加载：

- `keyword_extractor_int8.onnx.gz`
- `structure_model_int8.onnx.gz`
- `keyword_vocab.json`
- `structure_vocab.json`
- `stage3_resources.json`

上传 PDF 后，前端会在浏览器内完成 PDF 文本抽取、Stage 1 ONNX 推理、Stage 2 ONNX 推理和 Stage 3 公式化概念单元生成。

## 输出内容

主要输出是 Evidence-grounded Concept Units JSON。每个 concept unit 包含：

- 概念短语和支撑证据句；
- 所属 section 和 role 标签；
- `S_boundary`、`S_selector`、`S_BoW`、`S_candidate`、`S_coverage`、`S_rerank`；
- `S_evidence`、`S_sentence` 和 `I_concept`；
- n-gram、BoW、TF-IDF 三个 Stage 3 阈值的 trace。

英文版见 [README.md](README.md)。
