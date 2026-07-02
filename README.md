# SLM-in-SciPaper

Scientific paper analysis system for extracting evidence-grounded concept units from full-text papers. The project is organized around a three-stage NLP pipeline and a browser demo that runs the exported ONNX models locally.

## Data and Model Assets

The processed training datasets, supplementary lexicon resources, uploaded checkpoints, and the 178-paper TXT corpus are hosted on Hugging Face:

[KennySimpson/SLM-in-SciPaper](https://huggingface.co/datasets/KennySimpson/SLM-in-SciPaper)

## Pipeline

```text
Paper text / PDF
  -> Stage 1: SciBERT keyphrase candidate extraction
  -> Stage 2: section-aware sentence role, evidence, and importance modeling
  -> Stage 3: evidence-grounded concept unit generation
  -> JSON for downstream local overview generation
```

Stage 3 uses transparent report-defined formulas:

```text
S_candidate(u) = 0.65 S_boundary(u) + 0.25 S_selector(u) + 0.10 S_BoW(u)
S_rerank(u)   = 0.75 S_candidate(u) + 0.25 S_coverage(u)
I_concept(u)  = 0.50 S_rerank(u) + 0.25 S_evidence(u) + 0.25 S_sentence(u)
```

The final Stage 3 filter keeps a Top112 candidate pool, applies three feature gates, then writes the Top28 concept units:

```text
phrase_word_number(u) >= 2
bow_support_score(u) = term_confidence(u) * match_quality(u) >= 0.70
tfidf_support_score(u) = matched_tfidf(u) / max_tfidf(d) >= 0.50
```

A candidate passes if any one of the three gates passes. No extra downstream relevance score is used for Stage 3 selection.

## Repository Layout

- `Model/PaperCard/code/01_keyword_extractor`: Stage 1 training and keyword inference entry points.
- `Model/PaperCard/code/02_structure_card`: Stage 2 structure model training and smoke tests.
- `Model/PaperCard/code/03_inference_summary`: Stage 3 evidence-unit JSON inference and evaluation scripts.
- `Model/PaperCard/src/ske`: shared Python package for data processing, models, sectioning, inference, and schema code.
- `frontend`: Vue browser demo using PDF.js and ONNX Runtime Web.
- `backend`: lightweight chat proxy and retrieval visualization support for the browser demo.
- `frontend/public/models`: exported browser models and Stage 3 resources.

## Python Inference

Run from `Model/PaperCard`:

```powershell
python .\code\03_inference_summary\infer_paper_card.py `
  --input_txt .\datasets\03_demo_txt\full_library\2017_transformer_and_large_language_models_attention_is_all_you_need.txt `
  --output_json .\outputs\evidence_units_attention.json `
  --device cuda
```

The script uses the default checkpoints and lexicon resources under `Model/PaperCard/models` and `Model/PaperCard/datasets/lexicon` when they are available.

## Browser Demo

Run the frontend from `frontend`:

```powershell
npm install
npm run build
npm run dev
```

The browser worker loads:

- `keyword_extractor_int8.onnx.gz`
- `structure_model_int8.onnx.gz`
- `keyword_vocab.json`
- `structure_vocab.json`
- `stage3_resources.json`

The worker performs PDF text extraction, Stage 1 ONNX inference, Stage 2 ONNX inference, and Stage 3 formula-based concept-unit generation in the browser.

## Output

The main output is Evidence-grounded Concept Units JSON. Each unit contains:

- concept phrase and evidence sentence;
- section and role label;
- `S_boundary`, `S_selector`, `S_BoW`, `S_candidate`, `S_coverage`, `S_rerank`;
- `S_evidence`, `S_sentence`, and `I_concept`;
- Stage 3 threshold trace for n-gram, BoW, and TF-IDF support.

See [README_CN.md](README_CN.md) for the Chinese version.
