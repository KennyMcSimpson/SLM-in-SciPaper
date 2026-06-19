# Structure-aware Evidence-grounded Paper Card

This is the V2 line for the scientific keyphrase extractor. The goal is not to
replace the trained keyword extractor. The goal is to make its extracted
phrases useful for paper-level understanding.

## Core Chain

The method is one chain:

```text
long paper -> section structure -> concept candidates -> evidence sentence
-> functional role -> paper card -> five-part summary
```

The existing keyphrase extractor provides concept candidates. The V2 structure
model learns which section a sentence belongs to, what functional role the
sentence plays, whether it is evidence, and how important it is.

## Model Choice

The default encoder remains SciBERT. This is intentional:

- the existing keyphrase model already trained SciBERT for scientific phrase
  boundaries;
- role and evidence prediction are sentence-level scientific understanding
  tasks, not open-ended generation tasks;
- using the same encoder keeps the method explainable and avoids a loose
  "extractor plus LLM" pipeline.

The V2 model adds section embeddings and three sentence heads:

- `Role Head`: predicts background, objective, core_method, result, finding,
  and other paper-card roles.
- `Evidence Head`: predicts whether a sentence supports a selected concept or
  paper-card slot.
- `Importance Head`: scores whether the sentence/concept should enter the card.

QASPER evidence labels are sparse, so the training script automatically computes
an evidence positive-class weight from the train set. This prevents the evidence
head from learning the trivial all-negative solution.

The old BIO boundary head stays in the keyphrase extractor checkpoint. That is
the concept boundary part of the same chain.

## Inference Repair For Local Txt Papers

The local `txt/` papers are often de-punctuated and stored as one long line.
They contain inline headings such as `body introduction`, `background`,
`model architecture`, `training`, `results`, and `conclusion`, instead of clean
line-level section headings. This is why a trained structure model can still
produce a weak paper card: if the sectioner gives the model the wrong section
identity, the role/evidence heads are forced to explain damaged input.

The current inference layer therefore adds three explainable repairs:

- Inline section recovery searches all candidate headings globally, then rejects
  embedded mentions such as abstract-level `results` or sentence-level
  `a model architecture`.
- Section notes are selected with section-role compatibility, so an intro slot
  prefers background/problem/objective evidence rather than high-scoring result
  sentences.
- The verbalizer now writes a longer five-part Chinese paper-card summary from
  the same interpretable bottleneck. It still exposes evidence snippets; it is
  not a hidden generative model.

Validation outputs from this repair:

- `runs/paper_cards/attention_sectionfix.md`
- `runs/paper_cards/attention_sectionfix.json`
- `runs/paper_cards/bert_sectionfix.md`
- `runs/paper_cards/bert_sectionfix.json`

On the Attention paper, the old evidence-fix output had no related-work notes.
The section-fix output has non-empty notes for all five card sections. On the
BERT paper, the same repair separates intro, related work, method, experiment,
and conclusion without retraining the model.

## Batch Inference Quality Repair

After the first section-fix pass, the bottleneck moved from training to local txt
inference quality. A broader batch run over eight local papers showed three
systematic failure modes:

- global keyphrase top-k over-selected long experiment or appendix-like regions,
  leaving some paper-card sections with no concept units;
- terminal material such as `system card`, acknowledgments, references, and
  prompt/example blocks could be folded into conclusion;
- the old Chinese renderer was not a real card verbalizer: it exposed evidence,
  but it also produced unreadable text in some Windows display paths and stitched
  long evidence snippets too aggressively.

The current repair remains inside the same explainable chain and does not change
the trained checkpoints:

- `sectioning.py` now treats local terminal markers conservatively: terminal
  markers stop later appendix-like material only when they are sufficiently late
  or already follow a conclusion-like section. This avoids cutting real
  experiment sections while still removing system-card and appendix tails.
- `sectioning.py` adds method-transition cues such as `our final solution` and
  dense retriever/encoder descriptions, which fixes cases where a method section
  was previously swallowed by experiments.
- `card.py` now extracts a larger candidate pool and performs section-aware
  balanced unit selection. This prevents one long section from monopolizing the
  paper card while preserving global importance as the final ranking signal.
- `card.py` filters low-value configuration phrases and prompt/acknowledgment
  artifacts, e.g. `beam size`, `learning rate`, `redacted website`, contributor
  lists, and red-team participant lists.
- `verbalize.py` was rewritten as a UTF-8 Chinese paper-card renderer. It keeps
  evidence snippets visible but makes the five section summaries read like a
  structured paper understanding card rather than a raw evidence concatenation.
- `verbalize.py` now trims evidence around the selected phrase or cue rather
  than always taking the sentence prefix. This keeps formula-heavy local txt
  snippets from dominating the card.
- `sectioning.py` infers a conservative related-work bridge when a paper has no
  explicit related-work heading but the intro-to-method boundary contains clear
  prior-work cues such as `BM25`, `ORQA`, `existing`, or `additional
  pretraining`.
- `verbalize.py` has a narrow note-term fallback for method/experiment sections
  when the keyword checkpoint misses too many section concepts. It only uses
  evidence notes already selected by the card and prioritizes recognizable
  technical terms such as `sequence-to-sequence transformer architecture`,
  `dense passage retriever`, `self-attention`, and `residual function`.

Current batch output:

- `runs/paper_cards_batch_fixed11/batch_report.md`
- `runs/paper_cards_batch_fixed11/*.md`
- `runs/paper_cards_batch_fixed11/*.json`

Observed fixed11 quality:

- BERT and BART are the cleanest showcase samples. Both reach full five-section
  coverage in the diagnostic report, with BART no longer dominated by generated
  example entities such as `marathon` or `power shutoff`.
- Attention is usable: method concepts center on positional encoding,
  self-attention, attention, and computational complexity; the conclusion slot
  is supported by a stable conclusion note even when no separate conclusion
  concept unit survives.
- DPR now recovers a related-work bridge around BM25/ORQA/pretraining evidence
  and keeps the method focus on dense passage retriever, dense encoding, and
  inner product search. Its local txt still lacks a stable conclusion section,
  so the conclusion slot intentionally stays conservative.
- ResNet is usable but conclusion is note-backed rather than unit-backed because
  the local txt stores the strongest concluding sentence in the abstract-like
  lead, not in an explicit conclusion section.
- GPT-4 technical report and the ELECTRA-named txt should not be used as normal
  showcase samples. GPT-4 is a stress test with technical-report/system-card
  structure; ELECTRA appears mismatched or truncated.

Known limitations after fixed11:

- Papers without explicit related-work, experiment, or conclusion sections still
  produce conservative fallback text for those slots. This is better than
  hallucinating a section from weak evidence.
- The local txt extraction quality still matters. Some files are mismatched or
  partial, such as the ELECTRA-named file that appears to contain a different or
  truncated paper body.
- The card is still extractive and evidence-grounded. It is more readable, but
  it is not an abstractive LLM summary module.

## Dataset Roles

The datasets are not mixed as arbitrary A+B+C components. Each dataset only
supervises the part it genuinely contains.

- PubMed RCT: sentence role supervision. Its labels
  `BACKGROUND/OBJECTIVE/METHODS/RESULTS/CONCLUSIONS` map to the paper-card
  roles `background/objective/process/result/finding`.
- QASPER: evidence supervision. Its answer evidence and highlighted evidence
  mark sentences that support questions about a scientific paper.
- ACLSum: small facet-summary supervision. Its `challenge/approach/outcome`
  facets weakly map to `problem/core_method/result`.
- FacetSum is recorded as a future optional dataset because it is large and
  access-gated, so it is not part of the default one-command core resources.

## Commands

Download and normalize the core structure resources:

```powershell
.\.venv_keyword\Scripts\python.exe .\scientific_keyphrase_extractor\scripts\download_structure_resources.py --core
```

Quick smoke test:

```powershell
.\.venv_keyword\Scripts\python.exe .\scientific_keyphrase_extractor\scripts\smoke_test_structure_v2.py
```

Train the V2 structure heads:

```powershell
.\.venv_keyword\Scripts\python.exe .\scientific_keyphrase_extractor\src\ske\structure\train.py `
  --train_jsonl `
    .\scientific_keyphrase_extractor\data\structure\pubmed_rct\train.jsonl `
    .\scientific_keyphrase_extractor\data\structure\qasper\train.jsonl `
    .\scientific_keyphrase_extractor\data\structure\aclsum\train.jsonl `
  --dev_jsonl `
    .\scientific_keyphrase_extractor\data\structure\pubmed_rct\validation.jsonl `
    .\scientific_keyphrase_extractor\data\structure\qasper\validation.jsonl `
    .\scientific_keyphrase_extractor\data\structure\aclsum\validation.jsonl `
  --output_dir .\scientific_keyphrase_extractor\runs\structure_v2_scibert `
  --model_name .\scientific_keyphrase_extractor\resources\models\allenai_scibert_scivocab_uncased `
  --init_encoder_checkpoint .\scientific_keyphrase_extractor\runs\scibert_semeval2010_finetune_nobow `
  --epochs 3 `
  --batch_size 4 `
  --device cuda `
  --amp
```

The training log prints an `evidence_weight` row before epoch 1. It also reports
evidence precision, recall, best-threshold F1, positive prediction rate, and
positive/negative probability means. Use those values to judge whether evidence
learning is alive instead of looking only at the default `evidence_f1`.

Run paper-card inference:

```powershell
.\.venv_keyword\Scripts\python.exe .\scientific_keyphrase_extractor\scripts\infer_paper_card.py `
  --input_txt .\txt\2017_transformer_and_large_language_models_attention_is_all_you_need.txt `
  --keyword_checkpoint .\scientific_keyphrase_extractor\runs\scibert_semeval2010_finetune_nobow `
  --structured_checkpoint .\scientific_keyphrase_extractor\runs\structure_v2_scibert `
  --output_json .\scientific_keyphrase_extractor\runs\paper_cards\attention.json `
  --output_md .\scientific_keyphrase_extractor\runs\paper_cards\attention.md `
  --device cuda
```

The same inference command can run without `--structured_checkpoint`; in that
case it uses the interpretable rule layer for role and importance, which is
useful for debugging before long training.

Add a one-paragraph overview to an existing paper-card Markdown file:

```powershell
.\.venv_keyword\Scripts\python.exe -B .\scientific_keyphrase_extractor\scripts\add_one_paragraph_overview.py `
  .\scientific_keyphrase_extractor\runs\paper_cards_batch_fixed11\2018_natural_language_processing_bert_pre_training_of_deep_bidirectional_transformers_for_language_understanding.md `
  --output_md .\scientific_keyphrase_extractor\runs\paper_cards_batch_fixed11\2018_natural_language_processing_bert_pre_training_of_deep_bidirectional_transformers_for_language_understanding_with_overview.md
```

Batch diagnostic run:

```powershell
.\.venv_keyword\Scripts\python.exe -B .\scientific_keyphrase_extractor\scripts\evaluate_paper_cards.py `
  --input_dir .\txt `
  --output_dir .\scientific_keyphrase_extractor\runs\paper_cards_batch_fixed11 `
  --keyword_checkpoint .\scientific_keyphrase_extractor\runs\scibert_semeval2010_finetune_nobow `
  --structured_checkpoint .\scientific_keyphrase_extractor\runs\structure_v2_scibert_evidencefix `
  --device cuda `
  --files `
    2017_transformer_and_large_language_models_attention_is_all_you_need.txt `
    2018_natural_language_processing_bert_pre_training_of_deep_bidirectional_transformers_for_language_understanding.txt `
    2019_natural_language_processing_bart_denoising_sequence_to_sequence_pre_training_for_natural_language_generation.txt `
    2020_natural_language_processing_electra_pre_training_text_encoders_as_discriminators_rather_than_generators.txt `
    2014_deep_learning_generative_adversarial_nets.txt `
    2015_computer_vision_deep_residual_learning_for_image_recognition.txt `
    2020_information_retrieval_dense_passage_retrieval_for_open_domain_question_answering.txt `
    2023_transformer_and_large_language_models_gpt_4_technical_report.txt
```
