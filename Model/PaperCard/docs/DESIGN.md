# Design: SEG-Style Scientific Keyphrase Extractor

## Research Position

The previous project line learned sentence ranking from SciTLDR labels. That was useful diagnostically, but it did not directly supervise keyphrases. The new line treats keyphrase extraction as a phrase-level task.

## Core Paper Support

- SEG-Net, ACL-IJCNLP 2021: select salient sentences before extracting/generating keyphrases from long documents.
- One2Set, ACL-IJCNLP 2021: keyphrases are an unordered set, so final selection should avoid order bias and duplicates.
- SciBERT, EMNLP-IJCNLP 2019: scientific-domain encoder.
- EMNLP 2023 scholarly keyphrase boundary classification: phrase boundaries need explicit modeling.

## Proposed Pipeline

1. Sentence segmentation.
2. SciBERT encoding with sentence `[CLS]` positions.
3. Sentence selector predicts keyword-evidence sentences.
4. BIO boundary head predicts keyphrase spans.
5. Candidate spans are canonicalized through the BoW vocabulary.
6. Coverage-aware reranking selects non-duplicate final keyphrases.

## Data Roles

- KP20k: large-scale keyphrase warm-up. The MIDAS version provides `document + doc_bio_tags`, so present keyphrases are reconstructed from BIO spans.
- LDKP10k: long-document keyphrase training. `sections/sec_text` become title, abstract, and full text; explicit extractive/abstractive keyphrases are merged with any section BIO spans.
- SemEval2010 / Inspec / Krapivin / NUS: evaluation and fine-tuning. SemEval2010 is also BIO-derived in the current core setup.
- SciTLDR: selector warm-up only, not main keyphrase supervision.
- BoW vocabulary: alias-canonical alignment, weak evidence, hard-negative / duplicate control.

## Training Chain

1. Warm up on KP20k so the encoder and BIO head first learn scientific keyphrase boundaries from many short documents.
2. Fine-tune on LDKP10k so the selector learns long-document evidence and the extractor sees section-level scientific writing.
3. Fine-tune / evaluate on SemEval2010 as the smaller scientific benchmark.

Evidence-aware sentence packing is used during training. It first scans candidate sentences for keyphrase/BIO evidence, then packs positive evidence sentences, neighbors, lead sentences, and a small set of negatives into the 512-token encoder budget. This is necessary because a plain first-512-token window can miss long-document keyphrase spans.

## Main Risks

- Datasets differ in format and keyphrase annotation style.
- Exact span matching misses absent keyphrases and paraphrases.
- BoW vocabulary is useful but small and contains noisy wiki mappings.
- Evidence-aware packing uses gold keyphrases during training only; inference must scan text with windows instead of using gold evidence.
- A first model should prioritize present keyphrase extraction; absent generation can be added later.
