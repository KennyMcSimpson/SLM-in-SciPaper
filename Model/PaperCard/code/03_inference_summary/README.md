# 03 Evidence Unit Inference

This module contains Stage 3. The only main output is Evidence-grounded Concept Units JSON.

The module reads a TXT paper, calls the Stage 1 keyword checkpoint and the Stage 2 structure checkpoint, builds a fused Top112 ConceptUnit candidate pool, applies transparent feature thresholds, and writes a downstream JSON file.

Default Stage 3 resources live in `datasets/lexicon`:

- `section_bow_vocabulary.csv`
- `section_document_term_matrix_frequency.csv`
- `section_document_term_matrix_tfidf.csv`
- `evidence_cue_lexicon.csv`
- `sentence_evidence_candidates.csv`

Stage 3 follows the report workflow:

1. Stage 1 and Stage 2 produce fused ConceptUnit candidates.
2. The candidate pool keeps Top112 by `I_concept`.
3. Stage 3 applies only three transparent feature thresholds:
   - `phrase_word_number(u) = |tokenize(p_u)|`, pass if `>= 2`.
   - `bow_support_score(u) = clip_0_1(term_confidence(u) * match_quality(u))`, pass if `>= 0.70`.
   - `tfidf_support_score(u) = matched_feature_tfidf(u) / max_tfidf_in_document`, pass if `>= 0.50`.
4. A candidate passes if any one of the three thresholds passes.
5. Passed candidates are ranked by `I_concept`, and the final JSON keeps Top28 by default.

Common entry points:

```powershell
python .\code\03_inference_summary\infer_paper_card.py --help
python .\code\03_inference_summary\batch_infer_evidence_units.py --help
python .\code\03_inference_summary\batch_infer_keywords.py --help
```

Run one paper:

```powershell
python .\code\03_inference_summary\infer_paper_card.py `
  --input_txt .\datasets\03_demo_txt\full_library\2017_transformer_and_large_language_models_attention_is_all_you_need.txt `
  --output_json .\outputs\evidence_units_attention.json `
  --device cuda
```

Main JSON fields:

- `evidence_units`: concept phrase, evidence sentence, context window, role, score components, and BoW metadata.
- `stage3_threshold_policy`: candidate-pool size, final TopK, threshold formulas, and pass counts.
- `document_term_matrix`: document-level frequency and TF-IDF trace for the Stage 3 TF-IDF support feature.
- `section_profiles`: per-section token scale and section BoW hits.
- `corpus_statistics`: token/type counts, type-token ratio, and top unigram/bigram/trigram features.
- `course_nlp_features`: the three report-aligned Stage 3 features: n-gram length, BoW support, and TF-IDF support.
- `concept_graph`: lightweight links from same evidence sentences and same section-role clusters.
