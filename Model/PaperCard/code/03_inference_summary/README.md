# 03 Evidence Unit Inference

This module now treats JSON evidence units as the main output. It is not the final overview generator.

It reads a TXT paper, calls the keyword checkpoint and the structure checkpoint, then builds a Top112 fused ConceptUnit candidate pool and filters it into a downstream JSON file.

The default lexicon resources live in `datasets/lexicon`:

- `section_bow_vocabulary.csv`: enriched section BoW with term type, role prior, POS pattern, source corpus, and generic-term flags.
- `section_document_term_matrix_frequency.csv`: document-section term frequency matrix.
- `section_document_term_matrix_tfidf.csv`: document-section TF-IDF matrix.

Stage3 follows the report workflow:

1. Stage1 and Stage2 produce fused ConceptUnit candidates.
2. The candidate pool keeps Top112 by `I_concept`.
3. Stage3 applies only three transparent feature thresholds:
   - `phrase_word_number(u) = |tokenize(p_u)|`, pass if `>= 2`.
   - `bow_support_score(u) = clip_0_1(term_confidence(u) * match_quality(u))`, pass if `>= 0.70`.
   - `tfidf_support_score(u) = matched_feature_tfidf(u) / max_tfidf_in_document`, pass if `>= 0.50`.
4. A candidate passes the threshold gate if any one of the three feature thresholds passes.
5. Passed candidates are ranked by `I_concept`, and the final JSON keeps Top28 by default.

Common entry points:

```powershell
..\.venv_keyword\Scripts\python.exe .\code\03_inference_summary\infer_paper_card.py --help
..\.venv_keyword\Scripts\python.exe .\code\03_inference_summary\evaluate_paper_cards.py --help
..\.venv_keyword\Scripts\python.exe .\code\03_inference_summary\add_one_paragraph_overview.py --help
```

Run the Attention example:

```powershell
..\.venv_keyword\Scripts\python.exe .\code\03_inference_summary\infer_paper_card.py `
  --input_txt .\datasets\03_demo_txt\full_library\2017_transformer_and_large_language_models_attention_is_all_you_need.txt `
  --output_json .\outputs\evidence_units_attention.json `
  --device cuda
```

Main JSON fields:

- `evidence_units`: concept phrase, evidence sentence, context window, role, score components, BoW metadata.
- `stage3_threshold_policy`: candidate-pool size, final TopK, threshold formulas, and pass counts.
- `document_term_matrix`: document-level frequency and TF-IDF trace for the Stage3 TF-IDF support feature.
- `section_profiles`: per-section token scale and section BoW hits.
- `corpus_statistics`: token/type counts, TTR, top unigram/bigram/trigram features.
- `course_nlp_features`: only the three report-aligned Stage3 features: n-gram length, BoW support, and TF-IDF support.
- `concept_graph`: lightweight links from same evidence sentence and same section-role clusters.

`--output_md` is still available as a legacy showcase backup. `add_one_paragraph_overview.py` is also kept, but it is no longer the default third-stage target.
