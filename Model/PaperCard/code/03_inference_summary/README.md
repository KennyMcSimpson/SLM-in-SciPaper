# 03 Evidence Unit Inference

This module now treats JSON evidence units as the main output. It is not the final overview generator.

It reads a TXT paper, calls the keyword checkpoint and the structure checkpoint, then merges BIO phrases, sentence role/evidence/importance signals, section-aware BoW metadata, token/ngram statistics, and corpus-level traces into one downstream JSON file.

The default lexicon resources live in `datasets/lexicon`:

- `section_bow_vocabulary.csv`: enriched section BoW with term type, role prior, POS pattern, source corpus, and generic-term flags.
- `section_document_term_matrix_frequency.csv`: document-section term frequency matrix.
- `section_document_term_matrix_tfidf.csv`: document-section TF-IDF matrix.
- `evidence_cue_lexicon.csv`: transparent cue lexicon for method/result/metric/limitation evidence hints.
- `sentence_evidence_candidates.csv`: external cue-mined evidence candidate pool for the demo corpus.

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
- `document_term_matrix`: document-level frequency and TF-IDF trace for course-style BoW analysis.
- `section_profiles`: per-section token scale and section BoW hits.
- `corpus_statistics`: token/type counts, TTR, top unigram/bigram/trigram features.
- `course_nlp_features`: where the course concepts appear in the JSON schema.
- `external_evidence_candidates`: cue-mined candidate sentences from the corpus resource; useful as trace material, not a replacement for the model's evidence units.
- `concept_graph`: lightweight links from same evidence sentence and same section-role clusters.

`--output_md` is still available as a legacy showcase backup. `add_one_paragraph_overview.py` is also kept, but it is no longer the default third-stage target.
