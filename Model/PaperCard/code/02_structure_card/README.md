# 02 Structure Model

This module contains Stage 2: section-aware sentence modeling for scientific papers.

The model uses a SciBERT encoder with section embeddings, a sentence-context Transformer, and three sentence-level heads:

- `role`: predicts the rhetorical role of a sentence;
- `evidence`: predicts whether a sentence can ground an extracted concept;
- `importance`: predicts whether a sentence should contribute to downstream evidence units.

Training supervision is split by dataset:

- PubMed RCT provides hard role labels.
- QASPER provides evidence labels and section-level partial role constraints.
- ACLSum provides importance labels from an extractive ROUGE-L oracle over facet summaries; explicit facet roles are used only when reliable.

Common entry points:

```powershell
python .\code\02_structure_card\download_structure_resources.py --help
python .\code\02_structure_card\train_structure.py --help
python .\code\02_structure_card\smoke_test_structure.py
```

Expected processed data location:

```text
datasets/02_structure_card/processed
```

Default final checkpoint:

```text
models/checkpoints/structure_v4_partial_role_balanced_fulldev
```
