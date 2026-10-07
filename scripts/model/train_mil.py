"""TODO (stages 3 + 4, phase 1): train an aggregator + head on frozen embeddings.

    python scripts/model/train_mil.py --experiment abmil_dale2s

Intended: read configs/model/experiments/<name>.yaml, open the encoder's embedding store
(ct_model.embeddings.open_store(..., create=False)), build EmbeddingBagDatasets per split from the run
manifest, train with ct_model.training, and write outputs/experiments/<name>/<timestamp>/.
See docs/model/architecture.md ("Roadmap") and the docstrings in src/ct_model/training/.
"""
from __future__ import annotations

import sys


def main() -> int:
    print("train_mil.py is not implemented yet (stages 3 + 4, phase 1) -- see docs/model/architecture.md", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
