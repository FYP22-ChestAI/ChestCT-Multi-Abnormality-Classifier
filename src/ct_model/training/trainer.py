"""TODO: the training loop (phase 1: aggregator + head on frozen embeddings).

Intended design:
* Read an experiment YAML (configs/model/experiments/<name>.yaml) into a typed, strict config, like
  ct_model.config. Write everything to outputs/experiments/<name>/<UTC timestamp>/: the resolved config,
  the embedding store's record (which encoder, weights commit, HU-cache fingerprint), the label names,
  checkpoints (best by val macro AUROC, and last), metrics.csv per epoch, and test metrics per kernel class.
* Data: select_volumes(...) per split -> EmbeddingBagDataset -> DataLoader(collate_fn=collate_bags).
  Open the store with ct_model.embeddings.open_store(..., create=False), which refuses a store built
  from another HU cache. Train on the experiment's train kernel classes, validate on the same, and test
  on every kernel class, reported separately (data contract: "report test results per kernel").
* AdamW + cosine schedule with warmup, bf16 autocast, gradient clipping, early stopping on val macro
  AUROC, seed_everything(seed). Never tune anything on the test split.
"""
