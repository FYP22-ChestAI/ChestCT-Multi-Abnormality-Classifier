"""Stages 3 + 4, phase 1: train an aggregator + head on frozen embeddings, then evaluate it.

    config.py    experiment YAML (configs/model/experiments/*.yaml), strict and typed
    data.py      run manifest + embedding store -> datasets per split
    losses.py    bce / weighted_bce / asl (registered in ct_model.registry.LOSSES)
    metrics.py   AUROC / AUPRC / thresholded metrics / patient bootstrap (scikit-learn)
    trainer.py   the training loop (train + val only) and load_trained()
    evaluate.py  one finished run on test (or another run): overall, per kernel, per scanner
    records.py   experiment ids, folders and the ledgers (outputs/experiments/index.csv, evaluations.csv)
"""
