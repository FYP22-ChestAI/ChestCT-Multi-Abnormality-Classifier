"""Two numbers that describe an embedding store before any MIL model is trained on it -- check them for
every encoder (frozen, ablation, LoRA-adapted), next to the clinical report of the models built on it.

* ``kernel_probe``  how well the reconstruction kernel (sharp vs soft) can be read from a volume's mean
                    embedding: 5-fold cross-validated AUROC of a logistic regression. Near 1.0 = the kernel
                    dominates the embedding (a domain gap); domain adaptation should push it down.
* ``linear_probe``  per label, a logistic regression on the mean embedding, fitted on train and scored on
                    test: the floor that any aggregator should beat. If MIL barely beats it, the encoder --
                    not the aggregator -- limits performance.

    from ct_model.embeddings.diagnostics import mean_embeddings, kernel_probe, linear_probe
    X = mean_embeddings(store, records)
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..data.volumes import VolumeRecord
from .store import EmbeddingStore


def mean_embeddings(store: EmbeddingStore, records: Sequence[VolumeRecord]) -> np.ndarray:
    """(n_volumes, D) float32: the mean over slices of each volume's embeddings, in ``records`` order."""
    return np.stack([np.asarray(store.load(r.volume_id), dtype=np.float32).mean(0) for r in records])


def kernel_probe(X: np.ndarray, kernel_classes: Sequence[str], classes: tuple[str, str] = ("sharp", "soft"),
                 folds: int = 5, C: float = 0.1, seed: int = 0) -> float:
    """Cross-validated AUROC of telling ``classes[1]`` from ``classes[0]`` with a linear model; NaN if a
    class has fewer than ``folds`` volumes. Use volumes of one split (e.g. test, which holds both kernels)."""
    kernel_classes = np.asarray(kernel_classes).astype(str)
    keep = np.isin(kernel_classes, classes)
    y = kernel_classes[keep] == classes[1]
    if min(y.sum(), (~y).sum()) < folds:
        return float("nan")
    model = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    return float(cross_val_score(model, X[keep], y, cv=cv, scoring="roc_auc").mean())


def linear_probe(X_train: np.ndarray, Y_train: np.ndarray, X_test: np.ndarray, Y_test: np.ndarray,
                 label_names: Sequence[str], C: float = 0.01) -> pd.DataFrame:
    """Per label: test AUROC of a logistic regression on standardised mean embeddings (NaN labels skipped)."""
    scaler = StandardScaler().fit(X_train)
    A, T = scaler.transform(X_train), scaler.transform(X_test)
    rows = []
    for c, name in enumerate(label_names):
        tr, te = ~np.isnan(Y_train[:, c]), ~np.isnan(Y_test[:, c])
        y_tr, y_te = Y_train[tr, c], Y_test[te, c]
        auroc = np.nan
        if 0 < y_tr.sum() < y_tr.size and 0 < y_te.sum() < y_te.size:
            model = LogisticRegression(C=C, max_iter=3000).fit(A[tr], y_tr)
            auroc = roc_auc_score(y_te, model.predict_proba(T[te])[:, 1])
        rows.append({"label": name, "n_train": int(tr.sum()), "n_test": int(te.sum()), "auroc": auroc})
    return pd.DataFrame(rows)
