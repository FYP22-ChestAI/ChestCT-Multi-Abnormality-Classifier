"""TODO: evaluation metrics.

* Per label: AUROC, AUPRC, and F1 / sensitivity / specificity at a threshold chosen on VAL (never test).
* Summary: macro AUROC (the CT-RATE literature's headline number, comparable with the DALE-CT paper's
  linear-probe MIL results), micro AUROC, and 95% bootstrap CIs for test (resampling patients).
* Report test separately per kernel_class (sharp / soft) and per source (CT-RATE / NHRD).
* A label with only one class present in a split is reported as NaN, never dropped silently.
"""
