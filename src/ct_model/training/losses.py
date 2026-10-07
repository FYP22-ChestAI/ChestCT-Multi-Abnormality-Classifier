"""TODO: multi-label losses, all taking logits (B, L), targets (B, L) and label_mask (B, L).

* bce            BCE-with-logits, averaged over unmasked entries.
* weighted_bce   pos_weight per label = negatives / positives, counted on the TRAIN split only.
* asl            asymmetric loss (Ridnik et al., ICCV 2021) -- CT-RATE's labels are heavily imbalanced.

Missing labels (label_mask False) must contribute nothing to the loss.
"""
