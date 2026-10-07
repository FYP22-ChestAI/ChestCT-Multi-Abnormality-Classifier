"""The model side of ChestAI: stages 2-4 on top of the ct_preprocessing HU cache.

    stage 2  encoders/     one 2D backbone per slice -> (n_slices, D) embeddings  (embeddings/ stores them)
    stage 3  aggregators/  a bag of slice embeddings -> one volume embedding       (TODO: ABMIL, then query / QGMIL)
    stage 4  heads/        volume embedding -> multi-label logits                  (TODO)

See docs/model/architecture.md for how the stages connect and what each one hands to the next.
"""
