"""TODO (stages 2-4 together): the full volume classifier.

    VolumeClassifier(encoder: SliceEncoder | None, aggregator: Aggregator, head: Head)

* ``encoder=None`` (phase 1): the input is a padded batch of precomputed embeddings
  (ct_model.data.datasets.EmbeddingBagDataset + collate_bags) -> aggregator -> head.
* ``encoder`` set (phase 2 LoRA training, and clinical inference): the input is HU slices; each volume's
  slices go through the encoder in slice batches (as ct_model.embeddings.extract.encode_slices does), then
  the same aggregator and head. Inference must take this path on ct_preprocessing.inference.run_inference
  output, so training and serving see identically prepared inputs.

``forward`` returns the logits (B, n_labels) and the aggregator's attention, for the report notebooks.
"""
