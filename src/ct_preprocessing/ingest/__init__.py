"""Chunked ingest: fetch a chunk of raw scans, preprocess it into the permanent
cache, delete the raw data, repeat -- so a dataset far bigger than the server's
disk can be processed unattended. See docs/preprocessing/README.md."""
