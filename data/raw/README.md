# Add Budget PDFs Here

Place Indian budget PDF files in this folder to build the retrieval index.

## Sample Files (Pre-indexed)

The prototype comes with pre-indexed sample PDFs for demo purposes:

- budget 2024_25.pdf
- budget2021_22.pdf

These are already indexed in `data/index/`, so no rebuild is required for basic demo.

## To Index Your Own PDFs

1. Add new PDF files to this folder (`data/raw/`)
2. Run the index builder:

```bash
python -m backend.index_builder --input-dir data/raw --index-dir data/index
```

3. Restart the backend to reload the index

## Expected Index Output

After indexing, you should see:

- `data/index/vectors.npy` — Binary vector embeddings
- `data/index/chunks_meta.json` — Chunk metadata (source, page, text)
- `data/index/retrieval_config.json` — Embedding backend and configuration
- `data/index/faiss.index` — FAISS index (optional, if FAISS is installed)
