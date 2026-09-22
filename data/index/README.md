# Index Artifacts

The FAISS index and metadata are generated here.

Expected files after indexing:

- faiss.index
- chunks_meta.json
- documents_meta.json
- retrieval_config.json

`chunks_meta.json` contains provenance-linked chunk records. `documents_meta.json`
contains deterministic document records. Rebuild the index after provenance changes:

```bash
python -m backend.index_builder --input-dir data/raw --index-dir data/index
```

Validate the rebuilt artifacts with:

```bash
python -c "import json; from pathlib import Path; p=Path('data/index'); print(len(json.loads((p/'documents_meta.json').read_text()))); print(len(json.loads((p/'chunks_meta.json').read_text())))"
```
