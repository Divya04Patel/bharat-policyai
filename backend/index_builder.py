from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import re
from pathlib import Path
from typing import Dict, List

try:
    import faiss  # type: ignore
except ImportError:  # pragma: no cover
    faiss = None

import numpy as np

try:
    import fitz  # PyMuPDF
except ImportError:  # pragma: no cover
    fitz = None

from backend.config import get_settings
from backend.embeddings import hash_embed_texts


def _embedding_key(
    text: str,
    backend_name: str,
    embedding_model: str,
    hash_embedding_dim: int,
) -> str:
    material = (
        f"{backend_name}|{embedding_model}|{hash_embedding_dim}|{text}"
    ).encode("utf-8")
    return hashlib.sha1(material).hexdigest()


def _load_embedding_cache(cache_path: Path) -> Dict[str, np.ndarray]:
    if not cache_path.exists():
        return {}

    try:
        loaded = np.load(cache_path, allow_pickle=True)
        keys = loaded["keys"].tolist()
        vectors = loaded["vectors"].astype("float32")
        return {str(key): vectors[idx] for idx, key in enumerate(keys)}
    except Exception:
        return {}


def _save_embedding_cache(cache_path: Path, cache: Dict[str, np.ndarray]) -> None:
    if not cache:
        return

    keys = sorted(cache.keys())
    vectors = np.stack([cache[key] for key in keys]).astype("float32")
    np.savez_compressed(
        cache_path,
        keys=np.asarray(keys, dtype=object),
        vectors=vectors,
    )


def _embed_texts_with_cache(
    texts: List[str],
    embedding_model: str,
    backend_name: str,
    hash_embedding_dim: int,
    cache: Dict[str, np.ndarray],
) -> np.ndarray:
    keys = [
        _embedding_key(
            text=text,
            backend_name=backend_name,
            embedding_model=embedding_model,
            hash_embedding_dim=hash_embedding_dim,
        )
        for text in texts
    ]

    if backend_name == "sentence-transformers":
        sentence_transformers = importlib.import_module("sentence_transformers")
        sentence_transformer_cls = getattr(sentence_transformers, "SentenceTransformer")
        embedder = sentence_transformer_cls(embedding_model)

        dim = int(embedder.get_sentence_embedding_dimension())
        vectors = np.zeros((len(texts), dim), dtype="float32")
        missing_indices = [idx for idx, key in enumerate(keys) if key not in cache]
        if missing_indices:
            missing_texts = [texts[idx] for idx in missing_indices]
            encoded = embedder.encode(
                missing_texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                batch_size=64,
                show_progress_bar=True,
            )
            encoded_vectors = np.asarray(encoded, dtype="float32")
            for pos, idx in enumerate(missing_indices):
                cache[keys[idx]] = encoded_vectors[pos]

        for idx, key in enumerate(keys):
            vectors[idx] = cache[key]

        return vectors

    vectors = np.zeros((len(texts), hash_embedding_dim), dtype="float32")
    missing_indices = [idx for idx, key in enumerate(keys) if key not in cache]
    if missing_indices:
        missing_texts = [texts[idx] for idx in missing_indices]
        encoded_vectors = hash_embed_texts(texts=missing_texts, dim=hash_embedding_dim)
        for pos, idx in enumerate(missing_indices):
            cache[keys[idx]] = encoded_vectors[pos]

    for idx, key in enumerate(keys):
        vectors[idx] = cache[key]

    return vectors


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = 700, overlap: int = 120) -> List[str]:
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be larger than overlap")

    if not text:
        return []

    chunks: List[str] = []
    step = chunk_size - overlap
    for start in range(0, len(text), step):
        piece = text[start : start + chunk_size].strip()
        if len(piece) < 120:
            continue
        chunks.append(piece)
        if start + chunk_size >= len(text):
            break
    return chunks


def extract_pdf_chunks(
    pdf_path: Path, chunk_size: int = 700, overlap: int = 120
) -> List[Dict[str, str | int]]:
    if fitz is None:
        raise RuntimeError("PyMuPDF is not installed. Install dependencies first.")

    extracted: List[Dict[str, str | int]] = []
    with fitz.open(pdf_path) as doc:
        for page_number, page in enumerate(doc, start=1):
            page_text = clean_text(page.get_text("text"))
            if not page_text:
                continue
            for chunk in chunk_text(page_text, chunk_size=chunk_size, overlap=overlap):
                extracted.append(
                    {
                        "source": pdf_path.name,
                        "page": page_number,
                        "text": chunk,
                    }
                )

    return extracted


def build_index(
    input_dir: Path,
    index_dir: Path,
    embedding_model: str,
    embedding_backend: str = "hash",
    hash_embedding_dim: int = 384,
    chunk_size: int = 700,
    chunk_overlap: int = 120,
) -> Dict[str, str | int]:

    pdf_files = sorted(input_dir.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {input_dir}")

    all_chunks: List[Dict[str, str | int]] = []
    for pdf_path in pdf_files:
        all_chunks.extend(
            extract_pdf_chunks(
                pdf_path,
                chunk_size=chunk_size,
                overlap=chunk_overlap,
            )
        )

    if not all_chunks:
        raise RuntimeError("No extractable text found in input PDFs.")

    texts = [str(item["text"]) for item in all_chunks]
    backend_name = embedding_backend.strip().lower()
    if backend_name != "sentence-transformers":
        backend_name = "hash"

    index_dir.mkdir(parents=True, exist_ok=True)
    index_path = index_dir / "faiss.index"
    vectors_path = index_dir / "vectors.npy"
    metadata_path = index_dir / "chunks_meta.json"
    config_path = index_dir / "retrieval_config.json"
    cache_path = index_dir / "embedding_cache.npz"

    cache = _load_embedding_cache(cache_path)
    cache_before = len(cache)
    vectors = _embed_texts_with_cache(
        texts=texts,
        embedding_model=embedding_model,
        backend_name=backend_name,
        hash_embedding_dim=hash_embedding_dim,
        cache=cache,
    )
    _save_embedding_cache(cache_path, cache)

    storage = "numpy"
    vector_count = int(vectors.shape[0])
    if faiss is not None:
        index = faiss.IndexFlatL2(vectors.shape[1])
        index.add(vectors)
        faiss.write_index(index, str(index_path))
        if vectors_path.exists():
            vectors_path.unlink()
        storage = "faiss"
        vector_count = int(index.ntotal)
    else:
        np.save(vectors_path, vectors)
        if index_path.exists():
            index_path.unlink()

    metadata = []
    for idx, chunk in enumerate(all_chunks):
        metadata.append(
            {
                "chunk_id": idx,
                "source": chunk["source"],
                "page": chunk["page"],
                "text": chunk["text"],
            }
        )

    with metadata_path.open("w", encoding="utf-8") as output_file:
        json.dump(metadata, output_file, ensure_ascii=True, indent=2)

    with config_path.open("w", encoding="utf-8") as config_file:
        json.dump(
            {
                "embedding_backend": backend_name,
                "embedding_model": embedding_model,
                "hash_embedding_dim": hash_embedding_dim,
                "storage": storage,
                "vector_count": vector_count,
                "embedding_cache_file": cache_path.name,
                "embedding_cache_size": len(cache),
                "embedding_cache_hits": int(vector_count - (len(cache) - cache_before)),
            },
            config_file,
            ensure_ascii=True,
            indent=2,
        )

    return {
        "pdf_count": len(pdf_files),
        "chunk_count": len(all_chunks),
        "vector_count": vector_count,
        "storage": storage,
        "embedding_backend": backend_name,
    }


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Build FAISS index from budget PDFs")
    parser.add_argument("--input-dir", default=str(settings.data_dir / "raw"))
    parser.add_argument("--index-dir", default=str(settings.index_dir))
    parser.add_argument("--embedding-model", default=settings.embedding_model)
    parser.add_argument("--embedding-backend", default=settings.embedding_backend)
    parser.add_argument("--hash-embedding-dim", type=int, default=settings.hash_embedding_dim)
    parser.add_argument("--chunk-size", type=int, default=700)
    parser.add_argument("--chunk-overlap", type=int, default=120)
    args = parser.parse_args()

    stats = build_index(
        input_dir=Path(args.input_dir),
        index_dir=Path(args.index_dir),
        embedding_model=args.embedding_model,
        embedding_backend=args.embedding_backend,
        hash_embedding_dim=args.hash_embedding_dim,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
