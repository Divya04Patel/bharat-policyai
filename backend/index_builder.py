from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

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


def document_id_for_content(content: bytes) -> str:
    return f"doc-{hashlib.sha256(content).hexdigest()}"


def content_hash_for_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def chunk_id_for_content(document_id: str, position: int, text: str) -> str:
    material = f"{document_id}|{position}|{text}".encode("utf-8")
    return f"chunk-{hashlib.sha256(material).hexdigest()}"


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


def _chunk_text_with_boundaries(
    text: str, chunk_size: int = 700, overlap: int = 120
) -> List[Tuple[str, int, int]]:
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be larger than overlap")

    chunks: List[Tuple[str, int, int]] = []
    step = chunk_size - overlap
    for start in range(0, len(text), step):
        piece = text[start : start + chunk_size].strip()
        if len(piece) < 120:
            continue
        leading = len(text[start : start + chunk_size]) - len(
            text[start : start + chunk_size].lstrip()
        )
        chunk_start = start + leading
        chunks.append((piece, chunk_start, chunk_start + len(piece)))
        if start + chunk_size >= len(text):
            break
    return chunks


def extract_pdf_document(
    pdf_path: Path, chunk_size: int = 700, overlap: int = 120
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    if fitz is None:
        raise RuntimeError("PyMuPDF is not installed. Install dependencies first.")

    raw_content = pdf_path.read_bytes()
    document_id = document_id_for_content(raw_content)
    chunks: List[Dict[str, Any]] = []
    with fitz.open(stream=raw_content, filetype="pdf") as doc:
        metadata = doc.metadata or {}
        title = str(metadata.get("title") or "").strip() or None
        document = {
            "document_id": document_id,
            "original_filename": pdf_path.name,
            "source_url": None,
            "document_title": title,
            "publication_date": None,
            "effective_date": None,
            "ingestion_timestamp": datetime.now(timezone.utc).isoformat(),
            "content_hash": hashlib.sha256(raw_content).hexdigest(),
            "page_count": len(doc),
            "document_type": "application/pdf",
        }
        position = 0
        for page_number, page in enumerate(doc, start=1):
            page_text = clean_text(page.get_text("text"))
            if not page_text:
                continue
            for text, char_start, char_end in _chunk_text_with_boundaries(
                page_text, chunk_size=chunk_size, overlap=overlap
            ):
                chunks.append(
                    {
                        "chunk_id": chunk_id_for_content(document_id, position, text),
                        "document_id": document_id,
                        "source": pdf_path.name,
                        "page": page_number,
                        "section": None,
                        "text": text,
                        "chunk_position": position,
                        "char_start": char_start,
                        "char_end": char_end,
                        "content_hash": content_hash_for_text(text),
                    }
                )
                position += 1
    return document, chunks


def extract_pdf_chunks(
    pdf_path: Path, chunk_size: int = 700, overlap: int = 120
) -> List[Dict[str, str | int]]:
    _, extracted = extract_pdf_document(
        pdf_path, chunk_size=chunk_size, overlap=overlap
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

    all_chunks: List[Dict[str, Any]] = []
    documents: List[Dict[str, Any]] = []
    for pdf_path in pdf_files:
        document, chunks = extract_pdf_document(
            pdf_path, chunk_size=chunk_size, overlap=chunk_overlap
        )
        documents.append(document)
        all_chunks.extend(chunks)

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
    documents_path = index_dir / "documents_meta.json"
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

    with metadata_path.open("w", encoding="utf-8") as output_file:
        json.dump(all_chunks, output_file, ensure_ascii=True, indent=2)

    with documents_path.open("w", encoding="utf-8") as documents_file:
        json.dump(documents, documents_file, ensure_ascii=True, indent=2)

    with config_path.open("w", encoding="utf-8") as config_file:
        json.dump(
            {
                "schema_version": 2,
                "embedding_backend": backend_name,
                "embedding_model": embedding_model,
                "hash_embedding_dim": hash_embedding_dim,
                "storage": storage,
                "vector_count": vector_count,
                "embedding_cache_file": cache_path.name,
                "embedding_cache_size": len(cache),
                "embedding_cache_hits": int(vector_count - (len(cache) - cache_before)),
                "documents_metadata_file": documents_path.name,
                "chunks_metadata_file": metadata_path.name,
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
