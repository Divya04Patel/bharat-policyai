from __future__ import annotations

import json
import logging
import importlib
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    faiss = importlib.import_module("faiss")
except ImportError:  # pragma: no cover
    faiss = None

try:
    np = importlib.import_module("numpy")
except ImportError:  # pragma: no cover
    np = None

try:
    requests = importlib.import_module("requests")
except ImportError:  # pragma: no cover
    requests = None

from backend.config import Settings
from backend.embeddings import hash_embed_texts

LOGGER = logging.getLogger(__name__)


class RAGService:
    """Minimal RAG service for budget Q&A with chunk citations."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.index_path = settings.index_dir / "faiss.index"
        self.vectors_path = settings.index_dir / "vectors.npy"
        self.metadata_path = settings.index_dir / "chunks_meta.json"
        self.documents_path = settings.index_dir / "documents_meta.json"
        self.config_path = settings.index_dir / "retrieval_config.json"
        self.index = None
        self.vectors = None
        self.metadata: List[Dict[str, Any]] = []
        self.documents: List[Dict[str, Any]] = []
        self.provenance_ready = False
        self.embedder = None
        self.embedding_backend = settings.embedding_backend
        self.hash_embedding_dim = settings.hash_embedding_dim
        self.session = requests.Session() if requests is not None else None
        self.ready = False
        self.refresh_index()

    @property
    def indexed_chunks(self) -> int:
        return len(self.metadata)

    def refresh_index(self) -> None:
        self.ready = False
        self.metadata = []
        self.documents = []
        self.provenance_ready = False
        self.index = None
        self.vectors = None

        if np is None:
            LOGGER.warning("NumPy is not available. RAG retrieval disabled.")
            return

        if not self.metadata_path.exists():
            LOGGER.info("No index found yet. Run backend.index_builder first.")
            return

        try:
            with self.metadata_path.open("r", encoding="utf-8") as meta_file:
                self.metadata = json.load(meta_file)
            if self.config_path.exists():
                with self.config_path.open("r", encoding="utf-8") as config_file:
                    config_data = json.load(config_file)
                    self.embedding_backend = str(
                        config_data.get("embedding_backend", self.embedding_backend)
                    ).strip().lower()
                    self.hash_embedding_dim = int(
                        config_data.get("hash_embedding_dim", self.hash_embedding_dim)
                    )
                    schema_version = int(config_data.get("schema_version", 1))
                if schema_version == 2:
                    if not self.documents_path.exists():
                        raise ValueError("Provenance index is missing documents_meta.json")
                    with self.documents_path.open("r", encoding="utf-8") as documents_file:
                        self.documents = json.load(documents_file)
                    document_ids = {
                        str(document.get("document_id")) for document in self.documents
                    }
                    required = {
                        "chunk_id", "document_id", "source", "page", "section",
                        "text", "chunk_position", "content_hash",
                    }
                    if not all(required.issubset(chunk) for chunk in self.metadata):
                        raise ValueError("Provenance index has incomplete chunk metadata")
                    if not all(
                        chunk.get("document_id") in document_ids for chunk in self.metadata
                    ):
                        raise ValueError("Chunk references unknown document_id")
                    self.provenance_ready = True
                else:
                    LOGGER.warning("Loading legacy index metadata; rebuild index for provenance support.")

            if faiss is not None and self.index_path.exists():
                self.index = faiss.read_index(str(self.index_path))
                self.ready = self.index is not None and bool(self.metadata)
                return

            if self.vectors_path.exists():
                self.vectors = np.load(self.vectors_path)
                self.ready = bool(self.metadata) and len(self.metadata) == int(
                    self.vectors.shape[0]
                )
                return

            LOGGER.info("No FAISS index or fallback vectors found yet.")
            self.ready = False
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            LOGGER.exception("Failed to load index artifacts: %s", exc)
            self.ready = False

    def _load_embedder(self) -> Optional[Any]:
        if self.embedding_backend != "sentence-transformers":
            return None

        if self.embedder is not None:
            return self.embedder

        try:
            sentence_transformers = importlib.import_module("sentence_transformers")
            sentence_transformer_cls = getattr(
                sentence_transformers,
                "SentenceTransformer",
            )
        except ImportError:
            LOGGER.warning("sentence-transformers is not installed.")
            return None

        self.embedder = sentence_transformer_cls(self.settings.embedding_model)
        return self.embedder

    @lru_cache(maxsize=256)
    def _cached_query_embedding(self, question: str, backend: str, dim: int) -> Optional[tuple]:
        if np is None:
            return None

        if backend == "sentence-transformers":
            embedder = self._load_embedder()
            if embedder is not None:
                vector = embedder.encode(
                    [question],
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                ).astype("float32")[0]
                return tuple(float(v) for v in vector.tolist())

        vector = hash_embed_texts([question], dim=dim)[0]
        return tuple(float(v) for v in vector.tolist())

    def _embed_query(self, question: str) -> Optional[Any]:
        if np is None:
            return None

        cached = self._cached_query_embedding(
            question,
            self.embedding_backend,
            self.hash_embedding_dim,
        )
        if cached is None:
            return None
        return np.asarray([list(cached)], dtype="float32")

    def _retrieve_chunks(self, question: str, top_k: int) -> List[Dict[str, Any]]:
        if not self.ready or np is None:
            return []

        query_vector = self._embed_query(question)
        if query_vector is None:
            return []

        k = min(max(top_k, 1), len(self.metadata))

        retrieved: List[Dict[str, Any]] = []
        if self.index is not None and faiss is not None:
            distances, indices = self.index.search(query_vector.astype("float32"), k)
            for distance, idx in zip(distances[0].tolist(), indices[0].tolist()):
                if idx < 0 or idx >= len(self.metadata):
                    continue
                chunk = self.metadata[idx]
                score = 1.0 / (1.0 + max(float(distance), 0.0))
                retrieved.append(
                    {
                        "chunk_id": chunk.get("chunk_id", idx),
                        "document_id": chunk.get("document_id"),
                        "source": str(chunk.get("source", "unknown")),
                        "page": int(chunk.get("page", 0)),
                        "section": chunk.get("section"),
                        "text": str(chunk.get("text", "")),
                        "chunk_position": chunk.get("chunk_position"),
                        "char_start": chunk.get("char_start"),
                        "char_end": chunk.get("char_end"),
                        "content_hash": chunk.get("content_hash"),
                        "score": round(score, 4),
                    }
                )
        elif self.vectors is not None:
            vectors = self.vectors.astype("float32")
            similarities = vectors @ query_vector[0]
            top_indices = np.argsort(-similarities)[:k].tolist()
            for idx in top_indices:
                if idx < 0 or idx >= len(self.metadata):
                    continue
                chunk = self.metadata[idx]
                score = float((similarities[idx] + 1.0) / 2.0)
                retrieved.append(
                    {
                        "chunk_id": chunk.get("chunk_id", idx),
                        "document_id": chunk.get("document_id"),
                        "source": str(chunk.get("source", "unknown")),
                        "page": int(chunk.get("page", 0)),
                        "section": chunk.get("section"),
                        "text": str(chunk.get("text", "")),
                        "chunk_position": chunk.get("chunk_position"),
                        "char_start": chunk.get("char_start"),
                        "char_end": chunk.get("char_end"),
                        "content_hash": chunk.get("content_hash"),
                        "score": round(max(0.0, min(score, 1.0)), 4),
                    }
                )

        threshold = self.settings.retrieval_min_score
        filtered = [item for item in retrieved if float(item.get("score", 0.0)) >= threshold]

        LOGGER.info(
            "Retrieved %s chunks, kept %s above score threshold %.2f",
            len(retrieved),
            len(filtered),
            threshold,
        )
        return filtered

    def _answer_with_ollama(
        self, question: str, language: str, chunks: List[Dict[str, Any]]
    ) -> Optional[str]:
        if self.session is None:
            LOGGER.warning("requests is not available; Ollama generation disabled.")
            return None

        context_lines = []
        for chunk in chunks[: self.settings.max_context_chunks]:
            context_lines.append(
                f"[{chunk['source']} p.{chunk['page']}] {chunk['text']}"
            )

        context_block = "\n\n".join(context_lines)
        prompt = (
            "You are BharatPolicy AI. Explain budget content in simple language for Indian citizens. "
            "Use only the provided context and do not invent facts. "
            "If context is insufficient, say so clearly.\n\n"
            f"Language preference: {language}\n"
            f"Question: {question}\n\n"
            f"Context:\n{context_block}\n\n"
            "Answer in 4-6 concise lines and end with one practical next step."
        )

        try:
            started = time.perf_counter()
            response = self.session.post(
                f"{self.settings.ollama_base_url}/api/generate",
                json={
                    "model": self.settings.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "keep_alive": self.settings.ollama_keep_alive,
                    "options": {
                        "temperature": 0.2,
                        "num_predict": self.settings.llm_max_tokens,
                    },
                },
                timeout=self.settings.ollama_timeout_sec,
            )
            response.raise_for_status()
            payload = response.json()
            answer = str(payload.get("response", "")).strip()
            LOGGER.info(
                "Ollama response generated in %.2fs using model '%s'",
                time.perf_counter() - started,
                self.settings.ollama_model,
            )
            return answer or None
        except Exception as exc:  # pragma: no cover
            LOGGER.warning(
                "Ollama generation failed, fallback mode enabled (%s): %s",
                type(exc).__name__,
                exc,
            )
            return None

    def _confidence_from_chunks(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not chunks:
            return {
                "score": 0.0,
                "label": "low",
                "is_low_confidence": True,
            }

        top_scores = [float(item.get("score", 0.0)) for item in chunks[:3]]
        while len(top_scores) < 3:
            top_scores.append(0.0)

        weighted = (top_scores[0] * 0.6) + (top_scores[1] * 0.3) + (top_scores[2] * 0.1)
        score = round(max(0.0, min(weighted, 1.0)), 4)

        high = self.settings.confidence_high_threshold
        medium = self.settings.confidence_medium_threshold
        if score >= high:
            label = "high"
        elif score >= medium:
            label = "medium"
        else:
            label = "low"

        return {
            "score": score,
            "label": label,
            "is_low_confidence": label == "low",
        }

    @staticmethod
    def _low_confidence_notice() -> str:
        return (
            "Confidence is low for this answer. Please verify details from the cited source pages "
            "before making application decisions."
        )

    @staticmethod
    def _extractive_fallback(chunks: List[Dict[str, Any]]) -> str:
        if not chunks:
            return (
                "I could not find sufficiently relevant budget context for this question. "
                "Please try a more specific query or rebuild the index with richer source PDFs."
            )

        top_chunks = chunks[:2]
        lines = [
            "I found relevant budget passages and summarized them directly from indexed content.",
        ]
        for chunk in top_chunks:
            preview = chunk["text"][:320].strip()
            lines.append(f"- {preview}")

        return "\n".join(lines)

    @staticmethod
    def _citation_data(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [
            {
                "document_id": chunk.get("document_id"),
                "chunk_id": chunk.get("chunk_id"),
                "source": chunk.get("source"),
                "page": chunk.get("page"),
                "section": chunk.get("section"),
            }
            for chunk in chunks
        ]

    def answer(self, question: str, language: str = "en") -> Dict[str, Any]:
        cleaned_question = question.strip()
        if not cleaned_question:
            return {
                "answer": "Question is required.",
                "citations": [],
                "chunks": [],
                "confidence": {"score": 0.0, "label": "low", "is_low_confidence": True},
                "notice": self._low_confidence_notice(),
            }

        started = time.perf_counter()
        retrieved = self._retrieve_chunks(cleaned_question, self.settings.retrieval_k)
        if not retrieved:
            LOGGER.info("No relevant chunks found for query: %s", cleaned_question)
            return {
                "answer": (
                    "No relevant budget context was found for this query. "
                    "Try more specific keywords (scheme name, sector, or year) and ask again."
                ),
                "citations": [],
                "chunks": [],
                "confidence": {"score": 0.0, "label": "low", "is_low_confidence": True},
                "notice": self._low_confidence_notice(),
            }

        answer_text = self._answer_with_ollama(cleaned_question, language, retrieved)
        if not answer_text:
            answer_text = self._extractive_fallback(retrieved)

        confidence = self._confidence_from_chunks(retrieved)
        notice = self._low_confidence_notice() if confidence["is_low_confidence"] else ""

        citations = []
        citation_data = self._citation_data(retrieved[: self.settings.max_context_chunks])
        seen = set()
        for chunk in retrieved[: self.settings.max_context_chunks]:
            citation = f"{chunk['source']} (page {chunk['page']})"
            if citation not in seen:
                seen.add(citation)
                citations.append(citation)

        LOGGER.info(
            "Answered query in %.2fs with %s citations (confidence=%s)",
            time.perf_counter() - started,
            len(citations),
            confidence["label"],
        )

        return {
            "answer": answer_text,
            "citations": citations,
            "citation_data": citation_data,
            "chunks": retrieved[: self.settings.max_context_chunks],
            "confidence": confidence,
            "notice": notice,
        }
