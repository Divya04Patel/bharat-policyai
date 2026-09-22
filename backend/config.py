from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")


@dataclass(frozen=True)
class Settings:
    backend_host: str
    backend_port: int
    data_dir: Path
    index_dir: Path
    embedding_model: str
    embedding_backend: str
    hash_embedding_dim: int
    ollama_base_url: str
    ollama_model: str
    ollama_timeout_sec: int
    ollama_keep_alive: str
    llm_max_tokens: int
    max_context_chunks: int
    retrieval_k: int
    retrieval_min_score: float
    confidence_high_threshold: float
    confidence_medium_threshold: float
    evaluation_dir: Path
    baseline_eval_file: Path


def _resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (ROOT_DIR / path).resolve()


def get_settings() -> Settings:
    data_dir = _resolve_path(os.getenv("DATA_DIR", "data"))
    index_dir = _resolve_path(os.getenv("INDEX_DIR", str(data_dir / "index")))
    evaluation_dir = _resolve_path(os.getenv("EVALUATION_DIR", str(data_dir / "evaluation")))
    baseline_eval_file = _resolve_path(
        os.getenv("BASELINE_EVAL_FILE", str(evaluation_dir / "baseline_questions.json"))
    )

    return Settings(
        backend_host=os.getenv("BACKEND_HOST", "127.0.0.1"),
        backend_port=int(os.getenv("BACKEND_PORT", "5000")),
        data_dir=data_dir,
        index_dir=index_dir,
        embedding_model=os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        ),
        embedding_backend=os.getenv("EMBEDDING_BACKEND", "hash").strip().lower(),
        hash_embedding_dim=int(os.getenv("HASH_EMBEDDING_DIM", "384")),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/"),
        ollama_model=os.getenv("OLLAMA_MODEL", "mistral"),
        ollama_timeout_sec=int(os.getenv("OLLAMA_TIMEOUT_SEC", "75")),
        ollama_keep_alive=os.getenv("OLLAMA_KEEP_ALIVE", "15m"),
        llm_max_tokens=int(os.getenv("LLM_MAX_TOKENS", "128")),
        max_context_chunks=int(os.getenv("MAX_CONTEXT_CHUNKS", "4")),
        retrieval_k=int(os.getenv("RETRIEVAL_K", "6")),
        retrieval_min_score=float(os.getenv("RETRIEVAL_MIN_SCORE", "0.32")),
        confidence_high_threshold=float(os.getenv("CONFIDENCE_HIGH_THRESHOLD", "0.78")),
        confidence_medium_threshold=float(os.getenv("CONFIDENCE_MEDIUM_THRESHOLD", "0.55")),
        evaluation_dir=evaluation_dir,
        baseline_eval_file=baseline_eval_file,
    )
