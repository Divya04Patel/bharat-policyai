from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from backend.config import get_settings
from backend.index_builder import (
    build_index,
    chunk_id_for_content,
    chunk_page_units,
    document_id_for_content,
    extract_pdf_document,
)
from backend.rag_service import RAGService


RAW_PDF = Path("data/raw/budget 2024_25.pdf")


def test_document_and_chunk_ids_are_deterministic() -> None:
    content = b"same document content"
    document_id = document_id_for_content(content)

    assert document_id == document_id_for_content(content)
    assert document_id != document_id_for_content(b"different document content")
    assert chunk_id_for_content(document_id, 0, "chunk text") == chunk_id_for_content(
        document_id, 0, "chunk text"
    )
    assert chunk_id_for_content(document_id, 0, "chunk text") != chunk_id_for_content(
        document_id, 1, "chunk text"
    )


def test_extracted_chunks_reference_document_and_missing_dates_are_null() -> None:
    document, chunks = extract_pdf_document(RAW_PDF)

    assert chunks
    assert document["document_id"] == document_id_for_content(RAW_PDF.read_bytes())
    assert document["publication_date"] is None
    assert document["effective_date"] is None
    assert all(chunk["document_id"] == document["document_id"] for chunk in chunks)
    assert all(
        chunk["section"] is None
        or chunk["section"].lower().startswith("priority ")
        for chunk in chunks
    )
    assert all(chunk["content_hash"] for chunk in chunks)


def test_chunking_prefers_paragraph_boundaries_and_preserves_page_units() -> None:
    units = [
        ("First paragraph with enough text to stand alone.", None),
        ("Second paragraph with separate provenance context.", None),
    ]

    chunks = chunk_page_units(units, chunk_size=120)

    assert len(chunks) == 1
    assert "First paragraph" in chunks[0][0]
    assert "Second paragraph" in chunks[0][0]
    assert "\n\n" in chunks[0][0]


def test_chunking_is_deterministic_and_does_not_cross_pages() -> None:
    page_one = chunk_page_units([("Page one paragraph. " * 20, None)], chunk_size=120)
    page_two = chunk_page_units([("Page two paragraph. " * 20, None)], chunk_size=120)

    assert page_one == chunk_page_units([("Page one paragraph. " * 20, None)], chunk_size=120)
    assert all("Page two" not in chunk[0] for chunk in page_one)
    assert all("Page one" not in chunk[0] for chunk in page_two)


def test_rebuilt_index_exposes_provenance_and_citations(tmp_path: Path) -> None:
    input_dir = tmp_path / "raw"
    index_dir = tmp_path / "index"
    input_dir.mkdir()
    (input_dir / RAW_PDF.name).write_bytes(RAW_PDF.read_bytes())

    settings = replace(get_settings(), data_dir=input_dir, index_dir=index_dir)
    stats = build_index(
        input_dir=input_dir,
        index_dir=index_dir,
        embedding_model=settings.embedding_model,
        embedding_backend="hash",
        hash_embedding_dim=settings.hash_embedding_dim,
    )

    documents = json.loads((index_dir / "documents_meta.json").read_text(encoding="utf-8"))
    chunks = json.loads((index_dir / "chunks_meta.json").read_text(encoding="utf-8"))
    document_ids = {document["document_id"] for document in documents}

    assert stats["chunk_count"] == len(chunks)
    assert stats["vector_count"] == len(chunks)
    assert len(documents) == 1
    assert all(chunk["document_id"] in document_ids for chunk in chunks)

    service = RAGService(settings)
    retrieved = service._retrieve_chunks(
        "What are the main objectives and priorities of the budget?",
        settings.retrieval_k,
    )
    required = {
        "chunk_id",
        "document_id",
        "source",
        "page",
        "section",
        "text",
        "score",
    }
    assert service.provenance_ready is True
    assert retrieved
    assert all(required.issubset(chunk) for chunk in retrieved)
    citation_data = service._citation_data(retrieved[: settings.max_context_chunks])
    assert citation_data
    assert all(
        {"document_id", "chunk_id", "source", "page", "section"}.issubset(item)
        for item in citation_data
    )
