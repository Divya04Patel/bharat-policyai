from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from evaluation.baseline_runner import run_baseline_evaluation


def test_baseline_dataset_loads() -> None:
    dataset_path = Path("data/evaluation/baseline_questions.json")
    assert dataset_path.exists()

    with dataset_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    assert isinstance(payload, list)
    assert 30 <= len(payload) <= 50
    assert all(isinstance(item, dict) for item in payload)
    assert all("category" in item for item in payload)
    assert all("question" in item for item in payload)


def test_baseline_evaluation_generates_report() -> None:
    mocked_result = {
        "answer": "The budget prioritizes inclusive growth.",
        "citations": ["budget.pdf (page 1)"],
        "chunks": [
            {
                "chunk_id": 1,
                "source": "budget.pdf",
                "page": 1,
                "text": "The budget prioritizes inclusive growth.",
                "score": 0.9,
            }
        ],
        "confidence": {"score": 0.9, "label": "high", "is_low_confidence": False},
        "notice": "",
    }

    with patch("backend.app.RAGService.answer", return_value=mocked_result):
        report = run_baseline_evaluation(Path("data/evaluation/baseline_questions.json"))

    assert report["dataset_size"] >= 30
    assert "actual_metrics" in report
    assert "proxy_diagnostics" in report
    assert "api_success" in report["actual_metrics"]
    assert "answer_generated" in report["actual_metrics"]
    assert "fallback_used" in report["actual_metrics"]
    assert "evidence_found" in report["actual_metrics"]
    assert "responses_with_citations" in report["actual_metrics"]
    assert "api_latency_ms" in report["actual_metrics"]
    assert isinstance(report["proxy_diagnostics"], list)
