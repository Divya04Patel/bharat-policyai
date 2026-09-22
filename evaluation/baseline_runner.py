from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app import create_app
from backend.config import get_settings


def _load_dataset(dataset_path: Path) -> List[Dict[str, Any]]:
    if not dataset_path.exists():
        return []

    with dataset_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, list):
        raise ValueError(f"Baseline dataset at {dataset_path} must contain a list of questions.")

    return payload


def run_baseline_evaluation(dataset_path: Path | None = None) -> Dict[str, Any]:
    settings = get_settings()
    dataset_file = dataset_path or settings.baseline_eval_file
    items = _load_dataset(dataset_file)

    app = create_app()
    client = app.test_client()

    results: List[Dict[str, Any]] = []
    for item in items:
        started = time.perf_counter()
        response = client.post(
            "/api/ask-budget",
            json={
                "question": str(item.get("question", "")).strip(),
                "language": "en",
            },
        )
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)

        payload = response.get_json(silent=True) if response.is_json else {}
        body = payload if isinstance(payload, dict) else {}
        answer = body.get("answer", "")
        citations = body.get("citations", []) if isinstance(body.get("citations", []), list) else []
        fallback = bool(
            response.status_code == 200
            and (
                "No relevant budget context" in answer
                or "I could not find sufficiently relevant" in answer
                or "I found relevant budget passages and summarized them directly from indexed content." in answer
            )
        )
        chunks = body.get("chunks", []) if isinstance(body.get("chunks", []), list) else []
        api_success = bool(response.status_code == 200 and body.get("ok") is True)
        answer_generated = bool(api_success and answer and not fallback)

        results.append(
            {
                "id": item.get("id"),
                "category": item.get("category"),
                "question": item.get("question"),
                "answerable": bool(item.get("answerable", False)),
                "api_success": api_success,
                "answer_generated": answer_generated,
                "fallback_used": fallback,
                "evidence_found": bool(chunks),
                "citations_present": bool(citations),
                "response_status": response.status_code,
                "citation_count": len(citations),
                "api_latency_ms": elapsed_ms,
                "retrieval_latency_ms": None,
                "llm_latency_ms": None,
                "expected_source_present": None,
                "proxy_metric": "api_success_and_evidence_presence",
            }
        )

    api_successes = sum(1 for row in results if row["api_success"])
    answer_generated = sum(1 for row in results if row["answer_generated"])
    fallback_count = sum(1 for row in results if row["fallback_used"])
    evidence_count = sum(1 for row in results if row["evidence_found"])
    citations_count = sum(1 for row in results if row["citations_present"])
    api_latency = round(
        sum(row["api_latency_ms"] for row in results) / len(results),
        2,
    ) if results else 0.0

    return {
        "dataset_file": str(dataset_file),
        "dataset_size": len(results),
        "actual_metrics": {
            "api_success": api_successes,
            "answer_generated": answer_generated,
            "fallback_used": fallback_count,
            "evidence_found": evidence_count,
            "responses_with_citations": citations_count,
            "api_latency_ms": api_latency,
            "retrieval_latency_ms": None,
            "llm_latency_ms": None,
        },
        "proxy_diagnostics": results,
        "notes": "This evaluation measures current API behavior, evidence presence, fallback usage, citation presence, and API round-trip latency. Retrieval and LLM stage latencies are unavailable because the current application does not expose them. It does not claim Recall@K, Precision@K, faithfulness, or other retrieval metrics that the current system does not directly compute.",
    }


if __name__ == "__main__":
    report = run_baseline_evaluation()
    print(json.dumps(report, indent=2, ensure_ascii=True))
