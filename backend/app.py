from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List

from flask import Flask, jsonify, request
from flask_cors import CORS

from backend.config import get_settings
from backend.rag_service import RAGService
from backend.recommender import SchemeRecommender

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


def _error_response(message: str, code: str, status: int = 400) -> Any:
    return jsonify({"ok": False, "error": {"code": code, "message": message}}), status


def _success_response(payload: Dict[str, Any], status: int = 200) -> Any:
    return jsonify({"ok": True, **payload}), status


def _parse_json_payload() -> Dict[str, Any]:
    payload = request.get_json(silent=True)
    return payload if isinstance(payload, dict) else {}


def _parse_top_n(raw_value: Any, default: int = 5) -> int:
    try:
        return max(1, min(int(raw_value), 10))
    except (TypeError, ValueError):
        return default


def _build_impact_summary(
    profile: Dict[str, Any], recommendations: List[Dict[str, Any]]
) -> Dict[str, Any]:
    occupation = str(profile.get("occupation", "citizen")).strip() or "citizen"
    state = str(profile.get("state", "your state")).strip() or "your state"

    if not recommendations:
        return {
            "headline": f"Budget impact snapshot for {occupation}",
            "summary": (
                "No strong scheme match found with current profile. "
                "Try refining income range, occupation, or tags."
            ),
            "next_actions": [
                "Update profile details",
                "Try broader occupation tag",
                "Check state-specific portals",
            ],
        }

    top_names = ", ".join(item["scheme_name"] for item in recommendations[:3])
    next_actions = []
    for item in recommendations[:3]:
        docs = item.get("required_documents") or []
        doc_text = ", ".join(docs[:3]) if docs else "basic identity documents"
        next_actions.append(f"Prepare for {item['scheme_name']}: {doc_text}")

    return {
        "headline": f"Budget impact snapshot for {occupation}",
        "summary": (
            f"For a {occupation} in {state}, top relevant support opportunities are: {top_names}. "
            "These align with your current profile and likely eligibility filters."
        ),
        "next_actions": next_actions,
    }


def _build_citizen_report(
    profile: Dict[str, Any],
    summary: Dict[str, Any],
    recommendations: List[Dict[str, Any]],
    budget_context: Dict[str, Any],
) -> Dict[str, Any]:
    generated_at = datetime.now(timezone.utc).isoformat()

    report_lines = [
        "BharatPolicy AI - Citizen Benefit Report",
        f"Generated at (UTC): {generated_at}",
        "",
        "Profile",
        f"- Occupation: {profile.get('occupation', 'citizen')}",
        f"- State: {profile.get('state', 'India')}",
        f"- Age: {profile.get('age', 'n/a')}",
        f"- Income: {profile.get('income', 'n/a')}",
        "",
        "Summary",
        f"- Headline: {summary.get('headline', '')}",
        f"- Overview: {summary.get('summary', '')}",
        "",
        "Recommended Schemes",
    ]

    for item in recommendations[:5]:
        report_lines.append(
            f"- {item.get('scheme_name', 'Scheme')} (score: {item.get('score', 0)})"
        )
        eligibility = item.get("eligibility", {})
        report_lines.append(
            f"  eligibility: {'eligible' if eligibility.get('is_eligible') else 'check required'}"
        )
        report_lines.append(f"  apply: {item.get('apply_link', '')}")

    report_lines.extend(
        [
            "",
            "Budget Context",
            budget_context.get("answer", ""),
            "",
            "Citations",
        ]
    )
    for citation in budget_context.get("citations", []):
        report_lines.append(f"- {citation}")

    return {
        "generated_at": generated_at,
        "filename": f"citizen_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        "text": "\n".join(report_lines),
    }


def create_app() -> Flask:
    settings = get_settings()
    app = Flask(__name__)
    CORS(app)

    rag_service = RAGService(settings)
    recommender = SchemeRecommender(settings.data_dir / "schemes_master.csv")

    @app.get("/health")
    def health() -> Any:
        return _success_response(
            {
                "status": "ok",
                "rag_ready": rag_service.ready,
                "indexed_chunks": rag_service.indexed_chunks,
                "schemes_loaded": recommender.scheme_count,
            }
        )

    @app.post("/api/ask-budget")
    def ask_budget() -> Any:
        payload = _parse_json_payload()
        question = str(payload.get("question", "")).strip()
        language = str(payload.get("language", "en")).strip().lower()

        if not question:
            return _error_response("question is required", code="VALIDATION_ERROR")

        if language not in {"en", "hi"}:
            return _error_response(
                "language must be one of: en, hi",
                code="VALIDATION_ERROR",
            )

        result = rag_service.answer(question=question, language=language)
        return _success_response(result)

    @app.post("/api/recommend-schemes")
    def recommend_schemes() -> Any:
        payload = _parse_json_payload()
        profile = payload.get("profile") if isinstance(payload.get("profile"), dict) else payload
        top_n = _parse_top_n(payload.get("top_n", 5))

        if not isinstance(profile, dict):
            return _error_response("profile must be an object", code="VALIDATION_ERROR")

        recommendations = recommender.recommend(
            profile=profile,
            top_n=top_n,
        )
        return _success_response(
            {
                "count": len(recommendations),
                "recommendations": recommendations,
            }
        )

    @app.post("/api/impact-summary")
    def impact_summary() -> Any:
        payload = _parse_json_payload()
        profile = payload.get("profile") if isinstance(payload.get("profile"), dict) else {}

        recommendations = payload.get("recommendations")
        if recommendations is not None and not isinstance(recommendations, list):
            return _error_response(
                "recommendations must be an array when provided",
                code="VALIDATION_ERROR",
            )

        if recommendations is None:
            recommendations = recommender.recommend(profile=profile, top_n=3, min_score=2)

        summary = _build_impact_summary(profile, recommendations)

        occupation = str(profile.get("occupation", "citizen")).strip() or "citizen"
        state = str(profile.get("state", "India")).strip() or "India"
        budget_query = f"Key budget priorities for {occupation} in {state}"
        budget_context = rag_service.answer(budget_query, language="en")

        return _success_response(
            {
                **summary,
                "budget_context": budget_context.get("answer", ""),
                "citations": budget_context.get("citations", []),
                "confidence": budget_context.get("confidence", {}),
                "notice": budget_context.get("notice", ""),
                "report": _build_citizen_report(
                    profile=profile,
                    summary=summary,
                    recommendations=recommendations,
                    budget_context=budget_context,
                ),
            }
        )

    @app.post("/api/compare-schemes")
    def compare_schemes() -> Any:
        payload = _parse_json_payload()
        scheme_ids = payload.get("scheme_ids", [])
        profile = payload.get("profile") if isinstance(payload.get("profile"), dict) else {}

        if not isinstance(scheme_ids, list) or not scheme_ids:
            return _error_response(
                "scheme_ids must be a non-empty array",
                code="VALIDATION_ERROR",
            )

        if len(scheme_ids) > 5:
            return _error_response(
                "scheme_ids can include at most 5 items",
                code="VALIDATION_ERROR",
            )

        comparison = recommender.compare_schemes(scheme_ids=scheme_ids, profile=profile)
        return _success_response(
            {
                "count": len(comparison),
                "comparisons": comparison,
            }
        )

    @app.post("/api/reload-index")
    def reload_index() -> Any:
        rag_service.refresh_index()
        return _success_response(
            {
                "rag_ready": rag_service.ready,
                "indexed_chunks": rag_service.indexed_chunks,
            }
        )

    LOGGER.info("Backend ready on %s:%s", settings.backend_host, settings.backend_port)
    return app


app = create_app()


if __name__ == "__main__":
    runtime_settings = get_settings()
    app.run(
        host=runtime_settings.backend_host,
        port=runtime_settings.backend_port,
        debug=True,
    )
