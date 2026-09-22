from unittest.mock import patch

from backend.app import create_app


def test_health_endpoint() -> None:
    app = create_app()
    client = app.test_client()

    response = client.get("/health")

    assert response.status_code == 200
    body = response.get_json()
    assert body["ok"] is True
    assert body["status"] == "ok"


def test_recommend_endpoint_with_profile() -> None:
    app = create_app()
    client = app.test_client()

    response = client.post(
        "/api/recommend-schemes",
        json={
            "profile": {
                "age": 35,
                "income": 180000,
                "occupation": "farmer",
                "state": "maharashtra",
                "tags": ["farming", "credit"],
            }
        },
    )

    assert response.status_code == 200
    body = response.get_json()
    assert body["ok"] is True
    assert "recommendations" in body


def test_ask_budget_requires_question() -> None:
    app = create_app()
    client = app.test_client()

    response = client.post("/api/ask-budget", json={"language": "en"})

    assert response.status_code == 400
    body = response.get_json()
    assert body["ok"] is False
    assert body["error"]["code"] == "VALIDATION_ERROR"


def test_ask_budget_rejects_invalid_language() -> None:
    app = create_app()
    client = app.test_client()

    response = client.post(
        "/api/ask-budget",
        json={"question": "What changed?", "language": "fr"},
    )

    assert response.status_code == 400
    body = response.get_json()
    assert body["ok"] is False
    assert body["error"]["code"] == "VALIDATION_ERROR"


def test_impact_summary_rejects_invalid_recommendations_type() -> None:
    app = create_app()
    client = app.test_client()

    response = client.post(
        "/api/impact-summary",
        json={"profile": {"occupation": "student"}, "recommendations": "invalid"},
    )

    assert response.status_code == 400
    body = response.get_json()
    assert body["ok"] is False
    assert body["error"]["code"] == "VALIDATION_ERROR"


def test_ask_budget_response_contract() -> None:
    """Verify ask-budget returns expected response structure."""
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
        app = create_app()
        client = app.test_client()

        response = client.post(
            "/api/ask-budget",
            json={"question": "What is the budget?", "language": "en"},
        )

    assert response.status_code == 200
    body = response.get_json()
    assert body["ok"] is True
    assert "answer" in body
    assert "citations" in body
    assert isinstance(body.get("citations"), list)
    assert "confidence" in body
    assert "label" in body["confidence"]
    assert "score" in body["confidence"]


def test_recommend_schemes_empty_profile() -> None:
    """Verify recommend-schemes handles empty profile gracefully."""
    app = create_app()
    client = app.test_client()

    response = client.post(
        "/api/recommend-schemes",
        json={"profile": {}},
    )

    assert response.status_code == 200
    body = response.get_json()
    assert body["ok"] is True
    assert "recommendations" in body
    assert isinstance(body["recommendations"], list)
    assert "count" in body


def test_impact_summary_auto_computes_recommendations() -> None:
    """Verify impact-summary generates recommendations if not provided."""
    app = create_app()
    client = app.test_client()

    response = client.post(
        "/api/impact-summary",
        json={"profile": {"occupation": "student", "age": 22, "state": "delhi"}},
    )

    assert response.status_code == 200
    body = response.get_json()
    assert body["ok"] is True
    assert "headline" in body
    assert "summary" in body
    assert "next_actions" in body
    assert "report" in body
    assert "text" in body["report"]


def test_compare_schemes_endpoint() -> None:
    app = create_app()
    client = app.test_client()

    response = client.post(
        "/api/compare-schemes",
        json={
            "scheme_ids": ["SCH001", "SCH002"],
            "profile": {
                "age": 35,
                "income": 180000,
                "occupation": "farmer",
                "state": "maharashtra",
                "tags": ["farming", "credit"],
            },
        },
    )

    assert response.status_code == 200
    body = response.get_json()
    assert body["ok"] is True
    assert body["count"] >= 1
    assert isinstance(body.get("comparisons"), list)
