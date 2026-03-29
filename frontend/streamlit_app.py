from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:5000")
REQUEST_TIMEOUT_SECONDS = 30

PERSONA_PRESETS = {
    "Student - Uttar Pradesh": {
        "age": 22,
        "income": 180000,
        "occupation": "student",
        "state": "uttar pradesh",
        "tags": ["education", "scholarship"],
    },
    "Farmer - Maharashtra": {
        "age": 39,
        "income": 140000,
        "occupation": "farmer",
        "state": "maharashtra",
        "tags": ["agriculture", "credit"],
    },
    "MSME Owner - Gujarat": {
        "age": 33,
        "income": 450000,
        "occupation": "msme",
        "state": "gujarat",
        "tags": ["business", "loan"],
    },
    "Senior Citizen - Delhi": {
        "age": 67,
        "income": 300000,
        "occupation": "senior citizen",
        "state": "delhi",
        "tags": ["health", "pension"],
    },
}

DEFAULT_PROFILE = {
    "age": 30,
    "income": 300000,
    "occupation": "student",
    "state": "uttar pradesh",
    "tags": ["education"],
}


def _api_get(path: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    try:
        response = requests.get(
            f"{BACKEND_URL}{path}",
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        if response.status_code >= 400:
            return None, f"HTTP {response.status_code}: {response.text}"
        payload = response.json()
        if not payload.get("ok", False):
            error = payload.get("error", {})
            return None, str(error.get("message", "Unknown error"))
        data = {key: value for key, value in payload.items() if key != "ok"}
        return data, None
    except requests.RequestException as exc:
        return None, str(exc)


def _api_post(path: str, payload: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    try:
        response = requests.post(
            f"{BACKEND_URL}{path}",
            json=payload,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        if response.status_code >= 400:
            return None, f"HTTP {response.status_code}: {response.text}"
        body = response.json()
        if not body.get("ok", False):
            error = body.get("error", {})
            return None, str(error.get("message", "Unknown error"))
        data = {key: value for key, value in body.items() if key != "ok"}
        return data, None
    except requests.RequestException as exc:
        return None, str(exc)


def _init_state() -> None:
    if "profile" not in st.session_state:
        st.session_state.profile = DEFAULT_PROFILE.copy()
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "recommendations" not in st.session_state:
        st.session_state.recommendations = []
    if "impact_summary" not in st.session_state:
        st.session_state.impact_summary = None
    if "scheme_comparison" not in st.session_state:
        st.session_state.scheme_comparison = []


def _render_chat() -> None:
    st.title("Smart Budget Chat")
    st.write("Ask budget questions in simple language and get source-backed answers.")

    language = st.selectbox("Language", ["English", "Hindi-friendly"], index=0)
    question = st.text_area(
        "Your question",
        placeholder="Example: What budget benefits are relevant for farmers this year?",
        height=90,
    )

    if st.button("Ask Budget", use_container_width=True):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            response, error = _api_post(
                "/api/ask-budget",
                {
                    "question": question.strip(),
                    "language": "hi" if language.startswith("Hindi") else "en",
                },
            )
            if error:
                st.error(f"Could not reach backend: {error}")
            else:
                st.session_state.chat_history.insert(
                    0,
                    {
                        "question": question.strip(),
                        "answer": response.get("answer", "No answer returned."),
                        "citations": response.get("citations", []),
                        "confidence": response.get("confidence", {}),
                        "notice": response.get("notice", ""),
                    },
                )

    if st.session_state.chat_history:
        st.subheader("Recent Answers")
        for item in st.session_state.chat_history[:4]:
            with st.container(border=True):
                st.markdown(f"**Q:** {item['question']}")
                st.write(item["answer"])
                citations = item.get("citations", [])
                if citations:
                    st.caption("Sources: " + " | ".join(citations))
                confidence = item.get("confidence", {})
                if confidence:
                    st.caption(
                        f"Confidence: {confidence.get('label', 'low')} ({confidence.get('score', 0.0):.2f})"
                    )
                if item.get("notice"):
                    st.warning(item["notice"])


def _render_profile() -> None:
    st.title("User Profile")
    st.write("Capture quick profile details for personalized scheme matching.")

    preset_names = ["Custom"] + list(PERSONA_PRESETS.keys())
    selected_preset = st.selectbox("Persona preset", preset_names, index=0)
    if selected_preset != "Custom":
        if st.button("Load preset", use_container_width=True):
            st.session_state.profile = PERSONA_PRESETS[selected_preset].copy()
            st.success(f"Loaded preset: {selected_preset}")

    profile = st.session_state.profile
    with st.form("profile_form"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", min_value=16, max_value=100, value=int(profile.get("age", 30)))
            occupation = st.text_input("Occupation", value=str(profile.get("occupation", "student")))
        with col2:
            income = st.number_input(
                "Annual income (INR)",
                min_value=0,
                max_value=10000000,
                value=int(profile.get("income", 300000)),
                step=10000,
            )
            state = st.text_input("State", value=str(profile.get("state", "uttar pradesh")))

        tags_text = st.text_input(
            "Tags (comma separated)",
            value=", ".join(profile.get("tags", [])),
            help="Example: education, scholarship, agriculture, business",
        )

        if st.form_submit_button("Save Profile", use_container_width=True):
            tags = [tag.strip().lower() for tag in tags_text.split(",") if tag.strip()]
            st.session_state.profile = {
                "age": int(age),
                "income": float(income),
                "occupation": occupation.strip().lower(),
                "state": state.strip().lower(),
                "tags": tags,
            }
            st.success("Profile saved.")


def _render_recommendations() -> None:
    st.title("Scheme Recommendations")
    profile = st.session_state.profile

    st.info(
        f"Current profile: {profile.get('occupation', 'n/a')} | "
        f"{profile.get('state', 'n/a')} | income {int(profile.get('income', 0))}"
    )

    if st.button("Find My Schemes", use_container_width=True):
        response, error = _api_post(
            "/api/recommend-schemes",
            {"profile": profile, "top_n": 5},
        )
        if error:
            st.error(f"Could not fetch recommendations: {error}")
        else:
            st.session_state.recommendations = response.get("recommendations", [])

    recommendations = st.session_state.recommendations
    if not recommendations:
        st.warning("No recommendations yet. Click 'Find My Schemes'.")
        return

    for recommendation in recommendations:
        with st.container(border=True):
            st.subheader(recommendation.get("scheme_name", "Scheme"))
            st.write(recommendation.get("benefit_summary", ""))
            st.write(f"Match score: {recommendation.get('score', 0)}")

            eligibility = recommendation.get("eligibility", {})
            status = "Eligible" if eligibility.get("is_eligible") else "Needs review"
            st.write(f"Eligibility status: {status}")
            blockers = eligibility.get("blockers", [])
            if blockers:
                st.caption("Eligibility blockers: " + ", ".join(blockers))

            why_matched = recommendation.get("why_matched", [])
            if why_matched:
                st.caption("Why matched: " + ", ".join(why_matched))

            docs = recommendation.get("required_documents", [])
            if docs:
                st.write("Required documents: " + ", ".join(docs))

            apply_link = recommendation.get("apply_link", "")
            if apply_link:
                st.markdown(f"Apply: {apply_link}")

            checklist = recommendation.get("action_checklist", [])
            if checklist:
                st.markdown("**Action checklist**")
                for step in checklist:
                    st.write(f"- {step}")

    st.markdown("---")
    st.subheader("Compare Schemes")
    option_map = {
        f"{item.get('scheme_name', 'Scheme')} ({item.get('scheme_id', '')})": item.get("scheme_id", "")
        for item in recommendations
    }
    selected_labels = st.multiselect(
        "Select up to 3 schemes to compare",
        options=list(option_map.keys()),
        max_selections=3,
    )

    if st.button("Compare Selected", use_container_width=True):
        selected_ids = [option_map[label] for label in selected_labels if option_map.get(label)]
        if len(selected_ids) < 2:
            st.warning("Select at least 2 schemes to compare.")
        else:
            response, error = _api_post(
                "/api/compare-schemes",
                {"scheme_ids": selected_ids, "profile": profile},
            )
            if error:
                st.error(f"Could not compare schemes: {error}")
            else:
                st.session_state.scheme_comparison = response.get("comparisons", [])

    if st.session_state.scheme_comparison:
        comparison_rows = []
        for item in st.session_state.scheme_comparison:
            comparison_rows.append(
                {
                    "scheme": item.get("scheme_name", ""),
                    "score": item.get("score", 0),
                    "eligible": item.get("eligibility", {}).get("is_eligible", False),
                    "documents": ", ".join(item.get("required_documents", [])[:3]),
                    "apply_link": item.get("apply_link", ""),
                }
            )
        st.dataframe(comparison_rows, use_container_width=True)

    st.markdown("---")
    st.subheader("What-If Simulator")
    with st.expander("Try profile changes", expanded=False):
        base_profile = st.session_state.profile
        sim_age = st.number_input(
            "Simulated age",
            min_value=16,
            max_value=100,
            value=int(base_profile.get("age", 30)),
            key="sim_age",
        )
        sim_income = st.number_input(
            "Simulated annual income (INR)",
            min_value=0,
            max_value=10000000,
            value=int(base_profile.get("income", 300000)),
            step=10000,
            key="sim_income",
        )
        sim_occupation = st.text_input(
            "Simulated occupation",
            value=str(base_profile.get("occupation", "student")),
            key="sim_occupation",
        )
        sim_state = st.text_input(
            "Simulated state",
            value=str(base_profile.get("state", "uttar pradesh")),
            key="sim_state",
        )
        sim_tags = st.text_input(
            "Simulated tags (comma separated)",
            value=", ".join(base_profile.get("tags", [])),
            key="sim_tags",
        )

        if st.button("Run What-If", use_container_width=True):
            tags = [item.strip().lower() for item in sim_tags.split(",") if item.strip()]
            simulated_profile = {
                "age": int(sim_age),
                "income": float(sim_income),
                "occupation": sim_occupation.strip().lower(),
                "state": sim_state.strip().lower(),
                "tags": tags,
            }
            response, error = _api_post(
                "/api/recommend-schemes",
                {"profile": simulated_profile, "top_n": 5},
            )
            if error:
                st.error(f"What-if simulation failed: {error}")
            else:
                simulated = response.get("recommendations", [])
                st.write("Simulated top schemes")
                for item in simulated[:3]:
                    st.write(
                        f"- {item.get('scheme_name', '')} (score: {item.get('score', 0)}, "
                        f"eligible: {item.get('eligibility', {}).get('is_eligible', False)})"
                    )


def _render_budget_explanation() -> None:
    st.title("Budget Explanation")
    st.write("Get a concise impact snapshot for the active user profile.")

    profile = st.session_state.profile
    payload: Dict[str, Any] = {"profile": profile}
    if st.session_state.recommendations:
        payload["recommendations"] = st.session_state.recommendations

    if st.button("Generate Impact Summary", use_container_width=True):
        response, error = _api_post("/api/impact-summary", payload)
        if error:
            st.error(f"Could not generate summary: {error}")
        else:
            st.session_state.impact_summary = response

    summary = st.session_state.impact_summary
    if not summary:
        st.warning("No summary generated yet.")
        return

    st.subheader(summary.get("headline", "Impact Summary"))
    st.write(summary.get("summary", ""))

    actions = summary.get("next_actions", [])
    if actions:
        st.markdown("**Next actions**")
        for action in actions:
            st.write(f"- {action}")

    budget_context = summary.get("budget_context", "")
    if budget_context:
        st.markdown("**Budget context from source docs**")
        st.write(budget_context)

    citations = summary.get("citations", [])
    if citations:
        st.caption("Sources: " + " | ".join(citations))

    confidence = summary.get("confidence", {})
    if confidence:
        st.caption(
            f"Confidence: {confidence.get('label', 'low')} ({confidence.get('score', 0.0):.2f})"
        )

    notice = summary.get("notice", "")
    if notice:
        st.warning(notice)

    report = summary.get("report", {})
    if report and report.get("text"):
        st.download_button(
            "Download Citizen Report",
            data=report.get("text", ""),
            file_name=report.get("filename", "citizen_report.txt"),
            mime="text/plain",
            use_container_width=True,
        )


def main() -> None:
    st.set_page_config(
        page_title="BharatPolicy AI",
        page_icon="BP",
        layout="wide",
    )
    _init_state()

    st.sidebar.title("BharatPolicy AI")
    st.sidebar.caption("Samjho Budget. Paao Benefits. Har Din.")

    health_response, health_error = _api_get("/health")
    if health_error:
        st.sidebar.error("Backend offline")
        st.sidebar.caption(health_error)
    else:
        rag_status = "ready" if health_response.get("rag_ready") else "not indexed"
        st.sidebar.success("Backend online")
        st.sidebar.caption(
            f"RAG: {rag_status} | Schemes: {health_response.get('schemes_loaded', 0)}"
        )

        if st.sidebar.button("Reload Index", use_container_width=True):
            reload_response, reload_error = _api_post("/api/reload-index", {})
            if reload_error:
                st.sidebar.error(f"Reload failed: {reload_error}")
            else:
                refreshed_chunks = reload_response.get("indexed_chunks", 0)
                refreshed_ready = "ready" if reload_response.get("rag_ready") else "not indexed"
                st.sidebar.success(
                    f"Index reloaded: {refreshed_ready}, chunks: {refreshed_chunks}"
                )

    page = st.sidebar.radio(
        "Navigate",
        [
            "Smart Budget Chat",
            "User Profile",
            "Scheme Recommendations",
            "Budget Explanation",
        ],
    )

    if page == "Smart Budget Chat":
        _render_chat()
    elif page == "User Profile":
        _render_profile()
    elif page == "Scheme Recommendations":
        _render_recommendations()
    else:
        _render_budget_explanation()


if __name__ == "__main__":
    main()
