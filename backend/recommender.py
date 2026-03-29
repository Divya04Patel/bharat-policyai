from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


class SchemeRecommender:
    """Rule-based scheme ranking for fast and explainable hackathon recommendations."""

    def __init__(self, scheme_file: Path) -> None:
        self.scheme_file = Path(scheme_file)
        self.schemes_df = self._load_schemes()

    @property
    def scheme_count(self) -> int:
        return len(self.schemes_df)

    def _load_schemes(self) -> pd.DataFrame:
        if not self.scheme_file.exists():
            return pd.DataFrame()

        dataframe = pd.read_csv(self.scheme_file).fillna("")
        return dataframe

    @staticmethod
    def _parse_pipe_list(value: Any) -> List[str]:
        return [
            item.strip().lower()
            for item in str(value).split("|")
            if str(item).strip()
        ]

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            cleaned = str(value).replace(",", "").strip()
            if not cleaned:
                return default
            return float(cleaned)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _safe_int(value: Any, default: int = 0) -> int:
        try:
            cleaned = str(value).strip()
            if not cleaned:
                return default
            return int(float(cleaned))
        except (TypeError, ValueError):
            return default

    def _normalize_profile(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        raw_tags = profile.get("tags", [])
        if isinstance(raw_tags, str):
            tags = [tag.strip().lower() for tag in raw_tags.split(",") if tag.strip()]
        elif isinstance(raw_tags, list):
            tags = [str(tag).strip().lower() for tag in raw_tags if str(tag).strip()]
        else:
            tags = []

        return {
            "age": self._safe_int(profile.get("age"), 0),
            "income": self._safe_float(profile.get("income"), 0.0),
            "occupation": str(profile.get("occupation", "")).strip().lower(),
            "state": str(profile.get("state", "")).strip().lower(),
            "tags": tags,
        }

    def _score_scheme(
        self, scheme_row: pd.Series, profile: Dict[str, Any]
    ) -> Tuple[int, List[str]]:
        score = 0
        reasons: List[str] = []

        target_occupations = self._parse_pipe_list(scheme_row.get("target_occupations", ""))
        user_occupation = profile["occupation"]
        if target_occupations:
            if "all" in target_occupations or user_occupation in target_occupations:
                score += 3
                reasons.append("Occupation match")
            elif user_occupation and any(
                occ in user_occupation or user_occupation in occ for occ in target_occupations
            ):
                score += 2
                reasons.append("Partial occupation match")

        eligible_states = self._parse_pipe_list(scheme_row.get("eligible_states", ""))
        user_state = profile["state"]
        if eligible_states:
            if "all" in eligible_states or user_state in eligible_states:
                score += 2
                reasons.append("State eligible")

        income_cap = self._safe_float(scheme_row.get("income_cap", "-1"), -1.0)
        if income_cap < 0:
            score += 1
            reasons.append("No strict income cap")
        elif profile["income"] <= income_cap:
            score += 2
            reasons.append("Income criteria fit")

        min_age = self._safe_int(scheme_row.get("min_age", "0"), 0)
        max_age = self._safe_int(scheme_row.get("max_age", "200"), 200)
        if min_age <= profile["age"] <= max_age:
            score += 1
            reasons.append("Age criteria fit")

        scheme_tags = self._parse_pipe_list(scheme_row.get("tags", ""))
        if scheme_tags and set(scheme_tags).intersection(profile["tags"]):
            score += 1
            reasons.append("Need/category match")

        return score, reasons

    def _evaluate_eligibility(
        self, scheme_row: pd.Series, profile: Dict[str, Any]
    ) -> Tuple[bool, List[str], List[str]]:
        blockers: List[str] = []
        checks: List[str] = []

        target_occupations = self._parse_pipe_list(scheme_row.get("target_occupations", ""))
        user_occupation = profile["occupation"]
        if target_occupations and "all" not in target_occupations:
            if user_occupation in target_occupations:
                checks.append("Occupation eligibility passed")
            elif user_occupation and any(
                occ in user_occupation or user_occupation in occ for occ in target_occupations
            ):
                checks.append("Occupation partially aligned")
            else:
                blockers.append("Occupation not in target group")
        else:
            checks.append("Open to all occupations")

        eligible_states = self._parse_pipe_list(scheme_row.get("eligible_states", ""))
        user_state = profile["state"]
        if eligible_states and "all" not in eligible_states:
            if user_state in eligible_states:
                checks.append("State eligibility passed")
            else:
                blockers.append("Scheme not available for selected state")
        else:
            checks.append("Available across states")

        income_cap = self._safe_float(scheme_row.get("income_cap", "-1"), -1.0)
        if income_cap >= 0:
            if profile["income"] <= income_cap:
                checks.append("Income criteria passed")
            else:
                blockers.append(f"Income exceeds cap of INR {int(income_cap)}")
        else:
            checks.append("No income cap")

        min_age = self._safe_int(scheme_row.get("min_age", "0"), 0)
        max_age = self._safe_int(scheme_row.get("max_age", "200"), 200)
        age = profile["age"]
        if min_age <= age <= max_age:
            checks.append("Age criteria passed")
        else:
            blockers.append(f"Age must be between {min_age} and {max_age}")

        return len(blockers) == 0, blockers, checks

    @staticmethod
    def _build_action_checklist(
        scheme_name: str,
        required_documents: List[str],
        apply_link: str,
    ) -> List[str]:
        checklist: List[str] = []
        if required_documents:
            checklist.append(
                "Collect required documents: " + ", ".join(required_documents[:4])
            )
        checklist.append(f"Verify details and submit on portal for {scheme_name}")
        if apply_link:
            checklist.append(f"Open official link: {apply_link}")
        return checklist

    def _build_scheme_payload(
        self,
        row: pd.Series,
        profile: Dict[str, Any],
        score: Optional[int] = None,
        reasons: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        if score is None or reasons is None:
            score, reasons = self._score_scheme(row, profile)

        required_documents = self._parse_pipe_list(row.get("required_documents", ""))
        eligible, blockers, checks = self._evaluate_eligibility(row, profile)
        apply_link = str(row.get("apply_link", ""))
        scheme_name = str(row.get("scheme_name", ""))

        return {
            "scheme_id": str(row.get("scheme_id", "")),
            "scheme_name": scheme_name,
            "score": int(score),
            "why_matched": reasons,
            "benefit_summary": str(row.get("benefit_summary", "")),
            "apply_link": apply_link,
            "required_documents": required_documents,
            "eligible_states": self._parse_pipe_list(row.get("eligible_states", "")),
            "eligibility": {
                "is_eligible": eligible,
                "checks_passed": checks,
                "blockers": blockers,
            },
            "action_checklist": self._build_action_checklist(
                scheme_name=scheme_name,
                required_documents=required_documents,
                apply_link=apply_link,
            ),
        }

    def recommend(
        self, profile: Dict[str, Any], top_n: int = 5, min_score: int = 3
    ) -> List[Dict[str, Any]]:
        if self.schemes_df.empty:
            return []

        normalized_profile = self._normalize_profile(profile)
        recommendations: List[Dict[str, Any]] = []

        for _, row in self.schemes_df.iterrows():
            score, reasons = self._score_scheme(row, normalized_profile)
            if score < min_score:
                continue
            recommendations.append(
                self._build_scheme_payload(
                    row=row,
                    profile=normalized_profile,
                    score=score,
                    reasons=reasons,
                )
            )

        recommendations.sort(key=lambda item: item["score"], reverse=True)
        return recommendations[:top_n]

    def compare_schemes(
        self,
        scheme_ids: List[str],
        profile: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        if self.schemes_df.empty or not scheme_ids:
            return []

        profile_data = self._normalize_profile(profile or {})
        normalized_ids = {str(item).strip().upper() for item in scheme_ids if str(item).strip()}
        if not normalized_ids:
            return []

        filtered = self.schemes_df[
            self.schemes_df["scheme_id"].astype(str).str.upper().isin(normalized_ids)
        ]

        comparisons: List[Dict[str, Any]] = []
        for _, row in filtered.iterrows():
            comparisons.append(self._build_scheme_payload(row=row, profile=profile_data))

        comparisons.sort(key=lambda item: item["score"], reverse=True)
        return comparisons
