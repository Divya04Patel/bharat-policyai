from pathlib import Path

from backend.recommender import SchemeRecommender


def test_recommendations_for_student_profile() -> None:
    recommender = SchemeRecommender(Path("data/schemes_master.csv"))
    profile = {
        "age": 22,
        "income": 150000,
        "occupation": "student",
        "state": "uttar pradesh",
        "tags": ["education", "scholarship"],
    }

    recommendations = recommender.recommend(profile=profile, top_n=5)

    assert recommendations
    top_names = [item["scheme_name"] for item in recommendations]
    assert any("Scholarship" in name or "Skill" in name for name in top_names)
