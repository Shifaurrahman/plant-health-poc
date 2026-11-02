"""Ask for extra views only when they would change the analysis, and say why."""
from app.config import MAX_QUESTIONS

VIEWS = [
    {
        "question": "Can you add a whole-plant photo?",
        "why": "A whole-plant view shows whether symptoms are local or spread through the canopy.",
    },
    {
        "question": "Can you add a close-up of an affected leaf filling the frame?",
        "why": "A close-up shows lesion colour, shape and pattern used to separate similar causes.",
    },
    {
        "question": "Can you add a photo of the underside of an affected leaf?",
        "why": "The underside can show pests, eggs, or spotting that is not visible from above.",
    },
]


def quality_questions(image_count: int, quality_issues: list[list[str]], outcome: str) -> list[str]:
    questions: list[str] = []
    if quality_issues and all(quality_issues):
        questions.append(
            "Please upload a sharper photo of the affected part, in focus and filling the frame. "
            "A blurry or tiny image cannot be compared with reference photos."
        )
        return questions
    if outcome in {"insufficient_image", "unknown_cause", "conflict", "no_match", "healthy_appearance"}:
        start = 0 if image_count < 2 else 1
        for view in VIEWS[start:]:
            questions.append(f"{view['question']} {view['why']}")
    return questions[:MAX_QUESTIONS]


def capture_guidance() -> list[dict]:
    return VIEWS
