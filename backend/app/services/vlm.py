"""VLM adapter. Stub by default; Anthropic when PH_VLM_PROVIDER=anthropic.

The adapter never raises into the request path. On failure it returns None or an empty list
and sets last_error, so the pipeline falls back to the template answer and says why.

Environment variables:
    PH_VLM_PROVIDER            stub (default) or anthropic
    ANTHROPIC_API_KEY          your key (never commit it)
    PH_VLM_MODEL               model name, chosen from the Anthropic model list
    PH_VLM_ALLOW_REAL_PHOTOS   yes/no (default no). Farmer uploads are sent to the API only when yes.
    PH_VLM_TIMEOUT_S           request timeout in seconds (default 30)
"""
import base64
import io
import json
import os
import re
from pathlib import Path

from PIL import Image

from app.config import (
    MAX_ALTERNATIVES, MAX_CANDIDATE_CAUSES, MAX_GENERAL_KNOWLEDGE, MAX_OBSERVATIONS,
    MAX_QUESTIONS, UPLOAD_DIR, VLM_JPEG_QUALITY, VLM_MAX_SIDE_PX, VLM_MAX_TOKENS,
    VLM_PROVIDER, VLM_TIMEOUT_S,
)
from app.services.policy import scrub

GENERIC = {"leaves", "leaf", "shows", "show", "with", "that", "this", "from", "affected",
           "appear", "appears", "observed", "some", "more", "than", "which", "their", "there",
           "plant", "lower", "upper", "across", "surface", "visible", "present", "remain",
           "remains", "less", "minimal", "middle", "older", "young", "canopy", "foliage", "also"}


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]{4,}", text.lower()) if w not in GENERIC}


def _split_ungrounded(items, corpus: set[str], general: list[str]) -> list[str]:
    """Sentences with no word in the observations or reference text are not grounded: move them.
    Heuristic only (word overlap). It catches invented details, not every paraphrase."""
    if not corpus:
        return [str(x) for x in items or []]  # nothing to check against
    kept = []
    for x in items or []:
        s = str(x)
        w = _words(s)
        (kept if (not w or w & corpus) else general).append(s)
    return kept

PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"


def _load_prompt(name: str) -> str:
    return (PROMPTS_DIR / name).read_text(encoding="utf-8")


def _parse_json(text: str) -> dict | None:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.S)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _image_block(path: str) -> dict:
    img = Image.open(path).convert("RGB")
    img.thumbnail((VLM_MAX_SIDE_PX, VLM_MAX_SIDE_PX))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=VLM_JPEG_QUALITY)  # re-encoding also drops EXIF
    return {"type": "image", "source": {
        "type": "base64", "media_type": "image/jpeg",
        "data": base64.b64encode(buf.getvalue()).decode("ascii")}}


def _uploads_allowed(image_paths: list[str]) -> bool:
    if os.getenv("PH_VLM_ALLOW_REAL_PHOTOS", "no").lower() in {"yes", "true", "1"}:
        return True
    root = UPLOAD_DIR.resolve()
    return not any(Path(p).resolve().is_relative_to(root) for p in image_paths)


def _clean(items, limit: int) -> list[str]:
    """Strings from the model, with chemical names and doses removed unless reviewed."""
    out = []
    for x in items or []:
        clean, _ = scrub(str(x))
        out.append(clean)
    return out[:limit]


HEDGE = re.compile(r"\b(typically|usually|often|generally|normally|commonly)\b", re.IGNORECASE)
JARGON = re.compile(r"\b(microscop\w*|spore\w*|laborator\w*|lab|pcr|culture|magnification|serolog\w*)\b",
                    re.IGNORECASE)
FITS = {"strong", "partial", "poor"}


def _split_hedged(items, general: list[str]) -> list[str]:
    """Hedged statements ("typically ...") are model knowledge, not observations: move them."""
    kept = []
    for x in items or []:
        s = str(x)
        (general if HEDGE.search(s) else kept).append(s)
    return kept


def _normalise_draft(data: dict, outcome: str, allowed_names: set[str], valid_ids: set[str],
                     corpus: set[str] | None = None) -> dict:
    causes = []
    if outcome != "unknown_cause":
        for c in data.get("candidate_causes") or []:
            if not isinstance(c, dict) or str(c.get("name", "")) not in allowed_names:
                continue  # drop any cause that is not in the evidence
            general: list[str] = []
            corpus = corpus or set()
            support = _split_hedged(_split_ungrounded(c.get("support"), corpus, general), general)
            against = _split_hedged(_split_ungrounded(c.get("against_or_missing"), corpus, general), general)
            general += [str(x) for x in (c.get("general_knowledge") or [])]
            fit = str(c.get("fit", "partial")).lower()
            causes.append({
                "name": str(c["name"]),
                "fit": fit if fit in FITS else "partial",
                "support": _clean(support, 5),
                "against_or_missing": _clean(against, 5),
                "general_knowledge": _clean(general, MAX_GENERAL_KNOWLEDGE),
                "evidence_ids": [str(i) for i in (c.get("evidence_ids") or []) if str(i) in valid_ids],
            })
    questions = [str(q) for q in (data.get("questions") or []) if not JARGON.search(str(q))]
    return {
        "candidate_causes": causes[:MAX_CANDIDATE_CAUSES],
        "alternatives": [str(x) for x in (data.get("alternatives") or []) if str(x) in allowed_names][:MAX_ALTERNATIVES],
        "questions": questions[:MAX_QUESTIONS],
        "next_step": str(data.get("next_step") or "Confirm with a local plant health expert."),
    }


class StubVLM:
    """Placeholder used when no provider is configured."""
    name = "stub"
    model_name = "stub"
    prompt_version = "none"
    config_error = None

    def errors(self):
        return []

    def observe(self, image_paths, text, meta):
        if not image_paths:
            return []
        return [f"{len(image_paths)} image(s) received. Visual analysis not configured yet."]

    def synthesize(self, meta, text, observations, evidence, outcome, case_summary=None):
        return None  # None -> pipeline uses the template answer


class AnthropicVLM:
    name = "anthropic"
    prompt_version = "observe-v1+synthesize-v4"

    def __init__(self):
        import anthropic  # imported here so the stub works without the package
        key = os.getenv("ANTHROPIC_API_KEY")
        self.model_name = os.getenv("PH_VLM_MODEL")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        if not self.model_name:
            raise RuntimeError("PH_VLM_MODEL is not set (choose a model name from the Anthropic docs)")
        self.client = anthropic.Anthropic(api_key=key, timeout=VLM_TIMEOUT_S, max_retries=2)
        self.config_error = None
        self.observe_error = None
        self.synthesize_error = None

    def errors(self) -> list[str]:
        return [e for e in (self.observe_error, self.synthesize_error) if e]

    @property
    def last_error(self) -> str | None:
        return " | ".join(self.errors()) or None

    def _call(self, system: str, content: list) -> str:
        resp = self.client.messages.create(
            model=self.model_name, max_tokens=VLM_MAX_TOKENS, system=system,
            messages=[{"role": "user", "content": content}],
        )
        return "".join(getattr(b, "text", "") for b in resp.content if getattr(b, "type", "") == "text")

    def observe(self, image_paths, text, meta):
        self.observe_error = None
        if not image_paths:
            return []
        if not _uploads_allowed(image_paths):
            self.observe_error = ("Farmer photos were not sent to the VLM: consent and approval are not in place "
                               "(set PH_VLM_ALLOW_REAL_PHOTOS=yes only for approved testing).")
            return []
        try:
            content = [_image_block(p) for p in image_paths]
            content.append({"type": "text", "text": f"Crop: {meta.get('crop') or 'unknown'}. "
                                                    f"Farmer's question: {text or 'none'}."})
            data = _parse_json(self._call(_load_prompt("observe_v1.txt"), content))
            if not data or not isinstance(data.get("observations"), list):
                raise ValueError("invalid observation JSON")
            return [str(x) for x in data["observations"][:MAX_OBSERVATIONS]]
        except Exception as e:
            self.observe_error = f"Visual observation unavailable ({type(e).__name__}); template answer used."
            return []

    def synthesize(self, meta, text, observations, evidence, outcome, case_summary=None):
        self.synthesize_error = None
        if not evidence:
            return None
        allowed = {e.get("condition") or e["title"] for e in evidence}
        valid_ids = {e["id"] for e in evidence}
        corpus = _words(" ".join(observations or []) + " " + " ".join(
            e.get("reference_text") or "" for e in evidence))
        payload = {
            "outcome": outcome,
            "crop": meta.get("crop"),
            "question": text,
            "case_summary": case_summary or {},
            "observations": observations,
            "evidence": [{
                "id": e["id"],
                "condition": e.get("condition") or e["title"],
                "kind": e["type"],
                "source": e.get("source_title") or e.get("source_id"),
                "review_status": e.get("review_status"),
                "reference_text": e.get("reference_text") or "",
            } for e in evidence],
        }
        try:
            raw = self._call(_load_prompt("synthesize-v5.txt"),
                             [{"type": "text", "text": json.dumps(payload, ensure_ascii=False)}])
            data = _parse_json(raw)
            if not data:
                raise ValueError("invalid synthesis JSON")
            return _normalise_draft(data, outcome, allowed, valid_ids, corpus)
        except Exception as e:
            self.synthesize_error = f"Answer synthesis unavailable ({type(e).__name__}); template answer used."
            return None


def get_vlm():
    provider = VLM_PROVIDER
    if provider == "stub":
        return StubVLM()
    if provider == "anthropic":
        try:
            return AnthropicVLM()
        except Exception as e:
            stub = StubVLM()
            stub.config_error = f"VLM not configured ({e}); template answer used."
            return stub
    raise RuntimeError(f"Unknown PH_VLM_PROVIDER: {provider}")