from app.config import VLM_PROVIDER


class StubVLM:
    """Placeholder until a real VLM is wired in. Lets the whole pipeline run end to end."""
    name = "stub"

    def observe(self, image_paths, text, meta):
        if not image_paths:
            return []
        return [f"{len(image_paths)} image(s) received. Visual analysis not configured yet."]

    def synthesize(self, meta, text, observations, evidence):
        return None  # None -> pipeline uses the retrieval-only fallback draft


# TODO: add a real adapter (e.g. hosted multimodal API) implementing observe() and synthesize().
def get_vlm():
    if VLM_PROVIDER == "stub":
        return StubVLM()
    raise RuntimeError(f"Unknown PH_VLM_PROVIDER: {VLM_PROVIDER}")
