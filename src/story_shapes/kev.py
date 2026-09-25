"""Ask a System One endpoint (Kev locally, or Jev) about each passage."""

import os

import httpx

# Levels described as outcomes, not feelings: worded with emotions ("grief", "joy") they pulled the
# model towards the tone of the passage (ALE-199: 65 % → 90 % on tone traps in development).
FORTUNE_LEVELS = [
    "Disaster: they lose what matters most to them, or face ruin or death",
    "Setback: something goes wrong for them",
    "No real change in their situation",
    "Gain: something goes right for them",
    "Triumph: they get what they most wanted, or are saved",
]
TENSION_LEVELS = ["Calm", "Some unease", "Tense", "Extreme danger or climax"]


def questions(protagonist: str) -> dict:
    return {
        "fortune": {
            "type": "score",
            "instructions": f"At this point in the story, how are things going for {protagonist}?",
            "criteria": FORTUNE_LEVELS,
        },
        "tension": {
            "type": "score",
            "instructions": "How tense or suspenseful is this passage?",
            "criteria": TENSION_LEVELS,
        },
        "present": {
            "type": "noul",
            "instructions": f"Does {protagonist} appear or act in this passage?",
        },
    }


def state(protagonist: str, passage: str) -> dict:
    # No title or author on purpose: the model should judge the passage, not recall the ending.
    return {"protagonist": protagonist, "passage": passage}


class SystemOne:
    def __init__(self, base_url: str | None = None, api_key: str | None = None, model: str = "kev-latest"):
        base_url = base_url or os.environ.get("STORY_SHAPES_API", "http://127.0.0.1:8009")
        api_key = api_key or os.environ.get("STORY_SHAPES_KEY", "local")
        self.model = model
        self._http = httpx.Client(
            base_url=base_url, timeout=300, headers={"authorization": f"Bearer {api_key}"}
        )

    def model_card(self) -> dict | None:
        """What the endpoint says about the model behind `self.model`, if it says anything."""
        try:
            response = self._http.get("/v1/models")
            response.raise_for_status()
        except httpx.HTTPError:
            return None
        cards = response.json().get("models", [])
        return next((card for card in cards if card.get("name") == self.model), None)

    def decide(self, state: dict | str, questions: dict) -> dict:
        response = self._http.post(
            "/v1/systemone", json={"state": state, "model": self.model, "questions": questions}
        )
        response.raise_for_status()
        return response.json()
