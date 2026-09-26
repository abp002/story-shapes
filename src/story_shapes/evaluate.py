"""Does the model read the protagonist's fortune, or only the tone of the words?

Three tests, with criteria fixed before running them (ALE-198):
- traps: passages where tone and fortune disagree, plus controls where they agree;
- synthetic stories whose fortune arc was decided before they were written;
- a famous book with every name changed, compared passage by passage with the original.
VADER, a word-list sentiment scorer, is the baseline throughout.
"""

import hashlib
import json
import statistics
from pathlib import Path

from story_shapes import curve, kev, metrics
from story_shapes.cli import MODEL_FIELDS, READINGS_DIR, ROOT, resume

EVAL_DIR = ROOT / "eval"
RESULTS_DIR = EVAL_DIR / "results"

TONE = {
    "type": "score",
    "instructions": "Setting aside what happens to anyone, how positive or negative is the tone of the language in this passage?",
    "criteria": ["Very negative", "Negative", "Neutral", "Positive", "Very positive"],
}
TRAP_KINDS = ("cheerful-bad", "gloomy-good")
CONTROL_KINDS = ("congruent-bad", "congruent-good")
BANDWIDTHS = [1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24]
CRITERIA = {
    "C1": "Kev reads the direction of fortune right on ≥ 75 % of tone traps",
    "C2": "Kev and VADER are both right on ≥ 85 % of controls",
    "C3": "Median Spearman ρ ≥ 0.8 on synthetic stories, and the right shape for ≥ 10 of 12",
    "C4": "Renamed Romeo and Juliet correlates r ≥ 0.9 with the original, passage by passage",
}


def load_traps(folder: str = "traps") -> list[dict]:
    """Tone traps and controls: `traps` is the test set, read once per question; `dev` is for tuning."""
    items = []
    for path in sorted((EVAL_DIR / folder).glob("*.jsonl")):
        items += [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return items


# The first wording (ALE-178/198), kept so the development results stay reproducible.
FEELING_LEVELS = [
    "Very badly: ruin, grief, terror or despair",
    "Badly",
    "Neither well nor badly",
    "Well",
    "Very well: joy, love, triumph or relief",
]
OUTCOME_LEVELS = kev.FORTUNE_LEVELS


def _fortune(instructions: str, levels: list[str]) -> dict:
    return {"fortune": {"type": "score", "instructions": instructions, "criteria": levels}}


# Wordings of the fortune question tried on the development set. Questions are isolated in Kev
# (they cannot read each other), so asking fortune alone gives the same answer as asking it
# alongside tension and presence.
VARIANTS = {
    "v0-feelings": lambda p: _fortune(f"At this point in the story, how are things going for {p}?", FEELING_LEVELS),
    "v1-outcomes": lambda p: _fortune(f"At this point in the story, how are things going for {p}?", OUTCOME_LEVELS),
    "v2-outcomes-not-mood": lambda p: _fortune(
        f"Judge only what is happening to {p}, not the mood of the scene, the weather or the setting. "
        f"At this point in the story, how are things going for {p}?",
        OUTCOME_LEVELS,
    ),
    "v3-events": lambda p: {
        "good": {"type": "noul", "instructions": f"In this passage, does something good happen to {p}, or does {p} learn of something good?"},
        "bad": {"type": "noul", "instructions": f"In this passage, does something bad happen to {p}, or does {p} learn of something bad?"},
    },
}


def fortune_of(answers: dict) -> float:
    """Signed fortune from any variant: the score on a -1..1 scale, or P(good) - P(bad) for v3."""
    if "fortune" in answers:
        return signed(answers, "fortune")
    return answers["good"]["noul"] - answers["bad"]["noul"]


def all_variants(protagonist: str) -> dict:
    """Every variant's questions in one request, ids prefixed with the variant.

    Kev reads the passage once and answers each question on its own, so this gives the same
    answers as one request per variant at a fraction of the cost.
    """
    return {
        f"{name}/{question_id}": question
        for name, questions_for in VARIANTS.items()
        for question_id, question in questions_for(protagonist).items()
    }


def of_variant(answers: dict, name: str) -> dict:
    return {key.split("/", 1)[1]: value for key, value in answers.items() if key.startswith(f"{name}/")}


def tune(client, fresh: bool = False) -> dict:
    """Try every wording on the development set; the test set is not touched."""
    dev = load_traps("dev")
    answers = ask(dev, client, RESULTS_DIR / "dev.jsonl", all_variants, fresh)
    results = {}
    for name in VARIANTS:
        row = {}
        for label, kinds in [(kind, (kind,)) for kind in TRAP_KINDS + CONTROL_KINDS] + [("traps", TRAP_KINDS)]:
            items = [item for item in dev if item["kind"] in kinds]
            row[label] = metrics.direction_accuracy(
                [fortune_of(of_variant(answers[item["id"]], name)) for item in items],
                [curve.to_unit(item["fortune"], 5) for item in items],
            )
        results[name] = row
    (RESULTS_DIR / "dev.json").write_text(json.dumps(results, indent=2) + "\n")
    return results


def load_deaths() -> list[dict]:
    """Development stories with a known passage of death (or none): real, faked, visions, pairs."""
    stories = []
    for path in sorted((EVAL_DIR / "deaths").glob("*.json")):
        stories += json.loads(path.read_text())["stories"]
    return stories


def death_questions(protagonist: str) -> dict:
    questions = kev.questions(protagonist)
    return {key: questions[key] for key in ("dead", "present")}


DEATH_RULES = [
    {"threshold": t, "hold": h, "window": w, "revoke": r}
    for t in (0.7, 0.8, 0.9) for h in (0.4, 0.5, 0.6) for w in (3, 5, 8) for r in (False, True)
]


def death_right(cut: int | None, truth: int | None) -> bool:
    """A cut is right within one passage of the real death; no death means no cut."""
    return cut is None if truth is None else cut is not None and abs(cut - truth) <= 1


def tune_deaths(client, fresh: bool = False) -> list[dict]:
    """Score every death rule on the development stories, best first. The books are not touched."""
    stories = load_deaths()
    items = [
        {"id": f"{story['id']}-{i}", "protagonist": story["protagonist"], "passage": passage}
        for story in stories for i, passage in enumerate(story["passages"])
    ]
    answers = ask(items, client, RESULTS_DIR / "deaths-dev.jsonl", death_questions, fresh)
    scored = []
    for rule in DEATH_RULES:
        right = 0
        for story in stories:
            n = len(story["passages"])
            dead = [answers[f"{story['id']}-{i}"]["dead"]["noul"] for i in range(n)]
            present = [answers[f"{story['id']}-{i}"]["present"]["noul"] for i in range(n)]
            truth = next((i for i, d in enumerate(story["dead"]) if d), None)
            right += death_right(curve.death_cut(dead, present=present, **rule), truth)
        scored.append({**rule, "right": right, "of": len(stories)})
    scored.sort(key=lambda r: -r["right"])
    (RESULTS_DIR / "deaths-dev.json").write_text(json.dumps(scored, indent=2) + "\n")
    return scored


def load_synthetic() -> tuple[list[dict], dict[str, list[int]]]:
    """Synthetic stories, and the fortune template of each shape."""
    stories, shapes = [], {}
    for path in sorted((EVAL_DIR / "synthetic").glob("*.json")):
        data = json.loads(path.read_text())
        shapes[data["shape"]] = data["levels"]
        stories += [{**story, "shape": data["shape"], "levels": data["levels"]} for story in data["stories"]]
    return stories, shapes


def synthetic_items(stories: list[dict]) -> list[dict]:
    return [
        {"id": f"{story['id']}-{i:02d}", "protagonist": story["protagonist"], "passage": passage}
        for story in stories
        for i, passage in enumerate(story["passages"])
    ]


def _questions(protagonist: str, with_tone: bool) -> dict:
    questions = kev.questions(protagonist)
    if with_tone:
        questions["tone"] = TONE
    return questions


def ask(items: list[dict], client, path: Path, questions_for, fresh: bool = False) -> dict[str, dict]:
    """The model's answers for each item (`id`, `protagonist`, `passage`), cached in `path`.

    `questions_for(protagonist)` builds the questions; its template, filled with a placeholder,
    is part of the cache check.
    """
    card = client.model_card()
    meta = {
        "story_sha256": hashlib.sha256(json.dumps(items, sort_keys=True).encode()).hexdigest(),
        "chunking": None,
        "questions": questions_for("{protagonist}"),
        "model": {key: card[key] for key in MODEL_FIELDS if key in card} if card else None,
    }
    done = resume(path, meta, "id", fresh)
    with path.open("a") as out:
        for n, item in enumerate(items, 1):
            if item["id"] in done:
                continue
            response = client.decide(kev.state(item["protagonist"], item["passage"]), questions_for(item["protagonist"]))
            out.write(json.dumps({"id": item["id"], "answers": response["answers"]}, ensure_ascii=False) + "\n")
            out.flush()
            print(f"{path.stem} {n}/{len(items)}", flush=True)
    return cached(path)


def cached(path: Path) -> dict[str, dict]:
    return {record["id"]: record["answers"] for record in map(json.loads, path.read_text().splitlines())}


def run(client, fresh: bool = False) -> None:
    ask(load_traps(), client, RESULTS_DIR / "traps.jsonl", lambda p: _questions(p, with_tone=True), fresh=fresh)
    stories, _ = load_synthetic()
    ask(synthetic_items(stories), client, RESULTS_DIR / "synthetic.jsonl", lambda p: _questions(p, with_tone=False), fresh=fresh)


def signed(answers: dict, question: str) -> float:
    """A score answer on the signed scale: -1 for the lowest level, +1 for the highest."""
    probabilities = answers[question]["probabilities"]
    return curve.to_unit(curve.mean_level(probabilities), len(probabilities))


def vader(texts: list[str]) -> list[float]:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    analyzer = SentimentIntensityAnalyzer()
    return [analyzer.polarity_scores(text)["compound"] for text in texts]


def stale_reading(book_id: str, protagonist: str) -> bool:
    """Whether a book was read with questions other than the current ones."""
    meta = json.loads((READINGS_DIR / f"{book_id}.meta.json").read_text())
    return meta.get("questions") != kev.questions(protagonist)


def _book_fortune(book_id: str) -> tuple[list[float], list[float]]:
    records = sorted(map(json.loads, (READINGS_DIR / f"{book_id}.jsonl").read_text().splitlines()), key=lambda r: r["index"])
    return [signed(r["answers"], "fortune") for r in records], [r["answers"]["present"]["noul"] for r in records]


def report() -> dict:
    """Score every test against its criterion; returns the numbers written to results/summary.json."""
    traps = load_traps()
    answers = cached(RESULTS_DIR / "traps.jsonl")
    trap_vader = vader([item["passage"] for item in traps])
    by_kind = {}
    for kind in TRAP_KINDS + CONTROL_KINDS:
        rows = [(item, score) for item, score in zip(traps, trap_vader) if item["kind"] == kind]
        truth = [curve.to_unit(item["fortune"], 5) for item, _ in rows]
        tone = [1 if item["tone"] == "positive" else -1 for item, _ in rows]
        by_kind[kind] = {
            "n": len(rows),
            "kev_fortune": metrics.direction_accuracy([signed(answers[i["id"]], "fortune") for i, _ in rows], truth),
            "vader_fortune": metrics.direction_accuracy([s for _, s in rows], truth),
            "kev_tone": metrics.direction_accuracy([signed(answers[i["id"]], "tone") for i, _ in rows], tone),
            "vader_tone": metrics.direction_accuracy([s for _, s in rows], tone),
            "kev_mean": statistics.fmean(signed(answers[i["id"]], "fortune") for i, _ in rows),
            "vader_mean": statistics.fmean(s for _, s in rows),
        }

    def pooled(kinds, field):
        total = sum(by_kind[k]["n"] for k in kinds)
        return sum(by_kind[k][field] * by_kind[k]["n"] for k in kinds) / total

    stories, shapes = load_synthetic()
    synthetic_answers = cached(RESULTS_DIR / "synthetic.jsonl")
    per_story = []
    for story in stories:
        kev_curve = [signed(synthetic_answers[f"{story['id']}-{i:02d}"], "fortune") for i in range(len(story["passages"]))]
        vader_curve = vader(story["passages"])
        per_story.append({
            "id": story["id"],
            "shape": story["shape"],
            "kev_rho": metrics.spearman(kev_curve, story["levels"]),
            "vader_rho": metrics.spearman(vader_curve, story["levels"]),
            "kev_shape": metrics.nearest_shape(kev_curve, shapes),
            "vader_shape": metrics.nearest_shape(vader_curve, shapes),
            "kev": kev_curve,
            "vader": vader_curve,
            "levels": story["levels"],
        })

    from story_shapes.cli import load_books

    books = load_books()
    stale = sorted(b for b in ("christmas-carol", "metamorphosis", "romeo-and-juliet", "romeo-and-juliet-renamed")
                   if stale_reading(b, books[b]["protagonist"]))
    original, original_present = _book_fortune("romeo-and-juliet")
    renamed, renamed_present = _book_fortune("romeo-and-juliet-renamed")
    sigma = max(1.5, 0.05 * len(original))  # the width plot.draw uses
    smoothed = [
        curve.smooth(values, [0.1 + 0.9 * p for p in present], sigma)
        for values, present in ((original, original_present), (renamed, renamed_present))
    ]

    bandwidths = {}
    for book_id in ("christmas-carol", "metamorphosis", "romeo-and-juliet"):
        fortune, present = _book_fortune(book_id)
        best, errors = metrics.loo_bandwidth(fortune, BANDWIDTHS, [0.1 + 0.9 * p for p in present])
        bandwidths[book_id] = {"passages": len(fortune), "best_sigma": best, "best_fraction": best / len(fortune), "errors": errors}

    c1 = pooled(TRAP_KINDS, "kev_fortune")
    c2 = min(pooled(CONTROL_KINDS, "kev_fortune"), pooled(CONTROL_KINDS, "vader_fortune"))
    kev_rhos = [s["kev_rho"] for s in per_story]
    right_shapes = sum(s["kev_shape"] == s["shape"] for s in per_story)
    c4 = metrics.pearson(original, renamed)
    summary = {
        "criteria": {
            "C1": {"value": c1, "passed": c1 >= 0.75},
            "C2": {"value": c2, "passed": c2 >= 0.85},
            "C3": {"value": statistics.median(kev_rhos), "right_shapes": right_shapes,
                   "passed": statistics.median(kev_rhos) >= 0.8 and right_shapes >= 10},
            "C4": {"value": c4, "passed": c4 >= 0.9, "pending": bool({"romeo-and-juliet", "romeo-and-juliet-renamed"} & set(stale))},
        },
        "traps": {"by_kind": by_kind, "kev_on_traps": c1, "vader_on_traps": pooled(TRAP_KINDS, "vader_fortune")},
        "synthetic": {
            "kev_median_rho": statistics.median(kev_rhos),
            "vader_median_rho": statistics.median(s["vader_rho"] for s in per_story),
            "kev_right_shapes": right_shapes,
            "vader_right_shapes": sum(s["vader_shape"] == s["shape"] for s in per_story),
            "stories": per_story,
        },
        "renamed": {
            "pearson": c4,
            "passages": len(original),
            "posthoc_smoothed_pearson": metrics.pearson(*smoothed),
            "posthoc_mean_shift": statistics.fmean(renamed) - statistics.fmean(original),
        },
        "bandwidth": bandwidths,
        "stale_books": stale,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    return summary


def _pct(value: float) -> str:
    return f"{100 * value:.0f} %"


def markdown(summary: dict, model: str) -> str:
    """The results page, eval/RESULTS.md."""
    c = summary["criteria"]
    lines = [
        "# Does the model read fortune, or only tone?",
        "",
        f"Model: {model}. Baseline: VADER compound score. The criteria were fixed before the run (ALE-198).",
        "",
        "| | Criterion | Result | |",
        "|---|---|---|---|",
        f"| C1 | {CRITERIA['C1']} | {_pct(c['C1']['value'])} | {'✅' if c['C1']['passed'] else '❌'} |",
        f"| C2 | {CRITERIA['C2']} | {_pct(c['C2']['value'])} (the lower of the two) | {'✅' if c['C2']['passed'] else '❌'} |",
        f"| C3 | {CRITERIA['C3']} | ρ = {c['C3']['value']:.2f}, {c['C3']['right_shapes']}/12 shapes | {'✅' if c['C3']['passed'] else '❌'} |",
        (f"| C4 | {CRITERIA['C4']} | pending: the books have not been read again with the current questions | ⏳ |"
         if c["C4"].get("pending") else
         f"| C4 | {CRITERIA['C4']} | r = {c['C4']['value']:.2f} | {'✅' if c['C4']['passed'] else '❌'} |"),
        "",
        "## Tone traps",
        "",
        "Standalone passages. In the traps, tone and fortune point opposite ways; in the controls they agree.",
        "",
        "| Kind | n | Kev reads the fortune | VADER reads the fortune | Kev reads the tone | VADER matches the tone |",
        "|---|---|---|---|---|---|",
    ]
    for kind, row in summary["traps"]["by_kind"].items():
        lines.append(
            f"| {kind} | {row['n']} | {_pct(row['kev_fortune'])} | {_pct(row['vader_fortune'])} "
            f"| {_pct(row['kev_tone'])} | {_pct(row['vader_tone'])} |"
        )
    s = summary["synthetic"]
    lines += [
        "",
        "![Tone traps: Kev's fortune against VADER's score](traps.png)",
        "",
        "## Synthetic stories",
        "",
        "Twelve stories, two per shape, written to a fortune level fixed in advance for each of their 12 passages.",
        f"Median Spearman ρ with the planned levels: Kev {s['kev_median_rho']:.2f}, VADER {s['vader_median_rho']:.2f}. "
        f"Closest shape right: Kev {s['kev_right_shapes']}/12, VADER {s['vader_right_shapes']}/12.",
        "",
        "| Story | Kev ρ | VADER ρ | Kev's shape | VADER's shape |",
        "|---|---|---|---|---|",
    ]
    for story in s["stories"]:
        lines.append(
            f"| {story['id']} | {story['kev_rho']:.2f} | {story['vader_rho']:.2f} "
            f"| {story['kev_shape']}{'' if story['kev_shape'] == story['shape'] else ' ✗'} "
            f"| {story['vader_shape']}{'' if story['vader_shape'] == story['shape'] else ' ✗'} |"
        )
    lines += [
        "",
        "![Synthetic stories: planned fortune, Kev and VADER](synthetic.png)",
        "",
        "## Renamed characters",
        "",
        *([f"*Not yet redone with the current questions: {', '.join(summary['stale_books'])} still hold readings "
           "made with the first wording, so the numbers below and in the smoothing section are from that wording.*", ""]
          if summary["stale_books"] else []),
        f"*Romeo and Juliet* with every character and place renamed (Romeo → Tomas, Juliet → Clara, Verona → Tarsa…), "
        f"read again and compared passage by passage: Pearson r = {summary['renamed']['pearson']:.2f} "
        f"over {summary['renamed']['passages']} passages.",
        "",
        f"Post hoc, not part of the criterion: the smoothed curves correlate r = "
        f"{summary['renamed']['posthoc_smoothed_pearson']:.2f}, and the renamed reading is "
        f"{abs(summary['renamed']['posthoc_mean_shift']):.2f} {'lower' if summary['renamed']['posthoc_mean_shift'] < 0 else 'higher'} "
        "on average (on a -1 to 1 scale). The famous names do not make the reading more tragic.",
        "",
        "## Smoothing",
        "",
        "Width of the Gaussian kernel chosen by leave-one-out cross-validation: each passage predicted from its neighbours.",
        "",
        "| Book | Passages | Best σ (passages) | Share of the book |",
        "|---|---|---|---|",
    ]
    for book_id, row in summary["bandwidth"].items():
        lines.append(f"| {book_id} | {row['passages']} | {row['best_sigma']} | {100 * row['best_fraction']:.1f} % |")
    lines += [
        "",
        "## Caveats",
        "",
        "- The synthetic passages and traps were written to order by an LLM (Claude Sonnet), so they are probably more "
        "explicit than literature.",
        "- Renaming leaves Shakespeare's verse recognisable: C4 tests whether names move the curve, not memory as a whole.",
        "- One run of one model, in bf16 on a Mac (Kev reports probabilities within about 0.05 of its fp32 path).",
        "",
    ]
    return "\n".join(lines)


def write(summary: dict, model: str) -> Path:
    from story_shapes import plot

    plot.draw_traps(load_traps(), cached(RESULTS_DIR / "traps.jsonl"), EVAL_DIR / "traps.png")
    plot.draw_synthetic(summary["synthetic"]["stories"], EVAL_DIR / "synthetic.png")
    path = EVAL_DIR / "RESULTS.md"
    path.write_text(markdown(summary, model))
    return path
