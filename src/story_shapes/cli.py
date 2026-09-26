"""story-shapes read <book>... | plot <book>... | eval run|report"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

from story_shapes import chunk, gutenberg, kev, transform

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
READINGS_DIR = ROOT / "data" / "readings"
MODEL_FIELDS = ("name", "run", "base", "lora", "backend", "dtype", "temperature", "release_date")
# Answers are only comparable from the same weights on the same kind of hardware and precision.
MODEL_IDENTITY = ("run", "backend", "dtype")


class ReadingMismatch(Exception):
    """Cached answers were produced with other text, chunking, questions or model."""


def load_books() -> dict[str, dict]:
    return {book["id"]: book for book in json.loads((ROOT / "books.json").read_text())}


def reading_meta(story: str, questions: dict, model_card: dict | None) -> dict:
    """Everything that decides which passages were read and how; stored beside the answers.

    The answers file holds no text, so these settings are what ties each answer back to its
    passage: the same story cut with the same chunking gives the same passages again.
    """
    return {
        "story_sha256": hashlib.sha256(story.encode()).hexdigest(),
        "chunking": {
            "target_words": chunk.TARGET_WORDS,
            "max_words": chunk.MAX_WORDS,
            "min_words": chunk.MIN_WORDS,
        },
        "questions": questions,
        "model": {key: model_card[key] for key in MODEL_FIELDS if key in model_card} if model_card else None,
    }


def _same_reading(a: dict, b: dict) -> bool:
    model_a, model_b = a.get("model") or {}, b.get("model") or {}
    same_model = all(model_a.get(key) == model_b.get(key) for key in MODEL_IDENTITY)
    return same_model and all(a.get(key) == b.get(key) for key in ("story_sha256", "chunking", "questions"))


def resume(path: Path, meta: dict, key: str, fresh: bool = False) -> set:
    """Keys already answered in the cache at `path`, after checking it was made with `meta`.

    The meta is written to `<path stem>.meta.json`. A cache made with other settings raises
    ReadingMismatch instead of mixing answers; `fresh` discards it first.
    """
    meta_path = path.with_name(path.stem + ".meta.json")
    if fresh:
        path.unlink(missing_ok=True)
        meta_path.unlink(missing_ok=True)
    done = set()
    if path.exists():
        stored = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        if not _same_reading(stored, meta):
            raise ReadingMismatch(f"{path} was made with other settings; rerun with --fresh to start over")
        done = {json.loads(line)[key] for line in path.read_text().splitlines() if line.strip()}
    path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    return done


def read(book: dict, client, raw_dir: Path = RAW_DIR, readings_dir: Path = READINGS_DIR, fresh: bool = False) -> Path:
    """Ask the model about every passage of `book`; a rerun resumes after the last cached passage.

    Only positions and answers are stored, never the passage text: some editions are still under
    copyright, and the text can always be rebuilt from Gutenberg with the same chunking.
    """
    raw = gutenberg.fetch(book["gutenberg"], raw_dir)
    if gutenberg.is_copyrighted(raw):
        print(f"{book['id']}: Gutenberg marks this edition as copyrighted; do not publish its text", flush=True)
    story = gutenberg.story_text(raw, book["start"], book.get("end"))
    if "rename" in book:
        story = transform.rename(story, book["rename"])
    passages = chunk.passages(story)
    questions = kev.questions(book["protagonist"])
    meta = reading_meta(story, questions, client.model_card())

    path = readings_dir / f"{book['id']}.jsonl"
    done = resume(path, meta, "index", fresh)

    started = time.perf_counter()
    with path.open("a") as out:
        for passage in passages:
            if passage.index in done:
                continue
            response = client.decide(kev.state(book["protagonist"], passage.text), questions)
            record = {
                "index": passage.index,
                "start": passage.start,
                "end": passage.end,
                "words": passage.words,
                "answers": response["answers"],
                "latency_ms": response.get("latency_ms"),
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            fortune = response["answers"]["fortune"]["score"]
            print(f"{book['id']} {passage.index + 1}/{len(passages)} fortune={fortune:.2f}", flush=True)
    print(f"{book['id']}: {len(passages)} passages, {time.perf_counter() - started:.0f} s", flush=True)
    return path


def load_env(path: Path) -> None:
    """KEY=value lines from a local, untracked .env file; the real environment always wins."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        key, sep, value = line.partition("=")
        if sep and key.strip() and not key.lstrip().startswith("#"):
            os.environ.setdefault(key.strip(), value.strip())


def main(argv: list[str] | None = None) -> int:
    load_env(ROOT / ".env")
    parser = argparse.ArgumentParser(prog="story-shapes")
    commands = parser.add_subparsers(dest="command", required=True)
    read_cmd = commands.add_parser("read", help="read books with the model and cache the answers")
    read_cmd.add_argument("books", nargs="+", help="book ids from books.json, or 'all'")
    read_cmd.add_argument("--fresh", action="store_true", help="discard cached answers and start over")
    plot_cmd = commands.add_parser("plot", help="draw the curves of books already read")
    plot_cmd.add_argument("books", nargs="+")
    plot_cmd.add_argument("--out", type=Path, default=ROOT / "data" / "plots" / "shapes.png")
    plot_cmd.add_argument("--relative", action="store_true", help="draw each curve on its own book's scale")
    eval_cmd = commands.add_parser("eval", help="does the model read fortune or only tone? (ALE-198)")
    eval_cmd.add_argument("action", choices=["run", "report", "tune", "deaths", "corpus"])
    eval_cmd.add_argument("--fresh", action="store_true", help="discard cached answers and start over")
    args = parser.parse_args(argv)

    if args.command == "eval" and args.action == "corpus":
        from story_shapes import corpus

        summary = corpus.score()
        print(corpus.write(summary))
        print(f"{summary['hits']}/{summary['scored']} books, p = {summary['p']:.3f}")
        return 0

    if args.command == "eval":
        from story_shapes import evaluate

        client = kev.SystemOne()
        try:
            if args.action == "run":
                evaluate.run(client, fresh=args.fresh)
            elif args.action == "deaths":
                for rule in evaluate.tune_deaths(client, fresh=args.fresh)[:8]:
                    print(rule)
            elif args.action == "tune":
                for name, row in evaluate.tune(client, fresh=args.fresh).items():
                    print(name, "  ".join(f"{kind} {100 * value:.0f}%" for kind, value in row.items()))
            else:
                card = client.model_card() or {}
                print(evaluate.write(evaluate.report(), card.get("description", "unknown")))
        except ReadingMismatch as error:
            parser.exit(1, f"story-shapes: {error}\n")
        return 0

    books = load_books()
    ids = list(books) if args.books == ["all"] else args.books
    unknown = [book_id for book_id in ids if book_id not in books]
    if unknown:
        parser.error(f"unknown books: {', '.join(unknown)}")

    if args.command == "read":
        client = kev.SystemOne()
        for book_id in ids:
            try:
                read(books[book_id], client, fresh=args.fresh)
            except ReadingMismatch as error:
                parser.exit(1, f"story-shapes: {error}\n")
    else:
        from story_shapes import plot

        plot.draw([books[book_id] for book_id in ids], READINGS_DIR, args.out, relative=args.relative)
        print(args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
