# story-shapes

Draw the shape of a story, Vonnegut-style: how things go for the protagonist from the first
page to the last. Every passage is read by a System One model (Kev, the open replica of
TypeSafe's Jev), which returns probabilities instead of text. Public repo, written in English.

## QA
Nivel: activo

Bitácora: ALE

## How it works
- `books.json` lists the books: Gutenberg id, protagonist, the words the story starts with, an optional
  `end` marker (footnotes, the next play), and the `expected_shape` with `why`. Shapes were committed
  before the book was read (ALE-224); never edit them after seeing a curve.
- `story-shapes read <book>` downloads the text, cuts it into passages and asks Kev three questions
  per passage (`fortune` and `tension` as scores, `present` as a yes/no). Raw answers are cached in
  `data/readings/<book>.jsonl`, one passage per line, and a rerun resumes where it stopped.
- Readings never store passage text (Gutenberg marks some editions, like #5200, as copyrighted).
  The text is rebuilt from `data/raw/` (gitignored) with the same chunking; `<book>.meta.json`
  holds the text hash, chunking, questions and model, and `read` refuses to resume on a mismatch.
  Changing the chunking or the questions means rerunning with `--fresh`.
- `story-shapes plot <book>...` turns the readings into curves: the expected level of each score,
  smoothed over neighbouring passages. The band is a 95 % interval of the smoothed curve
  (`curve.confidence`: kernel-weighted variance of Kev's doubt plus passage disagreement, over the
  effective passage count, corrected for lag-1 autocorrelation). `--relative` puts each curve on
  its own book's scale (`curve.relative`).

## Kev
- A clone of jaredpalmer/kev sits next to this repo (`../kev`), served locally (MLX on Apple Silicon):
  `cd ../kev && uv run --extra serve python -m kev.serve --run jaredpalmer/kev-4b --port 8009`
  About 3 s a passage on an M4 with 24 GB; the first start downloads ~9 GB of weights.
- API: `POST /v1/systemone`. The same client works against Jev: set `STORY_SHAPES_API` and `STORY_SHAPES_KEY`.
- Kev was trained on states of up to 384 tokens; accuracy drops on longer ones. Passages stay
  around 200 words (hard cap 260) for that reason.
- The state carries the protagonist and the passage, never the book title: the title would let the
  model answer from what it already knows about the ending instead of from the passage.

## Commands
- `uv run pytest` — unit tests (chunking, curve maths, story slicing).
- `uv run story-shapes eval corpus` — score every read book against its expected shape (`eval/CORPUS.md`).
- `uv run story-shapes read christmas-carol`
- `uv run story-shapes plot christmas-carol metamorphosis romeo-and-juliet [--relative]`
