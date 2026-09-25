# story-shapes

The shape of a story, read passage by passage by a model that decides instead of writing.

Kurt Vonnegut argued that stories have shapes: plot how things go for the protagonist from the
first page to the last, and a handful of curves keep coming back — *man in a hole*, *from bad to
worse*, *Cinderella*. Researchers have since drawn these curves by scoring the tone of the words
([syuzhet](https://github.com/mjockers/syuzhet);
[Reagan et al., 2016](https://arxiv.org/abs/1606.07772)). But tone is not fortune: a gloomy
paragraph in which the hero wins still reads as a fall.

story-shapes asks the question directly. Every passage of about 200 words goes to
[Kev](https://github.com/jaredpalmer/kev), an open replica of TypeSafe's Jev: a *System One*
model that returns calibrated probabilities instead of text. It answers three questions per
passage: how are things going for the protagonist (five levels), how tense is it (four levels),
and does the protagonist appear.

![Fortune and tension curves for A Christmas Carol, The Metamorphosis and Romeo and Juliet](docs/first-look.png)

*First look: three short books read by Kev-4B on a laptop. Dots are passages, the line is a
Gaussian-smoothed average.*

## Status

Early. The first run is encouraging, but three books prove little on their own.

- *Romeo and Juliet* and *A Christmas Carol* come out with the shapes you would expect.
- *The Metamorphosis* is nearly flat, and its last passages rise: once Gregor is dead, the model
  reads the family's relief as his fortune.
- Not shown yet: that the model reads fortune better than word lists do, that it reads famous
  books rather than remembering them, and that the smoothing is not what makes the shapes. Next
  come a lexicon baseline, passages where tone and fortune disagree, synthetic stories with
  known arcs, and renamed characters.
- The band shows how torn the model is on each passage, widened by Kev's calibration
  temperature. It will become the uncertainty of the curve.

## How it works

- `books.json` lists each book: its Gutenberg id, its protagonist, and the words the story
  starts with, so the title page and table of contents are skipped.
- Passages stay around 200 words (never more than 260), because Kev was trained on inputs of up
  to 384 tokens.
- The model sees the protagonist's name and the passage, never the title: it should judge the
  passage, not recall how the book ends.
- `data/readings/<book>.jsonl` keeps each passage's position and the model's raw answers.
  `<book>.meta.json` records what produced them (text hash, chunking, questions, model), and a
  run with different settings refuses to resume.

## Run it

You need Python 3.12+, [uv](https://docs.astral.sh/uv/) and a Kev server (Jev works too).

```bash
git clone https://github.com/jaredpalmer/kev.git ../kev
(cd ../kev && uv sync --extra serve && uv run --extra serve python -m kev.serve --run jaredpalmer/kev-4b --port 8009)

# in another terminal
uv run story-shapes read romeo-and-juliet
uv run story-shapes plot romeo-and-juliet
```

`STORY_SHAPES_API` and `STORY_SHAPES_KEY` point the client at another System One endpoint. On an
Apple M4 with 24 GB, Kev-4B takes about 3 seconds a passage: a short novel in 10–15 minutes.

## Texts

Books come from [Project Gutenberg](https://www.gutenberg.org). This repository stores no book
text: passages are rebuilt from Gutenberg with the same chunking. Gutenberg marks some editions,
such as David Wyllie's translation of *The Metamorphosis*, as still under copyright; `read`
warns about them, and nothing from them beyond positions and model answers is published here.

## Credits

Kurt Vonnegut's *Shape of Stories*; Matthew Jockers' syuzhet; Reagan, Mitchell, Kiley, Danforth
and Dodds, *The emotional arcs of stories are dominated by six basic shapes*; Kev by Jared
Palmer; TypeSafe's System One API; Project Gutenberg.

## License

MIT, for the code.
