# Corpus: does Kev recover the expected shapes?

Expected shapes and scoring were committed before the corpus was read (see `src/story_shapes/corpus.py`).

**C5: Kev picks the expected shape for more books than shuffled labels would (permutation p < 0.05)** — fails: 2 of 27 books, p = 0.065. 3 have a resolvable shape; the expected shape is consistent with the curve's band for 25.

| book | expected | nearest | r expected | r nearest | resolvable | consistent | hit |
|---|---|---|---|---|---|---|---|
| Candide | ambiguous | rags to riches | — | 0.60 | no | — | — |
| Hamlet | ambiguous | cinderella | — | 0.39 | no | — | — |
| Great Expectations | cinderella | cinderella | 0.86 | 0.86 | no | yes | no |
| Jane Eyre | cinderella | cinderella | 0.71 | 0.71 | no | yes | no |
| Ethan Frome | from bad to worse | riches to rags | 0.56 | 0.56 | no | yes | no |
| The Metamorphosis | from bad to worse | icarus | 0.36 | 0.55 | no | yes | no |
| Macbeth | icarus | riches to rags | 0.22 | 0.94 | no | yes | no |
| Romeo and Juliet | icarus | icarus | 0.69 | 0.69 | no | yes | no |
| The Awakening | icarus | oedipus | 0.12 | 0.53 | no | yes | no |
| The Great Gatsby | icarus | riches to rags | 0.71 | 0.75 | no | yes | no |
| The Picture of Dorian Gray | icarus | riches to rags | 0.08 | 0.86 | no | yes | no |
| A Christmas Carol | man in a hole | rags to riches | 0.14 | 0.68 | no | yes | no |
| A Little Princess | man in a hole | man in a hole | 0.87 | 0.87 | no | yes | no |
| Black Beauty | man in a hole | man in a hole | 0.63 | 0.63 | no | yes | no |
| Pride and Prejudice | man in a hole | man in a hole | 0.62 | 0.62 | no | yes | no |
| Robinson Crusoe | man in a hole | rags to riches | 0.29 | 0.85 | no | yes | no |
| Silas Marner | man in a hole | rags to riches | 0.68 | 0.86 | yes | yes | no |
| The Call of the Wild | man in a hole | rags to riches | 0.44 | 0.76 | no | yes | no |
| The Wonderful Wizard of Oz | man in a hole | cinderella | 0.18 | 0.34 | no | yes | no |
| Treasure Island | man in a hole | cinderella | -0.02 | 0.22 | no | no | no |
| Tess of the d'Urbervilles | oedipus | icarus | -0.21 | 0.80 | no | no | no |
| Anne of Green Gables | rags to riches | man in a hole | 0.38 | 0.69 | no | yes | no |
| Oliver Twist | rags to riches | rags to riches | 0.51 | 0.51 | no | yes | no |
| The Secret Garden | rags to riches | rags to riches | 0.75 | 0.75 | yes | yes | yes |
| Daisy Miller | riches to rags | riches to rags | 0.86 | 0.86 | no | yes | no |
| Frankenstein | riches to rags | riches to rags | 0.74 | 0.74 | no | yes | no |
| King Lear | riches to rags | riches to rags | 0.92 | 0.92 | no | yes | no |
| Madame Bovary | riches to rags | riches to rags | 0.72 | 0.72 | yes | yes | yes |
| Oedipus the King | riches to rags | riches to rags | 0.75 | 0.75 | no | yes | no |
