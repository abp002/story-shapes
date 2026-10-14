import { band, dots, esc, label, load, path, template } from "./charts.js";

const LEVELS = ["Disaster", "Setback", "No change", "Gain", "Triumph"];
const COLORS = ["#8e2a12", "#c9653f", "#9a9a94", "#5d86c4", "#2a5caa"];
const FY = (v) => 200 - v * 190;
const TY = (v) => 155 - v * 150;
const X = (x) => x * 1000;
const $ = (id) => document.getElementById(id);

function draw(book, templates) {
  const rows = book.rows;
  const on = rows.filter((r) => r[2] !== null);
  const line = on.map((r) => r[2]);
  const xs = on.map((r) => X(r[0]));
  const chapterLines = (h) => book.chapters.slice(1).map((c) => `M${X(c.x).toFixed(1)} 0V${h}`).join("");
  // The expected shape spans the same stretch as the curve, which stops at the hero's death.
  const expected = book.template ? template(templates[book.template], line, xs[xs.length - 1], FY) : "";
  const death = book.died_at === null ? "" :
    `<line x1="${X(book.died_at)}" y1="0" x2="${X(book.died_at)}" y2="400" stroke="#161616" stroke-dasharray="6 4"/>
     <text x="${X(book.died_at) + 8}" y="16" font-family="IBM Plex Mono, monospace" font-size="13">† ${esc(book.protagonist)} dies</text>`;
  $("fortune").innerHTML = `
    <path d="${[-1, -0.5, 0.5, 1].map((v) => `M0 ${FY(v)}H1000`).join("")}" stroke="#dcddd8" fill="none"/>
    <path d="${chapterLines(400)}" stroke="#c9cac4" stroke-dasharray="2 4" fill="none"/>
    <line x1="0" y1="200" x2="1000" y2="200" stroke="#161616"/>
    <path d="${band(xs, line, on.map((r) => r[3]), FY)}" fill="#161616" fill-opacity="0.1"/>
    <path d="${dots(on.filter((r) => r[7] < 0.5).map((r) => [X(r[0]), FY(r[1])]), 3.2)}" fill="#2a5caa" fill-opacity="0.16"/>
    <path d="${dots(on.filter((r) => r[7] >= 0.5).map((r) => [X(r[0]), FY(r[1])]), 3.2)}" fill="#2a5caa" fill-opacity="0.5"/>
    <path d="${expected}" fill="none" stroke="#b23a17" stroke-width="2.5" stroke-dasharray="9 7" stroke-opacity="0.85"/>
    <path d="${path(xs.map((x, i) => [x, FY(line[i])]))}" fill="none" stroke="#161616" stroke-width="3.5" stroke-linejoin="round" stroke-linecap="round"/>
    ${death}
    <g id="cursor"></g>`;
  $("tension").innerHTML = `
    <path d="${chapterLines(160)}" stroke="#c9cac4" stroke-dasharray="2 4" fill="none"/>
    <path d="${band(rows.map((r) => X(r[0])), rows.map((r) => r[5]), rows.map((r) => r[6]), TY)}" fill="#d2601a" fill-opacity="0.13"/>
    <path d="${dots(rows.map((r) => [X(r[0]), TY(r[4])]), 2.4)}" fill="#d2601a" fill-opacity="0.35"/>
    <path d="${path(rows.map((r) => [X(r[0]), TY(r[5])]))}" fill="none" stroke="#b4500f" stroke-width="2.5" stroke-linejoin="round"/>
    <g id="tcursor"></g>`;
  $("levels").innerHTML = LEVELS.map((name, i) => `<span style="top:${(FY(-1 + i / 2) / 4).toFixed(1)}%">${name}</span>`).join("");
  // Label every chapter when they fit, else every few; never two labels on top of each other.
  const step = Math.ceil(book.chapters.length / 24);
  const shown = [];
  book.chapters.forEach((c, i) => {
    if (i % step === 0 && (!shown.length || c.x - shown[shown.length - 1].x >= 0.025)) shown.push(c);
  });
  $("chapters").innerHTML = shown
    .map((c) => `<span title="${esc(c.name)}" style="left:${(c.x * 100 + 0.8).toFixed(1)}%">${c.n}</span>`).join("");
  $("markers").innerHTML = book.moments.map((m, k) =>
    `<span class="marker" style="left:${(rows[m.i][0] * 100).toFixed(2)}%;top:${(FY(rows[m.i][2]) / 4).toFixed(2)}%">${k + 1}</span>`).join("");
}

function select(book, i) {
  const r = book.rows[i];
  const on = r[2] !== null;
  const cx = X(r[0]).toFixed(1);
  $("cursor").innerHTML = `<line x1="${cx}" y1="0" x2="${cx}" y2="400" stroke="#161616" stroke-width="1.5"/>` + (on ?
    `<circle cx="${cx}" cy="${FY(r[2])}" r="7" fill="#f3f4f1" stroke="#161616" stroke-width="3"/>
     <circle cx="${cx}" cy="${FY(r[1])}" r="5" fill="#2a5caa" stroke="#f3f4f1" stroke-width="1.5"/>` : "");
  $("tcursor").innerHTML = `<line x1="${cx}" y1="0" x2="${cx}" y2="160" stroke="#161616" stroke-width="1.5"/>`;
  const chapter = book.chapters.filter((c) => c.x <= r[0] + 1e-6).pop();
  const where = chapter ? ` · chapter ${chapter.n}${chapter.name && !/^[\dIVXLC.]+$/.test(chapter.name) ? `, ${chapter.name}` : ""}` : "";
  $("scrub").value = i;
  $("scrub-label").textContent = `Passage ${i + 1} of ${book.rows.length}${where}`;
  $("where").textContent = `What Kev read · ${Math.round(r[0] * 100)} % into the book`;
  $("quote").textContent = book.text ? `${book.text[i]} …` : "This edition is still under copyright, so its text is not shown.";
  $("present").textContent = `${book.protagonist} is in this passage: ${Math.round(r[7] * 100)} % sure` +
    (on ? "" : ` · after ${book.protagonist}'s death, so not on the curve`);
  $("bars").innerHTML = LEVELS.map((name, k) => `<div class="bar"><span>${name}</span>
    <div class="track"><div class="fill" style="width:${(r[8][k] * 100).toFixed(1)}%;background:${COLORS[k]}"></div></div>
    <span class="pct">${Math.round(r[8][k] * 100)}%</span></div>`).join("");
  for (const button of $("moments").children) button.setAttribute("aria-pressed", String(Number(button.dataset.i) === i));
}

async function main() {
  const id = new URLSearchParams(location.search).get("id") || "a-little-princess";
  const [index, book] = await Promise.all([load("data/index.json"), load(`data/${encodeURIComponent(id)}.json`)]);
  document.title = `${book.title} · story-shapes`;
  $("title").textContent = book.title;
  $("byline").textContent = `${book.author} · ${book.chapters.length > 1 ? `${book.chapters.length} chapters · ` : ""}${book.rows.length} passages`;
  $("why").textContent = book.why || "";
  $("fortune-title").textContent = `${book.protagonist}'s fortune`;
  $("question").textContent = `How Kev answered: what happens to ${book.protagonist} here?`;
  const verdict = !book.template ? "a control: no agreed shape" : book.hit ? "a hit: the band rules out flat"
    : book.nearest === book.template ? "nearest shape, but the band is too wide" : "another shape";
  $("facts").innerHTML = `<dt>Expected</dt><dd style="color:var(--expected)">${esc(book.expected)}</dd>
    <dt>Kev read</dt><dd>${esc(label(book.nearest))} · r = ${book.r_nearest}</dd><dt>Verdict</dt><dd>${esc(verdict)}</dd>`;
  $("fortune").setAttribute("aria-label", `${book.protagonist}'s fortune through ${book.title}, read as ${label(book.nearest)}`);
  $("source").textContent = `Kev-4B read ${book.rows.length} passages of this Project Gutenberg text. ` +
    (book.died_at === null ? `Kev never reads ${book.protagonist} as dead for good, so the curve runs to the last page.`
      : `The curve stops where ${book.protagonist} dies for good, ${Math.round(book.died_at * 100)} % into the book.`);
  draw(book, index.templates);
  $("moments").innerHTML = book.moments.map((m, k) =>
    `<button type="button" data-i="${m.i}" aria-pressed="false"><span class="n">${k + 1}</span><span>${esc(m.label)}</span></button>`).join("");
  $("moments").addEventListener("click", (e) => {
    const button = e.target.closest("button");
    if (button) select(book, Number(button.dataset.i));
  });
  $("scrub").max = book.rows.length - 1;
  $("scrub").addEventListener("input", (e) => select(book, Number(e.target.value)));
  const lowest = book.moments.find((m) => m.label === "Lowest point");
  select(book, lowest ? lowest.i : 0);
  $("book").removeAttribute("aria-busy");
}

main().catch((error) => {
  $("title").textContent = "Book not found";
  $("why").textContent = `Could not load this book (${error.message}).`;
});
