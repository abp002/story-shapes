import { SHAPES, band, esc, label, load, mark, mini, path, sharedRange, status, template } from "./charts.js";

const pValue = (p) => (p < 0.001 ? "p < 0.001" : `p = ${p.toFixed(3)}`);

function hero(book, templates) {
  const W = 1200, H = 420, left = 56, pad = 20;
  const rows = book.rows.filter((r) => r[2] !== null);
  const lo = Math.min(...rows.map((r) => r[2] - r[3])), hi = Math.max(...rows.map((r) => r[2] + r[3]));
  const y = (v) => H - pad - ((v - lo) / (hi - lo)) * (H - 2 * pad);
  const x = (v) => left + v * (W - left - 20);
  const xs = rows.map((r) => x(r[0]));
  const line = rows.map((r) => r[2]);
  const expected = book.template ? template(templates[book.template], line, W - left - 20, y, left) : "";
  return `
    <line x1="${left}" y1="${y(0)}" x2="${W - 20}" y2="${y(0)}" stroke="#161616" stroke-dasharray="2 4"/>
    <line x1="${left}" y1="10" x2="${left}" y2="${H - 10}" stroke="#161616" stroke-width="2"/>
    <line x1="${left}" y1="${H - 10}" x2="${W - 20}" y2="${H - 10}" stroke="#161616" stroke-width="2"/>
    <text x="22" y="34" font-family="IBM Plex Mono, monospace" font-size="22" font-weight="500">G</text>
    <text x="24" y="${H - 18}" font-family="IBM Plex Mono, monospace" font-size="22" font-weight="500">I</text>
    <text x="${left - 4}" y="${H + 12}" font-family="IBM Plex Mono, monospace" font-size="16">B</text>
    <text x="${W - 32}" y="${H + 12}" font-family="IBM Plex Mono, monospace" font-size="16">E</text>
    <path d="${band(xs, line, rows.map((r) => r[3]), y)}" fill="#161616" fill-opacity="0.1"/>
    <path d="${expected}" fill="none" stroke="#b23a17" stroke-width="3" stroke-dasharray="10 8"/>
    <path d="${path(xs.map((v, i) => [v, y(line[i])]))}" fill="none" stroke="#161616" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>`;
}

function card(book, templates, range) {
  const s = status(book);
  return `<a class="card" href="book.html?id=${encodeURIComponent(book.id)}">
    <h4><span>${esc(book.title)}</span>${mark(s.mark)}</h4>
    <span class="meta">${esc(book.author)} · ${book.passages} passages</span>
    <svg class="mini" viewBox="0 0 240 110" width="100%" role="img" aria-label="${esc(book.title)}: ${esc(s.text)}">${mini(book, templates, range)}</svg>
    <span class="status">${esc(s.text)}</span>
  </a>`;
}

async function main() {
  const index = await load("data/index.json");
  const s = index.summary;
  document.getElementById("hits").innerHTML = `${s.hits}<small> / ${s.scored}</small>`;
  document.getElementById("hits-verdict").textContent = `${s.passes ? "Passes" : "Fails"} · permutation ${pValue(s.p)}`;
  document.getElementById("resolvable").textContent = `Only ${s.resolvable} books cleared that second bar.`;
  document.getElementById("post-hoc").innerHTML = `${s.post_hoc_hits}<small> / ${s.scored}</small>`;
  document.getElementById("post-hoc-p").textContent = `Permutation ${pValue(s.post_hoc_p)}`;

  const range = sharedRange(index.books);
  const groups = document.getElementById("groups");
  groups.innerHTML = SHAPES.map(([shape, description]) => {
    const books = index.books.filter((b) => b.expected === shape);
    if (!books.length) return "";
    return `<div class="group">
      <div class="group-head"><h3>${esc(shape)}</h3><span class="mono muted">${esc(description)}</span>
      <span class="mono count" style="color:var(--faint)">${books.length} ${books.length === 1 ? "book" : "books"}</span></div>
      <div class="cards">${books.map((b) => card(b, index.templates, range)).join("")}</div></div>`;
  }).join("");
  groups.removeAttribute("aria-busy");

  const lear = await load("data/king-lear.json");
  document.getElementById("hero-chart").innerHTML = hero(lear, index.templates);
  document.getElementById("hero-meta").textContent =
    `${lear.author} · ${lear.rows.length} passages · expected: ${lear.expected} · read as: ${label(lear.nearest)}, r = ${lear.r_nearest}`;
  document.getElementById("hero").removeAttribute("aria-busy");
}

main().catch((error) => {
  document.getElementById("groups").textContent = `Could not load the readings (${error.message}).`;
});
