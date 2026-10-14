// Shared drawing helpers: every chart is an SVG string built from the book data.

export const SHAPES = [
  ["rags to riches", "a steady rise"],
  ["riches to rags", "a steady fall"],
  ["from bad to worse", "bad, then worse · scored as riches to rags"],
  ["man in a hole", "a fall, then a way out"],
  ["icarus", "a rise, then a fall"],
  ["cinderella", "a rise, a fall, a higher rise"],
  ["oedipus", "a fall, a rise, a fall"],
  ["ambiguous", "no agreed shape · controls, not scored"],
];

export const label = (shape) => shape.replace(/-/g, " ");

export function esc(text) {
  return String(text).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}

export async function load(path) {
  const response = await fetch(path);
  if (!response.ok) throw new Error(`${path}: ${response.status}`);
  return response.json();
}

export const path = (points) => points.map(([x, y], i) => `${i ? "L" : "M"}${x.toFixed(1)} ${y.toFixed(1)}`).join("");

export function band(xs, line, half, y) {
  const upper = xs.map((x, i) => [x, y(line[i] + half[i])]);
  const lower = xs.map((x, i) => [x, y(line[i] - half[i])]).reverse();
  return `${path(upper)}L${path(lower).slice(1)}Z`;
}

export function dots(points, r) {
  return points.map(([x, y]) => `M${(x - r).toFixed(1)} ${y.toFixed(1)}a${r} ${r} 0 1 0 ${2 * r} 0a${r} ${r} 0 1 0 ${-2 * r} 0`).join("");
}

function stats(values) {
  const mean = values.reduce((s, v) => s + v, 0) / values.length;
  return [mean, Math.sqrt(values.reduce((s, v) => s + (v - mean) ** 2, 0) / values.length)];
}

// The expected shape, moved onto the book's curve: same mean and spread, so only the shape compares.
export function template(values, line, width, y, left = 0) {
  const [tm, ts] = stats(values);
  const [lm, ls] = stats(line);
  const spread = Math.max(ls, 0.05);
  return path(values.map((v, k) => [left + (k / (values.length - 1)) * width, y(lm + ((v - tm) / ts) * spread)]));
}

// A small multiple: the overview curve, its band and the expected shape.
// The vertical range all small multiples share: wide enough for every smoothed curve, so the
// shapes stay comparable, and far narrower than the full Disaster–Triumph scale, so they show.
export function sharedRange(books) {
  const values = books.flatMap((b) => b.line);
  const lo = Math.min(...values), hi = Math.max(...values), margin = (hi - lo) * 0.15;
  return [lo - margin, hi + margin];
}

export function mini(book, templates, [lo, hi], { width = 240, height = 110 } = {}) {
  const y = (v) => height - ((v - lo) / (hi - lo)) * height;
  const n = book.line.length;
  const xs = book.line.map((_, i) => (book.x0 + ((book.x1 - book.x0) * i) / (n - 1)) * width);
  const expected = book.template ? template(templates[book.template], book.line, width, y) : "";
  return `
    <line x1="0" y1="${y(0)}" x2="${width}" y2="${y(0)}" stroke="#161616" stroke-width="1" stroke-dasharray="1 3"/>
    <path d="${band(xs, book.line, book.half, y)}" fill="#161616" fill-opacity="0.12"/>
    <path d="${expected}" fill="none" stroke="var(--expected)" stroke-width="2" stroke-dasharray="5 4"/>
    <path d="${path(xs.map((x, i) => [x, y(book.line[i])]))}" fill="none" stroke="#161616" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>`;
}

export function status(book) {
  if (!book.template) return { text: `control · nearest: ${label(book.nearest)}`, mark: "none" };
  if (book.hit) return { text: `hit · ${label(book.nearest)}`, mark: "hit" };
  if (book.nearest === book.template) return { text: `nearest: ${label(book.nearest)} · band too wide`, mark: "lean" };
  return { text: `read as ${label(book.nearest)}`, mark: "miss" };
}

export function mark(kind) {
  const square = { hit: '<rect x="1" y="1" width="10" height="10" fill="#2756a6" stroke="#2756a6" stroke-width="2"/>',
    lean: '<rect x="1" y="1" width="10" height="10" fill="none" stroke="#2756a6" stroke-width="2"/>',
    miss: '<line x1="1" y1="11" x2="11" y2="1" stroke="#6b6b67" stroke-width="2"/>', none: "" }[kind];
  return `<svg width="12" height="12" aria-hidden="true" style="flex:none">${square}</svg>`;
}
