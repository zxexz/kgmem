"use strict";
// Model and graph strings enter the DOM through textContent.

const $ = (s) => document.querySelector(s);
const NS = "http://www.w3.org/2000/svg";
const S = {
  state: null, run: null, sel: 0, mode: "ask", busy: false, layout: null, layoutKey: "", layoutNames: null,
  history: {}, edit: null, tour: null, graphFocus: null, graphNeighbourhood: false, tl: null,
};

// helpers
function el(tag, attrs = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "class") e.className = v;
    else if (k === "text") e.textContent = v;
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else e.setAttribute(k, v === true ? "" : v);
  }
  for (const k of kids) if (k !== null && k !== undefined && k !== false) e.append(k);
  return e;
}
function sv(tag, attrs = {}) {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "text") e.textContent = v;
    else if (v !== null && v !== undefined) e.setAttribute(k, v);
  }
  return e;
}
const cssv = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const fmt = (v, d = 3) => (v === null || v === undefined ? "–" : (+v).toFixed(d));
const showTok = (s) => s.replace(/\n/g, "⏎").replace(/ /g, "·") || "∅";
const plainTok = (s) => s.replace(/\n/g, "⏎");
const q = (s) => `“${plainTok(s)}”`;
const ent = (i) => (S.run && S.run.entities[i] !== undefined ? S.run.entities[i] : `#${i}`);
const rel = (i) => (S.run && S.run.relations[i] !== undefined ? S.run.relations[i] : `r${i}`);
const argmax = (a) => a.reduce((bi, v, i) => (v > a[bi] ? i : bi), 0);
const C = () => ({
  s1: cssv("--s1"), s1d: cssv("--s1-deep"), s2: cssv("--s2"), s3: cssv("--s3"), ink: cssv("--ink"), ink2: cssv("--ink-2"),
  muted: cssv("--muted"), edge: cssv("--edge"), grid: cssv("--grid"), surface: cssv("--surface"), wash: cssv("--s1-wash"),
  st: [cssv("--st1"), cssv("--st2"), cssv("--st3"), cssv("--st4")], neg: cssv("--div-neg"), mid: cssv("--div-mid"),
});
function isDark() {
  const t = document.documentElement.dataset.theme;
  return t ? t === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
}
function hex(h) { const n = parseInt(h.slice(1), 16); return [n >> 16, (n >> 8) & 255, n & 255]; }
function mix(a, b, f) { const x = hex(a), y = hex(b); return `rgb(${x.map((c, j) => Math.round(c + (y[j] - c) * f)).join(",")})`; }
// Separate light/dark ramps keep zero close to the surface color.
const RAMP = {
  light: ["#f1f5fb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
  dark: ["#1f252d", "#104281", "#1c5cab", "#2a78d6", "#5598e7", "#86b6ef", "#b7d3f6"],
};
function seq(v) {
  const r = RAMP[isDark() ? "dark" : "light"];
  const x = Math.max(0, Math.min(1, +v || 0)) * (r.length - 1);
  const i = Math.min(r.length - 2, Math.floor(x));
  return mix(r[i], r[i + 1], x - i);
}
// Red: negative probability change. Blue: positive.
function div(v) { const c = C(), t = Math.max(-1, Math.min(1, +v || 0)); return t >= 0 ? mix(c.mid, c.s1, t) : mix(c.mid, c.neg, -t); }
const rampCss = () => `linear-gradient(90deg, ${RAMP[isDark() ? "dark" : "light"].join(",")})`;
const divCss = () => { const c = C(); return `linear-gradient(90deg, ${c.neg}, ${c.mid}, ${c.s1})`; };

// Tooltips are aria-hidden; tables provide the numeric values.
const tip = $("#tooltip");
function showTip(ev, rows) {
  tip.replaceChildren(...rows.map((r) => {
    const d = el("div", { class: "row" });
    if (r.color) d.append(el("span", { class: "line", style: `background:${r.color}` }));
    if (r.v !== undefined) d.append(el("span", { class: "tv", text: r.v }));
    if (r.k) d.append(el("span", { class: "tk", text: r.k }));
    return d;
  }));
  tip.style.display = "block";
  const pad = 14, rc = tip.getBoundingClientRect();
  let x = ev.clientX + pad, y = ev.clientY + pad;
  if (x + rc.width > innerWidth - 8) x = ev.clientX - rc.width - pad;
  if (y + rc.height > innerHeight - 8) y = ev.clientY - rc.height - pad;
  tip.style.left = `${x}px`; tip.style.top = `${y}px`;
}
const hideTip = () => { tip.style.display = "none"; };
function hover(node, rowsFn) {
  node.addEventListener("pointermove", (e) => showTip(e, rowsFn()));
  node.addEventListener("pointerleave", hideTip);
}

async function api(path, body) {
  const opts = body === undefined ? {} : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
  const r = await fetch(path, opts);
  const j = await r.json();
  if (!r.ok) throw new Error((j.error || r.statusText).replace(/^\w+Error: /, ""));
  return j;
}

function status(where, msg, err = false) {
  const s = document.getElementById(where);
  if (!s) return;
  s.textContent = msg; s.classList.toggle("err", err);
}

// glossary
const GLOSS = {
  mention: ["mention attention", "Attention over earlier tokens. The current position inherits entity candidates from the tokens it attends to."],
  link: ["entity link", "Similarity between a token's hidden state and entity-name keys. Weights sum to one over the displayed top keys; cosine reports similarity before that normalization."],
  hop: ["hop", "One graph lookup: (head, relation) → tails. The next hop starts from those tails."],
  halt: ["halting", "Weights used to mix the candidate answers from each hop count."],
  lam: ["copy gate λ", "Fraction of the next-token distribution supplied by the retrieved-name pointer."],
  mass: ["pointer mass", "Retrieved-name weight available to the copy head at this position."],
  gkg: ["graph gate", "Scales the graph value added to the hidden state."],
  dkg: ["|Δ| / |h|", "Norm of the memory's hidden-state update divided by the original hidden-state norm."],
  gpad: ["scratchpad gate", "Scales the scratchpad read added to the hidden state."],
  beta: ["write strength β", "Scales the delta-rule update. Zero skips the write; with a unit key, one replaces the value read along that key."],
  addr: ["read / write address", "Slot weights from a shared map applied to the token selected by read or write attention."],
  readout: ["readout bias", "Logits added by projecting a scratchpad read through the frozen output embedding."],
  gro: ["readout gate", "Scales the scratchpad logit bias. Training targets scratchpad answers and their following line ends."],
  overlap: ["address overlap", "Sum of read/write address products, scaled by β. This compares addresses; intervening writes can change the stored value."],
  gain: ["memory effect", "Change in probability of the following token: memory on minus memory off. Blue is positive; red is negative."],
  stages: ["prediction stages", "Base LM: memory off. After injection: upper layers with the memory's hidden-state update. + readout: scratchpad logit bias. Final: copy-head mixture."],
  persistent: ["persistent scratchpad", "State saved with the adapter. Remember updates it; Ask and Trace can read it without committing their changes."],
  slotnorm: ["slot norm", "Vector norm of a scratchpad row. Zero is empty; the scale depends on the adapter."],
  addrc: ["read / write key", "Continuous addresses are unit vectors from a shared map. Reads recover the state projected along their key; key overlap measures potential interference."],
  overlapc: ["key overlap", "cosine(read key, write key) × β. This compares addresses; intervening writes can change the stored value."],
  varnorm: ["read norm", "Norm of the value read along a variable's key."],
};
// continuous keys (cfg.cont_keys): rows of the scratchpad are key dimensions, so views list variables instead of slots
const slotName = (pd, s) => (pd.labels ? `“${pd.labels[s]}”` : `slot ${s}`);
const via = (pd, t, list = pd.ptr_w) => (list[t] && list[t][0] ? q(S.run.trace.tokens[list[t][0][0]]) : "this token");
function renderGlossary() {
  const dl = $("#glossary");
  dl.replaceChildren(...Object.entries(GLOSS).flatMap(([k, [t, d]]) => [el("dt", { text: t }), el("dd", { id: `gl-${k}`, text: d })]));
}
function info(key) {
  const [term, def] = GLOSS[key];
  const b = el("button", { class: "info", type: "button", "aria-label": `What is ${term}?`, "aria-describedby": `gl-${key}`, text: "ⓘ" });
  b.addEventListener("pointermove", (e) => showTip(e, [{ v: term }, { k: def }]));
  b.addEventListener("pointerleave", hideTip);
  b.addEventListener("focus", () => { const r = b.getBoundingClientRect(); showTip({ clientX: r.right, clientY: r.bottom }, [{ v: term }, { k: def }]); });
  b.addEventListener("blur", hideTip);
  return b;
}

// small charts
function barList(items, { color, max, digits = 3, empty = "nothing above threshold", faded = false } = {}) {
  if (!items.length) return el("div", { class: "empty", text: empty });
  const box = el("div", { class: "bars" + (faded ? " faded" : "") });
  const m = max ?? Math.max(...items.map((i) => Math.abs(i.v)), 1e-9);
  for (const it of items) {
    const w = Math.max(0, it.v) / m, col = it.color || color;
    const track = el("div", { class: "track row-hit" },
      el("div", { class: "bar", style: `width:${(w * 78).toFixed(2)}%;background:${col}` }),
      el("span", { class: "val", style: `left:calc(${(w * 78).toFixed(2)}% + 6px)`, text: it.label ?? fmt(it.v, digits) }));
    hover(track, () => [{ v: fmt(it.v, 4), k: it.tip || it.name, color: col }]);
    if (it.onclick) { track.style.cursor = "pointer"; track.addEventListener("click", it.onclick); }
    box.append(el("div", { class: "name" + (it.dim ? " dim" : ""), text: it.name, title: it.name }), track);
  }
  return box;
}

function slotColumns(values, color, { height = 56, max = null, unit = "", label = "slot weights" } = {}) {
  const n = values.length, cw = 4, gap = 1, padL = 2, padT = 16, W = padL + n * (cw + gap) + 8, H = height + padT + 16;
  const m = max ?? Math.max(...values, 1e-9);
  const peak = argmax(values);
  const svg = sv("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: "img",
    "aria-label": `${label}: largest is slot ${peak} at ${fmt(values[peak], 2)}${unit}` });
  const base = padT + height, c = C();
  svg.append(sv("line", { x1: 0, x2: W - 8, y1: base + 0.5, y2: base + 0.5, stroke: c.edge }));
  values.forEach((v, i) => {
    const x = padL + i * (cw + gap), h = Math.max(0, (v / m) * height);
    if (h > 0.4) {
      const y = base - h, r = Math.min(2, h);
      svg.append(sv("path", { d: `M${x},${base}V${y + r}Q${x},${y} ${x + r},${y}H${x + cw - r}Q${x + cw},${y} ${x + cw},${y + r}V${base}Z`, fill: color }));
    }
    const hit = sv("rect", { x: x - 1, y: padT - 4, width: cw + gap, height: height + 8, fill: "transparent" });
    hover(hit, () => [{ v: fmt(v, 4) + unit, k: `slot ${i}`, color }]);
    svg.append(hit);
  });
  for (const t of [0, 16, 32, 48, n - 1]) if (t < n) svg.append(sv("text", { x: padL + t * (cw + gap), y: base + 12, "font-size": 10, fill: c.muted, text: t }));
  if (values[peak] > 0) {
    const x = padL + peak * (cw + gap) + cw / 2;
    svg.append(sv("text", { x, y: padT - 4, "text-anchor": x > W / 2 ? "end" : "start", "font-size": 10, fill: c.ink2, text: `slot ${peak} · ${fmt(values[peak], 2)}${unit}` }));
  }
  return svg;
}

function legend(items) {
  return el("div", { class: "legend" }, ...items.map((i) => el("span", { class: "key" },
    el("span", { class: i.kind || "rect", style: i.kind === "ramp" ? `background:${rampCss()}` : i.kind === "div" ? `background:${divCss()}` : `background:${i.color}` }),
    document.createTextNode(i.label), i.info ? info(i.info) : null)));
}
function table(head, rowsData) {
  const t = el("table", { class: "t" });
  t.append(el("thead", {}, el("tr", {}, ...head.map((h) => el("th", { text: h, scope: "col" })))));
  const tb = el("tbody");
  for (const r of rowsData) tb.append(el("tr", {}, ...r.map((c, i) => el("td", { class: i === 0 ? "mono" : "", text: c }))));
  t.append(tb);
  return t;
}
function statTile(label, value, key) {
  return el("span", { class: "stat" }, el("span", { class: "label" }, document.createTextNode(label), key ? info(key) : null), el("span", { class: "value", text: value }));
}
function cardHead(node, step, title, sub, key) {
  const h = el("h2", {}, step ? el("span", { class: "step", text: String(step) }) : null, document.createTextNode(title), key ? info(key) : null);
  node.replaceChildren(h);
  if (sub) node.append(el("p", { class: "sub", text: sub }));
}
function spot(id) {
  const n = document.getElementById(id);
  if (!n) return;
  n.scrollIntoView({ behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "start" });
  n.classList.add("spot"); setTimeout(() => n.classList.remove("spot"), 1600);
}

// examples
const PRESETS = [
  { label: "Capital of Velmora (1 hop)", mode: "ask", text: "Q: What is the capital of Velmora?\nA:", needs: "Velmora" },
  { label: "Where the leader of Velmora lives (2 hops)", mode: "ask", text: "Q: Where does the leader of Velmora live?\nA:", needs: "Velmora" },
  { label: "Where Talia's friend lives (2 hops)", mode: "ask", text: "Q: Where does Talia's friend live?\nA:", needs: "Talia" },
  { label: "Currency of Quistan (a multi-token name)", mode: "ask", text: "Q: Which currency is used in Quistan?\nA:", needs: "Quistan" },
  { label: "Scratchpad 1 · Remember: set kel, then vom, then kel again", mode: "remember", text: "Set kel to fox. Set vom to tea. Set kel to rice.\n" },
  { label: "Scratchpad 2 · Ask in a new session: what is kel?", mode: "ask", persist: true, text: "Q: What is kel?\nA:" },
  { label: "Scratchpad · write and read in one sequence (Trace)", mode: "trace", text: "Set kel to fox. Set vom to tea. Set kel to rice.\nQ: What is kel?\nA: rice\n" },
  { label: "Change Velmora's capital and compare answers", special: "teach" },
  { label: "strandedcpu · square of 7", mode: "ask", text: "Q: What is the square of 7?\nA:", graph: "strandedcpu", rel: "square" },
  { label: "strandedcpu · smallest prime factor of 91", mode: "ask", text: "Q: What is the smallest prime factor of 91?\nA:", graph: "strandedcpu", rel: "smallest prime factor" },
  { label: "strandedcpu · digit sum of square of 12 (2 hops)", mode: "ask", text: "Q: What is the digit sum of the square of 12?\nA:", graph: "strandedcpu", rel: "digit sum" },
];
const TPL1 = {
  capital: "What is the capital of {h}?", currency: "Which currency is used in {h}?", "located in": "In which country is {h}?",
  "lives in": "Where does {h} live?", leader: "Who leads {h}?", friend: "Who is {h}'s friend?",
};
const TPL2 = {
  "lives in|located in": "In which country does {h} live?", "leader|lives in": "Where does the leader of {h} live?",
  "friend|lives in": "Where does {h}'s friend live?", "located in|capital": "Name the capital of the country containing {h}.",
  "located in|currency": "What currency is used in the city of {h}?",
};
function graphExamples(g) {
  const out = [], E = g.entities, R = g.relations, byHR = new Map();
  for (const [h, r, t] of g.triples) { const k = `${h},${r}`; if (!byHR.has(k)) byHR.set(k, []); byHR.get(k).push(t); }
  const seen = new Set();
  for (const [h, r, t] of g.triples) {
    if (seen.has(r) || out.length >= 4) continue;
    seen.add(r);
    const tpl = TPL1[R[r]] || `What is the ${R[r]} of {h}?`;
    const qn = tpl.replace("{h}", E[h]);
    out.push({ label: `${qn} → ${E[t]}`, mode: "ask", text: `Q: ${qn}\nA:` });
  }
  // Use relation names when no domain-specific two-hop template applies.
  if (!Object.keys(TPL2).some((k) => k.split("|").every((n) => R.includes(n))))
    for (const [h, r1, m] of g.triples) {
      const hit = g.triples.find(([h2, r2]) => h2 === m && r2 !== r1);
      if (!hit || out.length >= 6) continue;
      const qn = `What is the ${R[hit[1]]} of the ${R[r1]} of ${E[h]}?`;
      out.push({ label: `${qn} → ${E[hit[2]]} (2 hops)`, mode: "ask", text: `Q: ${qn}\nA:` });
      if (out.filter((o) => o.label.endsWith("(2 hops)")).length >= 2) break;
    }
  for (const [key, tpl] of Object.entries(TPL2)) {
    if (out.length >= 6) break;
    const [r1, r2] = key.split("|").map((n) => R.indexOf(n));
    if (r1 < 0 || r2 < 0) continue;
    for (const [h, r, m] of g.triples) {
      if (r !== r1) continue;
      const t = (byHR.get(`${m},${r2}`) || [])[0];
      if (t === undefined) continue;
      const qn = tpl.replace("{h}", E[h]);
      out.push({ label: `${qn} → ${E[t]} (2 hops)`, mode: "ask", text: `Q: ${qn}\nA:` });
      break;
    }
  }
  return out;
}
function renderPresets() {
  const ps = $("#presets");
  const gex = graphExamples(S.state.graph);
  ps.replaceChildren(el("option", { value: "", text: "Examples…" }),
    el("optgroup", { label: "Examples" }, ...PRESETS.map((p, i) => el("option", { value: `p${i}`, text: p.label }))),
    el("optgroup", { label: `From the loaded graph (${S.state.graph.label})` }, ...gex.map((p, i) => el("option", { value: `g${i}`, text: p.label }))));
  S.graphPresets = gex;
}
async function applyPreset(p) {
  if (p.special === "teach") return teach();
  if (p.graph === "strandedcpu" && !(S.state.graph.computed || {})[p.rel]) {
    await graphOp({ op: "strandedcpu" }, { rerun: false });
    status("status", "Loaded the strandedcpu arithmetic graph.");
  }
  if (p.needs && !S.state.graph.entities.includes(p.needs)) {
    await graphOp({ op: "demo" }, { rerun: false });
    status("status", "Loaded the demo graph for this example.");
  }
  setMode(p.mode); $("#prompt").value = p.text; $("#persist").checked = !!p.persist || p.mode === "remember";
  await run();
}
async function teach() {
  setMode("ask"); $("#persist").checked = false; $("#prompt").value = "Q: What is the capital of Velmora?\nA:";
  await graphOp({ op: "demo" }, { rerun: false });
  await run();
  const before = S.run && S.run.answer;
  try {
    await api("/api/graph", { op: "add_entity", name: "Gwenvencor" });
    const res = await api("/api/graph", { op: "set", h: "Velmora", r: "capital", t: "Gwenvencor" });
    S.state.graph = res.graph;
  } catch (e) { status("graph-status", e.message, true); return; }
  await run(S.run.req, { keepEdit: true });
  S.edit = { before, after: S.run.answer, note: "Velmora's capital changed to Gwenvencor." };
  renderPresets(); renderAnswers(); renderGraph();
}

// controls and running
function setMode(m) {
  S.mode = m;
  for (const b of document.querySelectorAll("#mode button")) b.setAttribute("aria-pressed", String(b.dataset.mode === m));
  $("#compare").disabled = m !== "ask"; $("#max-new").disabled = m !== "ask";
  $("#persist").disabled = m === "remember";
  if (m === "remember") { $("#persist").checked = true; $("#use-memory").checked = true; }
}
const readControls = () => ({
  mode: S.mode, text: $("#prompt").value, use_memory: $("#use-memory").checked, persist: $("#persist").checked,
  compare_base: $("#compare").checked, max_new_tokens: +$("#max-new").value || 24,
});
const graphKey = () => { const g = S.state.graph; return g.entities.join("\u0001") + "|" + g.relations.join("\u0001") + "|" + g.triples.map((t) => t.join(",")).join(";"); };
const adapterName = () => S.state.adapter.split("/").pop();

function setupControls() {
  const ps = $("#presets");
  ps.addEventListener("change", async () => {
    const v = ps.value; ps.value = "";
    const p = v.startsWith("p") ? PRESETS[+v.slice(1)] : v.startsWith("g") ? S.graphPresets[+v.slice(1)] : null;
    if (p) { await applyPreset(p); $("#timeline").focus({ preventScroll: true }); }
  });
  for (const b of document.querySelectorAll("#mode button")) b.addEventListener("click", () => setMode(b.dataset.mode));
  $("#run").addEventListener("click", () => run());
  $("#tour-start").addEventListener("click", startTour);
  $("#prompt").addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) { e.preventDefault(); run(); } });
  $("#theme").addEventListener("click", () => { document.documentElement.dataset.theme = isDark() ? "light" : "dark"; renderAll(); });
  $("#adapter").addEventListener("change", async (e) => {
    try {
      status("status", "Loading adapter…");
      S.state = await api("/api/adapter", { path: e.target.value });
      renderStatic();
      if (S.run) await run(S.run.req); else status("status", "");
    } catch (err) { status("status", err.message, true); }
  });
  document.addEventListener("keydown", (e) => {
    if (!S.run || ["TEXTAREA", "INPUT", "SELECT", "BUTTON"].includes(document.activeElement.tagName)) return;
    if (e.key === "ArrowRight") { select(Math.min(S.run.trace.T - 1, S.sel + 1), true); e.preventDefault(); }
    if (e.key === "ArrowLeft") { select(Math.max(0, S.sel - 1), true); e.preventDefault(); }
    if (e.key === "Escape" && S.tour) endTour();
  });
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", renderAll);
  $("#prompt").value = PRESETS[0].text;
}

async function run(override, { keepEdit = false } = {}) {
  if (S.busy) return;
  const req = override || readControls();
  S.busy = true; $("#run").disabled = true;
  for (const c of document.querySelectorAll("main > section:not(#card-prompt)")) c.classList.add("busy");
  status("status", "Running…");
  try {
    const res = await api("/api/run", req);
    Object.assign(res, { req, entities: [...S.state.graph.entities], relations: [...S.state.graph.relations], graphKey: graphKey(), adapter: adapterName() });
    S.run = res; S.state.memory = res.memory; S.strandedcpuRun = null;
    if (!keepEdit) S.edit = null;
    if (S.tour) endTour();
    const hk = `${req.mode}|${req.text}|${req.persist}|${req.use_memory}|${res.graphKey}`;
    if (req.mode === "ask") (S.history[hk] = S.history[hk] || {})[res.adapter] = res.answer;
    res.historyKey = hk;
    S.sel = defaultSel(res);
    renderAll();
    const tr = res.trace;
    status("status", `${tr.T} positions · ${res.ms} ms` + (res.committed ? " · persistent scratchpad updated" : ""));
  } catch (e) { status("status", e.message, true); }
  finally {
    S.busy = false; $("#run").disabled = false;
    for (const c of document.querySelectorAll("main > section")) c.classList.remove("busy");
  }
}

function gainAt(tr, t) { return tr.p_actual[t] === null || tr.p_actual_base[t] === null ? null : tr.p_actual[t] - tr.p_actual_base[t]; }
function defaultSel(res) {
  const tr = res.trace;
  if (res.mode === "ask") return Math.max(0, res.n_prompt - 1);           // the position that starts the answer
  if (res.mode === "remember" && tr.lanes && tr.lanes.beta) {
    let best = -1; tr.lanes.beta.forEach((b, i) => { if (b > 0.5) best = i; });
    if (best >= 0) return best;
  }
  for (let t = tr.T - 2; t > 0; t--)                                         // trace: the answer position after "A:", if any
    if (tr.tokens[t].trim() === ":" && tr.tokens[t - 1].trim() === "A") return t;
  let best = tr.T - 1, bv = -Infinity;                                       // otherwise where the memory helps most
  for (let t = 0; t < tr.T; t++) { const g = gainAt(tr, t); if (g !== null && g > bv) { bv = g; best = t; } }
  return best;
}
function select(t, scroll = false) {
  S.sel = t;
  renderTimeline(scroll); renderSummary(); renderLink(); renderTraversal(); renderPadPos(); renderNext(); renderPad(); renderGraph();
}

// header, answers, notes
function evalText(ad) {
  const e = ad.eval;
  if (!e) return "";
  if (e["1-hop"] !== undefined) return ` · held-out ${["1-hop", "2-hop", "scratchpad"].map((k) => fmt(e[k], 2)).join(" / ")}`;
  if (e.link_acc !== undefined) return ` · stage 1: link ${fmt(e.link_acc, 2)}, slots apart ${fmt(e.slot_apart, 2)}`;
  return "";
}
function renderStatic() {
  const a = S.state.arch;
  $("#model-chip").textContent = `${a.model} · ${a.n_layers} layers · d=${a.hidden} · memory block after layer ${a.layer_idx}`;
  $("#adapter").replaceChildren(...S.state.adapters.map((ad) => el("option", {
    value: ad.path, text: `${ad.name}${ad.stage === 1 ? " (stage 1 only)" : ""}${evalText(ad)}`, selected: ad.path === S.state.adapter ? "selected" : null, title: ad.path })));
  renderPresets(); renderDiagram(); renderGraph(); renderMemory(); renderArch();
}

function renderAnswers() {
  const box = $("#answers"), notes = $("#notes"), r = S.run;
  box.replaceChildren(); notes.replaceChildren();
  if (!r) return;
  const tr = r.trace;
  if (r.mode === "ask") {
    box.append(el("div", { class: "answer" }, el("span", { class: "k", text: tr.use_memory ? "with memory" : "memory off" }), el("span", { class: "v", text: plainTok(r.answer) || "∅" })));
    if (r.base_answer !== undefined) box.append(el("div", { class: "answer" }, el("span", { class: "k", text: "memory off" }), el("span", { class: "v base", text: plainTok(r.base_answer) || "∅" })));
    if (r.no_buffer_answer !== undefined) box.append(el("div", { class: "answer" }, el("span", { class: "k", text: "without the remembered state" }), el("span", { class: "v base", text: plainTok(r.no_buffer_answer) || "∅" })));
  } else if (r.mode === "remember") {
    const n = (tr.lanes && tr.lanes.beta || []).filter((b) => b > 0.5).length;
    box.append(el("div", { class: "answer" }, el("span", { class: "k", text: "remembered" }), el("span", { class: "v base", text: `${n} strong write${n === 1 ? "" : "s"} into the persistent scratchpad` })));
  } else {
    box.append(el("div", { class: "answer" }, el("span", { class: "k", text: "traced" }), el("span", { class: "v base", text: `${tr.T} tokens, memory ${tr.use_memory ? "on" : "off"}` })));
  }
  const note = (text, warn = false) => notes.append(el("div", { class: "notice" + (warn ? " warn" : ""), text }));
  if (S.edit) note(`Before the edit: ${plainTok(S.edit.before || "∅")}  →  after: ${plainTok(S.edit.after || "∅")}.  ${S.edit.note}`);
  const hist = S.history[r.historyKey];
  if (hist && Object.keys(hist).length > 1) {
    const h = el("div", { class: "history" }, document.createTextNode("Answers by adapter: "));
    Object.entries(hist).forEach(([a, ans], i) => { if (i) h.append(document.createTextNode("  ·  ")); h.append(el("b", { text: a }), document.createTextNode(` → ${plainTok(ans) || "∅"}`)); });
    notes.append(h);
  }
  if (r.truncated) note(`The prompt was cut to its first ${r.max_chars} of ${r.n_chars} characters.`, true);
  if (r.req.persist && r.buffer_empty && r.mode !== "remember") note("The saved scratchpad is empty. Use Remember to write values first.", true);
  const mentionsEntity = S.state.graph.entities.some((e) => r.req.text.includes(e));
  if (r.mode === "ask" && tr.use_memory && !mentionsEntity) note("No graph entity name matches this prompt verbatim. Check the graph and spelling if you expected a graph answer.");
  if (!tr.use_memory) note("Memory is off; this trace shows the base model.");
}

// diagram
function renderDiagram() {
  const a = S.state.arch, L = a.layer_idx, N = a.n_layers;
  const node = (label, card, cls = "", n = null) => el("button", { type: "button", class: `node ${cls}`, onclick: () => spot(card) },
    n ? el("span", { class: "n", text: String(n) }) : null, document.createTextNode(label));
  const ar = (t = "→") => el("span", { class: "arrow", "aria-hidden": "true", text: t });
  const block = el("div", { class: "block" },
    el("div", { class: "lab", text: "memory block (the only trained part)" }),
    el("div", { class: "row" }, node("Entity linking", "card-link", "kg", 1), ar(), node("Graph traversal", "card-traversal", "kg", 2), ar(), node("knowledge graph", "card-graph", "kg")),
    el("div", { class: "row" }, node("Scratchpad: write · slots · read", "card-pad-pos", "pad", 3), ar("↔"), node("persistent buffer", "card-memory", "pad")));
  $("#diagram").replaceChildren(el("div", { class: "diagram", style: "display:flex;align-items:center;gap:6px;flex-wrap:wrap" },
    node("your tokens", "card-timeline"), ar(), node(`layers 0–${L} (frozen)`, "card-arch", "frozen"), ar(), block, ar(),
    node(`layers ${L + 1}–${N - 1} (frozen)`, "card-arch", "frozen"), ar(), node("LM head + readout", "card-next", "frozen"), ar(),
    node("Copy head → next token", "card-next", "kg", 4)));
}

// timeline
const LANES = [
  { key: "p_actual", label: "p(actual next token)", scale: "abs", get: (tr) => tr.p_actual },
  { key: "gain", label: "memory effect", scale: "div", info: "gain", get: (tr) => tr.p_actual.map((_, t) => gainAt(tr, t)) },
  { key: "kg_top", label: "graph answer, top weight", scale: "abs", get: (tr) => tr.kg && tr.kg.map((k) => (k.final.length ? k.final[0][1] : 0)) },
  { key: "g_kg", label: "graph gate", scale: "abs", info: "gkg" },
  { key: "d_kg", label: "graph |Δ|/|h|", scale: "max", info: "dkg" },
  { key: "halt2", label: "halting: 2-hop share", scale: "abs", info: "halt" },
  { key: "lam", label: "copy gate λ", scale: "abs", info: "lam", get: (tr) => tr.next.map((n) => n.lam ?? null) },
  { key: "mass", label: "pointer mass", scale: "abs", info: "mass", get: (tr) => tr.next.map((n) => n.mass ?? null) },
  { key: "g_pad", label: "scratchpad gate", scale: "abs", info: "gpad" },
  { key: "g_ro", label: "readout gate", scale: "abs", info: "gro" },
  { key: "beta", label: "write strength β", scale: "abs", info: "beta" },
  { key: "d_pad", label: "scratchpad |Δ|/|h|", scale: "max", info: "dkg" },
];
const measure = (() => { const cv = document.createElement("canvas").getContext("2d"); return (s) => { cv.font = "12px ui-monospace, SFMono-Regular, Menlo, monospace"; return cv.measureText(s).width; }; })();
const tokKind = (r, t) => (r.mode === "ask" && t >= r.n_prompt ? "gen" : t === 0 && /^<.*>$/.test(r.trace.tokens[0]) ? "bos" : "prompt");

function jumpTargets(kind) {
  const tr = S.run.trace, out = [];
  for (let t = 0; t < tr.T; t++) {
    if (kind === "write" && tr.lanes && tr.lanes.beta && tr.lanes.beta[t] > 0.5) out.push(t);
    if (kind === "mention" && tr.mention_strength && tr.mention_strength[t] > 0.5) out.push(t);
    if (kind === "gain") { const g = gainAt(tr, t); if (g !== null && Math.abs(g) > 0.2) out.push(t); }
    if (kind === "gen" && tokKind(S.run, t) === "gen") out.push(t);
  }
  return out;
}
function renderTools() {
  const box = $("#tl-tools"); box.replaceChildren();
  if (!S.run) return;
  const kinds = [["gain", "memory changes the next token"], ["write", "strong writes"], ["mention", "tokens used as mentions"], ["gen", "generated tokens"]];
  const pick = el("select", { "aria-label": "Jump to" }, ...kinds.map(([k, l]) => el("option", { value: k, text: `${l} (${jumpTargets(k).length})` })));
  pick.value = S.jumpKind || "gain";
  pick.addEventListener("change", () => { S.jumpKind = pick.value; });
  const go = (dir) => () => {
    const ts = jumpTargets(pick.value);
    const t = dir > 0 ? ts.find((x) => x > S.sel) : [...ts].reverse().find((x) => x < S.sel);
    if (t !== undefined) select(t, true); else status("status", `No ${dir > 0 ? "later" : "earlier"} position of that kind.`);
  };
  box.append(el("span", { class: "note", text: "Jump to" }), pick,
    el("button", { class: "ghost small", type: "button", text: "◀ previous", onclick: go(-1) }),
    el("button", { class: "ghost small", type: "button", text: "next ▶", onclick: go(1) }));
}

function renderTimeline(scroll = false) {
  const box = $("#timeline"), r = S.run;
  box.replaceChildren(); $("#tl-legend").replaceChildren(); $("#overview-wrap").replaceChildren(); $("#tl-table").replaceChildren();
  renderTools();
  if (!r) return;
  const tr = r.trace, T = tr.T, c = C();
  $("#tl-legend").append(legend([
    { label: "mention attention", color: c.s1, kind: "line", info: "mention" }, { label: "write address from", color: c.s2, kind: "line", info: "addr" },
    { label: "read address from", color: c.s3, kind: "line" }, { label: "0 → max", kind: "ramp" }, { label: "memory hurts ↔ helps", kind: "div", info: "gain" }]));
  const lanes = LANES.map((L) => ({ ...L, vals: L.get ? L.get(tr) : (tr.lanes || {})[L.key] }))
    .filter((L) => L.vals && L.vals.some((v) => v !== null && v !== undefined));
  const ARC = 72, TOKY = ARC, TOKH = 26, LY = TOKY + TOKH + 8, LH = 14, LG = 3, GAP = 2;
  const H = LY + lanes.length * (LH + LG) + 4;
  const widths = tr.tokens.map((s) => Math.max(24, Math.min(120, measure(showTok(s)) + 12)));
  const xs = []; let acc = 4;
  widths.forEach((w) => { xs.push(acc); acc += w + GAP; });
  const W = acc + 4, cx = (t) => xs[t] + widths[t] / 2;
  // labels: HTML so the ⓘ buttons are real buttons
  const labels = el("div", { class: "labels", style: `position:relative;height:${H}px` });
  labels.append(el("div", { class: "note", style: `position:absolute;left:8px;top:${TOKY + 5}px`, text: "token" }));
  lanes.forEach((L, i) => {
    const vals = L.vals.filter((v) => v !== null && v !== undefined);
    L.max = L.scale === "max" ? Math.max(...vals, 1e-9) : 1;
    const scaleTxt = L.scale === "abs" ? "0–1" : L.scale === "div" ? "−1…+1" : `0–max ${fmt(L.max, 2)}`;
    labels.append(el("div", { style: `position:absolute;left:8px;right:4px;top:${LY + i * (LH + LG) - 2}px;font-size:11px;color:var(--ink-2);white-space:nowrap;overflow:hidden` },
      document.createTextNode(`${L.label} · ${scaleTxt}`), L.info ? info(L.info) : null));
  });
  const svg = sv("svg", { width: W, height: H, "aria-hidden": "true" });
  // selected column: wash plus a 2px outline (the wash alone was ~1.1:1 against the surface)
  svg.append(sv("rect", { x: xs[S.sel] - 1, y: TOKY - 2, width: widths[S.sel] + 2, height: H - TOKY + 2, fill: c.wash, rx: 4 }));
  svg.append(sv("rect", { x: xs[S.sel] - 1, y: TOKY - 2, width: widths[S.sel] + 2, height: TOKH + 2, fill: "none", stroke: c.s1, "stroke-width": 2, rx: 4 }));
  if (r.mode === "ask" && r.n_prompt < T) svg.append(sv("line", { x1: xs[r.n_prompt] - 1.5, x2: xs[r.n_prompt] - 1.5, y1: TOKY - 4, y2: H, stroke: c.edge }));
  // Fade write arcs below β = 0.1.
  const beta = tr.lanes && tr.lanes.beta ? tr.lanes.beta[S.sel] : 0;
  const arcs = [];
  if (tr.mention_att) arcs.push({ list: tr.mention_att[S.sel], color: c.s1, k: "mention attention", lift: 1, fade: 1 });
  if (tr.pad) {
    arcs.push({ list: tr.pad.ptr_w[S.sel], color: c.s2, k: "write address comes from", lift: 0.82, fade: beta < 0.1 ? 0.2 : 1 });
    arcs.push({ list: tr.pad.ptr_r[S.sel], color: c.s3, k: "read address comes from", lift: 0.64, fade: 1 });
  }
  for (const a of arcs) for (const [s, w] of a.list || []) {
    if (w < 0.05) continue;
    const x1 = cx(S.sel), x2 = cx(s);
    const path = s === S.sel ? `M${x1 - 5},${TOKY} C${x1 - 10},${TOKY - 22 * a.lift} ${x1 + 10},${TOKY - 22 * a.lift} ${x1 + 5},${TOKY}`
      : (() => { const h = Math.min(ARC - 8, 14 + Math.abs(x1 - x2) * 0.25) * a.lift; return `M${x1},${TOKY} Q${(x1 + x2) / 2},${TOKY - 2 * h} ${x2},${TOKY}`; })();
    svg.append(sv("path", { d: path, fill: "none", stroke: a.color, "stroke-width": 1 + 2.5 * w, "stroke-linecap": "round", opacity: (0.35 + 0.65 * w) * a.fade }));
    const hit = sv("path", { d: path, fill: "none", stroke: "transparent", "stroke-width": 10 });
    hover(hit, () => [{ v: fmt(w, 3), k: `${a.k}: ${q(tr.tokens[s])} @${s}${a.fade < 1 ? " (write strength below 0.1)" : ""}`, color: a.color }]);
    svg.append(hit);
  }
  tr.tokens.forEach((s, t) => {
    const kind = tokKind(r, t);
    const g = sv("g", { style: "cursor:pointer" });
    g.append(sv("rect", { x: xs[t], y: TOKY, width: widths[t], height: TOKH, fill: "transparent" }));
    g.append(sv("text", { x: cx(t), y: TOKY + 17, "text-anchor": "middle", class: `tok-text ${kind}`, text: showTok(s) }));
    hover(g, () => [{ v: JSON.stringify(s), k: `position ${t} · ${kind}` }]);
    g.addEventListener("click", () => select(t));
    svg.append(g);
  });
  lanes.forEach((L, i) => {
    const y = LY + i * (LH + LG);
    L.vals.forEach((v, t) => {
      if (v === null || v === undefined) return;
      const cell = sv("rect", { x: xs[t], y, width: widths[t], height: LH, rx: 2, fill: L.scale === "div" ? div(v) : seq(v / L.max) });
      hover(cell, () => [{ v: fmt(v, 4), k: `${L.label} · ${q(tr.tokens[t])} @${t}` }]);
      cell.addEventListener("click", () => select(t));
      svg.append(cell);
    });
  });
  const scrollBox = el("div", { class: "scroll" }, svg);
  box.append(labels, scrollBox);
  S.tl = { xs, widths, W, scrollBox };
  if (scroll || !renderTimeline.done) scrollBox.scrollLeft = Math.max(0, xs[S.sel] - scrollBox.clientWidth / 2);
  renderTimeline.done = true;
  renderOverview(lanes.find((L) => L.key === "gain") || lanes[0]);
  // table view
  $("#tl-table").append(table(["position", "token", ...lanes.map((L) => L.label)],
    tr.tokens.map((s, t) => [String(t), showTok(s), ...lanes.map((L) => fmt(L.vals[t], 3))])));
}

function renderOverview(lane) {
  const wrap = $("#overview-wrap"), tl = S.tl;
  if (!tl || tl.W <= tl.scrollBox.clientWidth + 8) return;            // only when the sequence doesn't fit
  const c = C();
  const svg = sv("svg", { class: "overview", viewBox: `0 0 ${tl.W} 26`, preserveAspectRatio: "none", role: "img", "aria-label": `Overview of all ${S.run.trace.T} positions; click to scroll there` });
  lane.vals.forEach((v, t) => { if (v !== null && v !== undefined) svg.append(sv("rect", { x: tl.xs[t], y: 4, width: tl.widths[t], height: 18, fill: lane.scale === "div" ? div(v) : seq(v / lane.max) })); });
  svg.append(sv("rect", { x: tl.xs[S.sel], y: 0, width: tl.widths[S.sel], height: 26, fill: c.s1 }));
  const win = sv("rect", { y: 1, height: 24, fill: "none", stroke: c.ink, "stroke-width": 2, "vector-effect": "non-scaling-stroke" });
  const place = () => { win.setAttribute("x", tl.scrollBox.scrollLeft); win.setAttribute("width", tl.scrollBox.clientWidth); };
  place(); svg.append(win);
  tl.scrollBox.addEventListener("scroll", place);
  svg.addEventListener("click", (e) => { const r = svg.getBoundingClientRect(); tl.scrollBox.scrollLeft = ((e.clientX - r.left) / r.width) * tl.W - tl.scrollBox.clientWidth / 2; });
  wrap.append(el("div", { class: "note", text: `Overview (${lane.label}); the box is the visible part. Click to move it.` }), svg);
}

function renderSummary() {
  const box = $("#summary"), r = S.run;
  box.replaceChildren();
  if (!r) return;
  const tr = r.trace, t = S.sel, nx = tr.next[t], k = tr.kg && tr.kg[t];
  const item = (label, value) => el("span", {}, el("span", { class: "k", text: label }), document.createTextNode(value));
  box.append(el("span", { class: "tokbig", text: `${t} · ${JSON.stringify(tr.tokens[t])}` }));
  const top = nx.cands[0];
  if (top) box.append(item("top prediction", `${showTok(top[0])} ${fmt(top[4], 2)}`));
  if (nx.actual !== null && nx.actual !== undefined) box.append(item("actual next", `${showTok(tr.tokens[t + 1])}: ${fmt(tr.p_actual_base[t], 2)} without memory → ${fmt(tr.p_actual[t], 2)} with`));
  if (k) {
    const m = (tr.mention_att[t] || [])[0];
    box.append(item("top head", k.heads.length ? `${ent(k.heads[0][0])}${m ? ` (via ${q(tr.tokens[m[0]])} @${m[0]})` : ""}` : "–"));
    box.append(item("top graph candidate", k.final.length ? `${ent(k.final[0][0])} ${fmt(k.final[0][1], 2)}` : "–"));
  }
  if (nx.lam !== undefined) box.append(item("λ", fmt(nx.lam, 2)));
  if (tr.pad) {
    const b = tr.lanes.beta[t];
    box.append(item("write", b < 0.1 ? "none" : tr.pad.cont ? `under ${via(tr.pad, t)}, β ${fmt(b, 2)}` : `slot ${argmax(tr.pad.kw[t])}, β ${fmt(b, 2)}`));
    box.append(item("read", tr.pad.cont ? `under ${via(tr.pad, t, tr.pad.ptr_r)}` : `slot ${argmax(tr.pad.kr[t])}`));
  }
}

// step 1: linking
function renderLink() {
  const card = $("#card-link"), r = S.run;
  cardHead(card, 1, "Entity linking", "Mention attention selects token positions; their entity candidates become graph heads.", "mention");
  if (!r) return;
  const tr = r.trace, t = S.sel, k = tr.kg && tr.kg[t], c = C();
  if (!k) { card.append(el("div", { class: "empty", text: tr.use_memory ? "This adapter has no graph head." : "Memory is off for this run." })); return; }
  card.append(el("h3", { text: `Mention attention from ${q(tr.tokens[t])} @${t}` }));
  card.append(barList((tr.mention_att[t] || []).map(([s, w]) => ({ name: `${showTok(tr.tokens[s])} @${s}`, v: w, onclick: () => select(s, true), tip: `click to inspect @${s}` })), { color: c.s1, max: 1 }));
  card.append(el("h3", { text: "Inherited head candidates" }));
  card.append(barList(k.heads.map(([e, w]) => ({ name: ent(e), v: w })), { color: c.s1, max: 1 }));
  const used = (tr.mention_strength || [])[t] >= 0.3;
  card.append(el("h3", {}, document.createTextNode("This token's own links"), info("link")));
  card.append(el("div", { class: "axis-note", text: used ? "Weights sum to one over the top 4 keys. Cosine reports similarity." :
    "Mention attention to this token is below the display threshold. Link weights sum to one over the top 4 keys." }));
  card.append(barList(k.link.map(([e, w, cos]) => ({ name: ent(e), v: w, label: `${fmt(w, 2)} · cos ${fmt(cos, 2)}`, tip: `${ent(e)}: renormalised ${fmt(w, 3)}, cosine ${fmt(cos, 3)}` })),
    { color: c.s1, max: 1, faded: !used }));
}

// step 2: traversal
function renderTraversal() {
  const card = $("#card-traversal"), r = S.run;
  cardHead(card, 2, "Graph traversal", "Each hop follows a relation from its head candidates. Halting weights mix answers across hop counts.", "hop");
  if (!r) return;
  const tr = r.trace, k = tr.kg && tr.kg[S.sel], c = C(), nx = tr.next[S.sel];
  if (!k) { card.append(el("div", { class: "empty", text: "No graph trace for this run." })); return; }
  card.append(legend([{ label: "hop 1", color: c.s1 }, { label: "hop 2", color: c.s1d }]));
  const chips = (rels) => el("div", {}, ...rels.map(([i, w]) => el("span", { class: "chip-rel" }, document.createTextNode(rel(i) + " "), el("b", { text: fmt(w, 2) }))));
  const col = (title, ...nodes) => el("div", {}, el("div", { class: "colhead", text: title }), ...nodes);
  const namedLine = (hop) => hop && hop.named && hop.named.length ? el("div", { class: "axis-note",
    text: `relation named by ${hop.named.map(([s, w]) => `${q(tr.tokens[s])} @${s} (${fmt(w, 2)})`).join(", ")} · mention-path weight ${fmt(hop.trust, 2)}` +
      (hop.reads_as ? ` · that token reads as ${hop.reads_as.map(([i, w]) => `${rel(i)} ${fmt(w, 2)}`).join(", ")}` : "") }) : null;
  const strandedcpuLine = (hop) => hop && hop.strandedcpu ? el("div", { class: "axis-note" },
    document.createTextNode(hop.strandedcpu.fault ? `strandedcpu refused ${hop.strandedcpu.relation}(${hop.strandedcpu.x}): ${hop.strandedcpu.fault.replace(/_/g, " ")}; no tail returned `
      : `computed by strandedcpu: ${hop.strandedcpu.relation}(${hop.strandedcpu.x}) = ${hop.strandedcpu.y} · ${hop.strandedcpu.steps} instructions${hop.strandedcpu.in_graph ? "" : " · outside the graph, so no tail"} `),
    el("button", { class: "ghost small", type: "button", text: "Trace on transformer", onclick: () => strandedcpuRun(hop.strandedcpu.relation, hop.strandedcpu.x) })) : null;
  card.append(el("div", { class: "flow" },
    col("heads", barList(k.heads.map(([e, w]) => ({ name: ent(e), v: w })), { color: c.ink2, max: 1 })),
    col("hop 1: relation → tails", chips(k.hops[0].rels), namedLine(k.hops[0]), barList(k.hops[0].tails.map(([e, w]) => ({ name: ent(e), v: w })), { color: c.s1, max: 1, empty: "no edge for these (head, relation) pairs" }), strandedcpuLine(k.hops[0])),
    k.hops[1] ? col("hop 2: relation → tails", chips(k.hops[1].rels), namedLine(k.hops[1]), barList(k.hops[1].tails.map(([e, w]) => ({ name: ent(e), v: w })), { color: c.s1d, max: 1, empty: "no edge for these (head, relation) pairs" }), strandedcpuLine(k.hops[1])) : el("span")));
  card.append(el("div", { id: "strandedcpu-run" }));
  if (S.strandedcpuRun) renderStrandedCPURun();
  const h = k.halt;
  const segs = el("div", { style: "display:flex;gap:2px;max-width:420px;height:10px;margin:6px 0 2px", role: "img", "aria-label": `halting: one hop ${fmt(h[0], 2)}, two hops ${fmt(h[1] ?? 0, 2)}` });
  h.forEach((v, i) => { const s = el("div", { style: `width:${(v * 100).toFixed(2)}%;background:${i ? c.s1d : c.s1};border-radius:${i ? "0 4px 4px 0" : "4px 0 0 4px"}` }); hover(s, () => [{ v: fmt(v, 4), k: `stop after ${i + 1} hop${i ? "s" : ""}`, color: i ? c.s1d : c.s1 }]); segs.append(s); });
  const lam = nx.lam ?? 0;
  card.append(el("div", { class: "grid2", style: "margin-top:10px" },
    el("div", {}, el("h3", {}, document.createTextNode(`Halting: 1 hop ${fmt(h[0], 2)} · 2 hops ${fmt(h[1] ?? 0, 2)}`), info("halt")), segs),
    el("div", {}, el("h3", { text: "Final candidates (what the copy head can spell)" }),
      el("div", { class: "axis-note", text: lam >= 0.3 ? `Copy contribution: λ = ${fmt(lam, 2)}.` : `Faded below the display threshold (λ = ${fmt(lam, 3)}).` }),
      barList(k.final.map(([e, w]) => ({ name: ent(e), v: w })), { color: c.ink2, max: 1, faded: lam < 0.3 }))));
}

async function strandedcpuRun(relation, x) {
  S.strandedcpuRun = { relation, x, busy: true };
  renderStrandedCPURun();
  try { S.strandedcpuRun = await api("/api/strandedcpu", { relation, x: +x, backend: "transformer" }); }
  catch (e) { S.strandedcpuRun = { relation, x, error: e.message }; }
  renderStrandedCPURun();
}
function renderStrandedCPURun() {
  const box = document.getElementById("strandedcpu-run"), L = S.strandedcpuRun;
  if (!box || !L) return;
  box.replaceChildren(el("h3", { text: `strandedcpu execution: ${L.relation}(${L.x})` }));
  if (L.busy) { box.append(el("div", { class: "note", text: "Running transformer trace…" })); return; }
  if (L.error) { box.append(el("div", { class: "status err", text: L.error })); return; }
  box.append(el("div", {}, statTile("result y", String(L.y)), statTile("instructions executed", String(L.steps)), statTile(`on the ${L.backend}`, `${fmt(L.seconds, 2)} s`)),
    L.agrees !== undefined ? el("div", { class: L.agrees ? "pill ok" : "pill bad", text: L.agrees ? "Transformer and interpreter traces agree" : "Transformer and interpreter traces differ" }) : null,
    el("pre", { class: "meta", text: L.source }),
    el("details", {}, el("summary", { text: `Instruction trace (${L.trace.length} steps; writes: old → new)` }),
      el("div", { class: "table-scroll" }, table(["step", "pc", "op", "writes"], L.trace.map((t, i) => [String(i), String(t.pc), t.op,
        Object.entries(t.writes).map(([a, [o, n]]) => `mem[${a}]: ${o} → ${n}`).join("  ")])))),
    el("button", { class: "ghost small", type: "button", text: "close", onclick: () => { S.strandedcpuRun = null; box.replaceChildren(); } }));
}

// step 3: scratchpad here
function overlaps(t) {
  const r = S.run, tr = r.trace, pd = tr.pad, kr = pd.kr[t], out = [];
  for (let u = 0; u < t; u++) {
    const b = tr.lanes.beta[u]; if (b < 0.05) continue;
    const src = pd.ptr_w[u][0];
    out.push({ name: `${showTok(tr.tokens[u])} @${u}${src ? ` via ${showTok(tr.tokens[src[0]])}` : ""}`, v: kr.reduce((s, x, i) => s + x * pd.kw[u][i], 0) * b });
  }
  if (r.req.persist && r.mode !== "remember")
    for (const w of S.state.memory.log || []) out.push({ name: `${showTok(w.token)} via ${showTok(w.via)} (remembered)`, v: kr.reduce((s, x, i) => s + x * w.kw[i], 0) * w.beta });
  return out.filter((o) => o.v > 0.01).sort((a, b) => b.v - a.v).slice(0, 5);
}
function renderPadPos() {
  const card = $("#card-pad-pos"), r = S.run;
  const cont = !!(r && r.trace.pad && r.trace.pad.cont);
  cardHead(card, 3, "Scratchpad at this position", `Read before write. A shared ${cont ? "key map" : "slot map"} turns attended token states into addresses.`, cont ? "addrc" : "addr");
  if (!r) return;
  const tr = r.trace, pd = tr.pad, t = S.sel, c = C(), nx = tr.next[t];
  if (!pd) { card.append(el("div", { class: "empty", text: tr.use_memory ? "This adapter has no scratchpad." : "Memory is off for this run." })); return; }
  const b = tr.lanes.beta[t];
  card.append(el("div", {}, statTile("write strength β", fmt(b, 3), "beta"), statTile("scratchpad gate", fmt(tr.lanes.g_pad[t], 3), "gpad"),
    tr.lanes.g_ro ? statTile("readout gate", fmt(tr.lanes.g_ro[t], 3), "gro") : null, statTile("|Δ|/|h|", fmt(tr.lanes.d_pad[t], 3), "dkg")));
  const src = (list) => (list && list.length ? list.slice(0, 3).map(([s, w]) => `${q(tr.tokens[s])} @${s} (${fmt(w, 2)})`).join(", ") : "–");
  card.append(el("h3", { text: "Write" }));
  if (b < 0.1) card.append(el("div", { class: "axis-note", text: `Small write (β = ${fmt(b, 3)}); the address is shown faded.` }));
  card.append(el("div", { class: "note" + (b < 0.1 ? " faded" : ""), text: `address from ${src(pd.ptr_w[t])}` }));
  if (!cont) { const wc = slotColumns(pd.kw[t], c.s2, { max: 1, label: "write address" }); if (b < 0.1) wc.classList.add("faded"); card.append(wc); }
  card.append(el("h3", { text: "Read" }));
  card.append(el("div", { class: "note", text: `address from ${src(pd.ptr_r[t])}` }));
  if (!cont) card.append(slotColumns(pd.kr[t], c.s3, { max: 1, label: "read address" }));
  card.append(el("h3", {}, document.createTextNode("Read/write address overlap"), info(cont ? "overlapc" : "overlap")));
  card.append(barList(overlaps(t).map((o) => ({ name: o.name, v: o.v })), { color: c.s2, max: 1, empty: "No write overlap above the display threshold." }));
  if (nx.readout) {
    card.append(el("h3", {}, document.createTextNode("Scratchpad logit bias"), info("readout")));
    card.append(barList(nx.readout.map(([s, v]) => ({ name: showTok(s), v, label: `+${fmt(v, 2)} logits` })), { color: c.s3 }));
  }
}

// step 4: next token
const STAGES = ["base LM (memory off)", "upper layers after injection", "+ scratchpad readout", "final (copy head mixed in)"];
const logPos = (p) => Math.max(0, (Math.log10(Math.max(p, 1e-4)) + 4) / 4);       // 1e-4 … 1  ->  0 … 1
function renderNext() {
  const card = $("#card-next"), r = S.run;
  cardHead(card, 4, "Next-token prediction", "Next-token probabilities after each memory contribution.", "stages");
  if (!r) return;
  const tr = r.trace, t = S.sel, c = C(), nx = tr.next[t];
  const actual = nx.actual !== null && nx.actual !== undefined ? tr.tokens[t + 1] : null;
  card.append(el("p", { class: "note", text: `At ${q(tr.tokens[t])} @${t}${actual !== null ? `, predicting the next token (${q(actual)} in this text)` : ""}.` }));
  if (nx.lam !== undefined) card.append(el("div", { style: "margin-bottom:6px" }, statTile("copy gate λ", fmt(nx.lam, 3), "lam"), statTile("pointer mass", fmt(nx.mass, 3), "mass"),
    actual !== null ? statTile("p(actual) off → on", `${fmt(tr.p_actual_base[t], 2)} → ${fmt(tr.p_actual[t], 2)}`, "gain") : null));
  if (nx.carry) card.append(el("span", { class: "pill", text: "spelling a name started earlier" }));
  if (nx.completed) card.append(el("span", { class: "pill", text: "candidate name completed" }));
  card.append(legend(STAGES.map((s, i) => ({ label: s, color: c.st[i] }))));
  const box = el("div", { class: "bars" });
  for (const [s, ...rest] of nx.cands) {
    const ps = rest.slice(0, 4), id = rest[4], isActual = id === nx.actual;
    const track = el("div", { class: "track row-hit", style: "height:28px" });
    ps.forEach((v, i) => track.append(el("div", { class: "bar", style: `width:${(logPos(v) * 78).toFixed(2)}%;height:4px;background:${c.st[i]}` })));
    track.append(el("span", { class: "val", style: `left:calc(${(Math.max(...ps.map(logPos)) * 78).toFixed(2)}% + 6px);top:7px`, text: fmt(ps[3], 3) }));
    hover(track, () => [...ps.map((v, i) => ({ v: fmt(v, 4), k: STAGES[i], color: c.st[i] })).reverse()]);
    box.append(el("div", { class: "name", text: (isActual ? "▸ " : "  ") + showTok(s), title: isActual ? "the following token in this sequence" : s }), track);
  }
  const ticks = el("div", { class: "track", style: "height:14px" });
  ["0.0001", "0.001", "0.01", "0.1", "1"].forEach((l, i) => ticks.append(el("span", { class: "val", style: `left:calc(${(i / 4 * 78).toFixed(2)}% - 6px);top:0`, text: l })));
  box.append(el("div", { class: "name dim", text: "log scale" }), ticks);
  card.append(box);
  if (nx.pointer) {
    const lam = nx.lam ?? 0;
    card.append(el("h3", { text: "Copy-head probabilities × λ" }));
    if (lam < 0.05) card.append(el("div", { class: "axis-note", text: `Small copy contribution (λ = ${fmt(lam, 3)}).` }));
    card.append(barList(nx.pointer.map(([s, v]) => ({ name: showTok(s), v: lam * v, tip: `${showTok(s)}: λ × p_pointer = ${fmt(lam, 3)} × ${fmt(v, 3)}` })), { color: c.s1, max: 1, faded: lam < 0.05, empty: "no candidate names proposed here" }));
  }
  card.append(el("details", {}, el("summary", { text: "Table" }),
    table(["token", ...STAGES], nx.cands.map(([s, ...p]) => [showTok(s), ...p.slice(0, 4).map((v) => fmt(v, 4))]))));
}

// scratchpad over time
function slotTable(slots, log, emptyText) {
  if (!slots.length) return el("div", { class: "empty", text: emptyText });
  const max = Math.max(...slots.map((x) => x.norm)), cont = slots[0].label !== undefined, unit = cont ? "variables" : "slots";
  const last = {};
  (log || []).forEach((w) => { last[cont ? w.via.trim() : argmax(w.kw)] = w; });
  const key = (x) => (cont ? x.label : x.slot);
  const row = (x) => [cont ? x.label : String(x.slot), fmt(x.norm, 2), x.tokens.map(([s, v]) => `${showTok(s)} (+${fmt(v, 1)})`).join("  "),
    last[key(x)] ? `${showTok(last[key(x)].token)} via ${showTok(last[key(x)].via)}` : ""];
  const head = [cont ? "variable" : "slot", cont ? "read norm" : "norm", "reads as (logit bias)", ...(log ? ["last strong write"] : [])];
  // near-empty: under 5% of the largest norm, or under 0.5 outright (an empty memory's rows are ~0.01)
  const cut = Math.max(0.05 * max, 0.5);
  const big = slots.filter((x) => x.norm >= cut), small = slots.filter((x) => x.norm < cut);
  const fix = (rows) => (log ? rows : rows.map((r) => r.slice(0, 3)));
  const box = el("div", {}, big.length ? table(head, fix(big.map(row))) : el("div", { class: "empty", text: `${emptyText} (largest norm ${fmt(max, 3)})` }));
  if (small.length) box.append(el("details", {}, el("summary", { text: `${small.length} near-empty ${unit} (norm under ${fmt(cut, 2)})` }), table(head, fix(small.map(row)))));
  return box;
}
function renderPad() {
  const card = $("#card-pad"), r = S.run;
  const cont = !!(r && r.trace.pad && r.trace.pad.cont), unit = cont ? "variables" : "slots";
  cardHead(card, null, "Scratchpad state over the sequence", cont ? "Read norms along written variable keys after each position. Replayed from the recorded updates."
    : "Slot norms after each position, replayed from the recorded updates.", cont ? "varnorm" : "slotnorm");
  if (!r || !r.trace.pad) { card.append(el("div", { class: "empty", text: "No scratchpad trace for this run." })); return; }
  const pd = r.trace.pad, tr = r.trace, T = tr.T, c = C(), N = pd.slot_norms;
  const wroteHere = (t, s) => tr.lanes.beta[t] > 0.5 && (cont ? plainTok(tr.tokens[pd.ptr_w[t][0] ? pd.ptr_w[t][0][0] : t]).trim() === pd.labels[s] : argmax(pd.kw[t]) === s);
  card.append(el("span", { class: pd.replay_ok ? "pill ok" : "pill bad", text: pd.replay_ok ? "replay matches the block's own final state" : "replay disagrees with the block" }));
  const gmax = Math.max(...N.flat(), ...pd.start_norms, 0);
  const dmax = Math.max(gmax, 1), thresh = Math.max(0.5, 0.05 * gmax);
  const used = [];
  for (let s = 0; s < pd.n_slots; s++) if (Math.max(pd.start_norms[s], ...N.map((row) => row[s])) > thresh) used.push(s);
  card.append(legend([{ label: `${cont ? "read norm" : "slot norm"}, 0 → ${fmt(dmax, 1)} (absolute)`, kind: "ramp" }, { label: "strong write (β > 0.5) here", color: c.ink, kind: "dot" }]));
  card.append(el("p", { class: "note", text: used.length ? `${used.length} of ${pd.n_slots} ${unit} hold something (norm above ${fmt(thresh, 1)}).${pd.start_norms.some((v) => v > 1e-4) ? " This run started from the persistent buffer." : ""}`
    : cont && !pd.n_slots ? "No writes with β > 0.5 were recorded." : `No slot exceeds the display threshold. Largest norm: ${fmt(gmax, 3)}.` }));
  if (used.length) {
    const cw = Math.max(5, Math.min(14, 760 / T)), ch = 12, g = 1, padL = cont ? 78 : 46, padT = 6;
    const W = padL + T * (cw + g) + 8, H = padT + used.length * (ch + g) + 30;
    const svg = sv("svg", { width: W, height: H, role: "img", "aria-label": `Norms of ${used.length} scratchpad ${unit} across ${T} positions; the table below has the values` });
    svg.append(sv("rect", { x: padL + S.sel * (cw + g) - 1, y: 0, width: cw + 2, height: H - 18, fill: "none", stroke: c.s1, "stroke-width": 2, rx: 2 }));
    used.forEach((s, i) => {
      const y = padT + i * (ch + g);
      svg.append(sv("text", { x: padL - 6, y: y + 9.5, "text-anchor": "end", "font-size": 10, fill: c.ink2, text: cont ? pd.labels[s] : `slot ${s}` }));
      for (let t = 0; t < T; t++) {
        const v = N[t][s];
        const cell = sv("rect", { x: padL + t * (cw + g), y, width: cw, height: ch, rx: 1.5, fill: seq(v / dmax) });
        hover(cell, () => [{ v: fmt(v, 3), k: `${slotName(pd, s)} after ${q(tr.tokens[t])} @${t}` }]);
        cell.addEventListener("click", () => select(t, true));
        svg.append(cell);
        if (wroteHere(t, s)) svg.append(sv("circle", { cx: padL + t * (cw + g) + cw / 2, cy: y + ch / 2, r: 2.2, fill: c.ink, "pointer-events": "none" }));
      }
    });
    const yAx = padT + used.length * (ch + g) + 12;
    for (let t = 0; t < T; t += Math.max(1, Math.round(60 / (cw + g)))) svg.append(sv("text", { x: padL + t * (cw + g), y: yAx, "font-size": 10, fill: c.muted, text: t }));
    svg.append(sv("text", { x: padL + S.sel * (cw + g), y: yAx + 12, "font-size": 10, fill: c.ink, text: `▲ ${plainTok(tr.tokens[S.sel])}` }));
    card.append(el("div", { style: "overflow-x:auto" }, svg));
    card.append(el("details", {}, el("summary", { text: "Heatmap as a table" }), el("div", { class: "table-scroll" },
      table(["position", "token", ...used.map((s) => slotName(pd, s))], tr.tokens.map((tok, t) => [String(t), showTok(tok), ...used.map((s) => fmt(N[t][s], 2))])))));
  }
  card.append(el("h3", { text: cont ? "What a read of each variable returns at the end" : "What each slot holds at the end" }));
  card.append(slotTable(pd.final_slots, null, cont ? "No variable holds anything." : "All slots are empty."));
}

// persistent buffer
function renderMemory() {
  const card = $("#card-memory"), m = S.state.memory, c = C();
  cardHead(card, null, "Persistent scratchpad", "Saved with the adapter. Switching adapters retains a separate buffer for each.", "persistent");
  if (!m.enabled) { card.append(el("div", { class: "empty", text: "This adapter has no scratchpad." })); return; }
  const empty = !m.slots.length;
  if (!m.cont) {
    card.append(el("h3", { text: "Slot norms" }));
    card.append(slotColumns(m.norms, c.s2, { label: "persistent slot norms" }));
  }
  card.append(el("h3", { text: m.cont ? "What a read of each remembered variable returns" : "What each slot holds" }));
  card.append(slotTable(m.slots, m.log, "Empty. Choose “Scratchpad 1 · Remember” in Examples to write something."));
  card.append(el("div", { style: "margin-top:10px;display:flex;gap:8px;align-items:center" },
    el("button", { class: "ghost", type: "button", text: "Reset persistent scratchpad", disabled: empty ? "disabled" : null, onclick: async () => {
      try { const res = await api("/api/memory", { op: "reset" }); S.state.memory = res.memory; renderMemory(); status("memory-status", "Reset. The persistent scratchpad is empty."); } catch (e) { status("memory-status", e.message, true); }
    } }),
    el("span", { class: "status", id: "memory-status", role: "status" })));
}

// knowledge graph
function forceLayout(n, edges, W, H, prev) {
  const P = [];
  for (let i = 0; i < n; i++) P.push(prev && prev[i] ? { ...prev[i] } : { x: W / 2 + W * 0.33 * Math.cos((2 * Math.PI * i) / n), y: H / 2 + H * 0.36 * Math.sin((2 * Math.PI * i) / n) });
  const k = Math.sqrt((W * H) / Math.max(n, 1)) * 0.62;
  let temp = prev ? W / 40 : W / 8;
  for (let it = 0; it < 400; it++) {
    const D = P.map(() => ({ x: 0, y: 0 }));
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
      const dx = P[i].x - P[j].x, dy = P[i].y - P[j].y, d = Math.max(0.5, Math.hypot(dx, dy)), f = (k * k) / d;
      D[i].x += (dx / d) * f; D[i].y += (dy / d) * f; D[j].x -= (dx / d) * f; D[j].y -= (dy / d) * f;
    }
    for (const [a, b] of edges) {
      if (a === b) continue;
      const dx = P[a].x - P[b].x, dy = P[a].y - P[b].y, d = Math.max(0.5, Math.hypot(dx, dy)), f = (d * d) / k;
      D[a].x -= (dx / d) * f; D[a].y -= (dy / d) * f; D[b].x += (dx / d) * f; D[b].y += (dy / d) * f;
    }
    for (let i = 0; i < n; i++) {
      D[i].x += (W / 2 - P[i].x) * 0.6; D[i].y += (H / 2 - P[i].y) * 0.6;
      const d = Math.max(1e-6, Math.hypot(D[i].x, D[i].y)), m = Math.min(d, temp);
      P[i].x = Math.max(40, Math.min(W - 90, P[i].x + (D[i].x / d) * m));
      P[i].y = Math.max(24, Math.min(H - 24, P[i].y + (D[i].y / d) * m));
    }
    temp *= 0.985;
  }
  return P;
}
function graphHighlight() {
  const r = S.run, g = S.state.graph;
  if (!r || !r.trace.kg || r.graphKey !== graphKey()) return null;
  const k = r.trace.kg[S.sel];
  const heads = new Map(k.heads.filter(([, w]) => w > 0.05));
  const rels1 = new Set(k.hops[0].rels.filter(([, w]) => w > 0.05).map(([i]) => i));
  const t1 = new Map(k.hops[0].tails.filter(([, w]) => w > 0.02));
  const rels2 = k.hops[1] ? new Set(k.hops[1].rels.filter(([, w]) => w > 0.05).map(([i]) => i)) : new Set();
  const t2 = k.hops[1] ? new Map(k.hops[1].tails.filter(([, w]) => w > 0.02)) : new Map();
  const e1 = new Set(), e2 = new Set();
  g.triples.forEach(([h, rr, t], i) => {
    if (heads.has(h) && rels1.has(rr) && t1.has(t)) e1.add(i);
    if (t1.has(h) && rels2.has(rr) && t2.has(t)) e2.add(i);
  });
  const lam = r.trace.next[S.sel].lam ?? 0;
  const answer = k.final.length && k.final[0][1] >= 0.3 && lam >= 0.3 ? k.final[0][0] : null;     // Label only candidates above both thresholds.
  return { heads, t1, t2, e1, e2, hop2: k.halt[1] ?? 0, answer };
}
function renderGraph() {
  const card = $("#card-graph"), g = S.state.graph, c = C();
  cardHead(card, null, "Knowledge graph", `${g.label}: ${g.entities.length} entities, ${g.relations.length} relations, ${g.triples.length} edges. Highlighted: the traversal at the selected position. Edits re-run the last prompt and show the answer before and after.`);
  const n = g.entities.length, W = 800, H = 440;
  const key = graphKey();
  if (S.layoutKey !== key) {
    const prev = S.layout && S.layoutNames ? g.entities.map((e) => { const i = S.layoutNames.indexOf(e); return i >= 0 ? S.layout[i] : null; }) : null;
    S.layout = forceLayout(n, g.triples.map(([h, , t]) => [h, t]), W, H, prev && prev.some(Boolean) ? prev : null);
    S.layoutKey = key; S.layoutNames = [...g.entities];
    if (S.graphFocus !== null && S.graphFocus >= n) S.graphFocus = null;
  }
  // search and neighbourhood focus
  const search = el("input", { type: "search", placeholder: "find an entity", "aria-label": "Find an entity", list: "entity-list", value: S.graphFocus !== null ? g.entities[S.graphFocus] : "" });
  const dl = el("datalist", { id: "entity-list" }, ...g.entities.map((e) => el("option", { value: e })));
  search.addEventListener("change", () => { const i = g.entities.indexOf(search.value.trim()); S.graphFocus = i >= 0 ? i : null; renderGraph(); });
  const nb = el("input", { type: "checkbox", checked: S.graphNeighbourhood ? "checked" : null });
  nb.addEventListener("change", () => { S.graphNeighbourhood = nb.checked; renderGraph(); });
  card.append(el("div", { class: "graph-tools" }, search, dl, el("label", { class: "check" }, nb, document.createTextNode("show only its neighbourhood")),
    S.graphFocus !== null ? el("button", { class: "ghost small", type: "button", text: "clear", onclick: () => { S.graphFocus = null; renderGraph(); } }) : null));
  card.append(legend([{ label: "mention head", color: c.ink, kind: "dot" }, { label: "hop 1", color: c.s1, kind: "line" }, { label: "hop 2", color: c.s1d, kind: "line" }]));
  const P = S.layout, hl = graphHighlight();
  const near = new Set();
  if (S.graphFocus !== null) { near.add(S.graphFocus); g.triples.forEach(([h, , t]) => { if (h === S.graphFocus) near.add(t); if (t === S.graphFocus) near.add(h); }); }
  const hide = (i) => S.graphNeighbourhood && S.graphFocus !== null && !near.has(i);
  const pathDesc = hl ? [...hl.e1, ...hl.e2].map((i) => { const [h, rr, t] = g.triples[i]; return `${g.entities[h]} ${g.relations[rr]} ${g.entities[t]}`; }).join("; ") : "";
  const svg = sv("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": `Knowledge graph with ${n} entities. ${pathDesc ? `Traversal at the selected position: ${pathDesc}.` : "No traversal highlighted."} The edge table below lists every edge.` });
  const defs = sv("defs");
  for (const [id, col] of [["arr0", c.edge], ["arr1", c.s1], ["arr2", c.s1d]]) {
    const m = sv("marker", { id, viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: "auto-start-reverse" });
    m.append(sv("path", { d: "M0,1L9,5L0,9z", fill: col })); defs.append(m);
  }
  svg.append(defs);
  const pairCount = new Map(), layers = [[], [], []];
  g.triples.forEach(([h, rr, t], i) => {
    if (hide(h) || hide(t)) return;
    const pk = `${h},${t}`, idx = pairCount.get(pk) || 0; pairCount.set(pk, idx + 1);
    const a = P[h], b = P[t]; if (!a || !b) return;
    const dx = b.x - a.x, dy = b.y - a.y, d = Math.max(1, Math.hypot(dx, dy)), ux = dx / d, uy = dy / d;
    const x1 = a.x + ux * 8, y1 = a.y + uy * 8, x2 = b.x - ux * 9, y2 = b.y - uy * 9;
    const bend = 16 + 10 * idx, mx = (x1 + x2) / 2 - uy * bend, my = (y1 + y2) / 2 + ux * bend;
    const lvl = hl && hl.e2.has(i) ? 2 : hl && hl.e1.has(i) ? 1 : 0;
    const col = [c.edge, c.s1, c.s1d][lvl], path = `M${x1},${y1} Q${mx},${my} ${x2},${y2}`;
    const grp = sv("g");
    grp.append(sv("path", { d: path, fill: "none", stroke: col, "stroke-width": lvl ? 2 : 1, "marker-end": `url(#arr${lvl})`, opacity: lvl === 2 && hl.hop2 < 0.05 ? 0.45 : 1 }));
    const hit = sv("path", { d: path, fill: "none", stroke: "transparent", "stroke-width": 10 });
    hover(hit, () => [{ v: g.relations[rr], k: `${g.entities[h]} → ${g.entities[t]}`, color: lvl ? col : undefined }]);
    grp.append(hit);
    if (lvl) grp.append(sv("text", { x: (x1 + x2) / 2 - uy * (bend * 0.55), y: (y1 + y2) / 2 + ux * (bend * 0.55), "text-anchor": "middle", "font-size": 10, fill: c.ink2, stroke: c.surface, "stroke-width": 3, "paint-order": "stroke", text: g.relations[rr] }));
    layers[lvl].push(grp);
  });
  for (const L of layers) for (const x of L) svg.append(x);
  const dense = n > 25;
  g.entities.forEach((name, i) => {
    const p = P[i]; if (!p || hide(i)) return;
    const isHead = hl && hl.heads.has(i), w1 = hl && hl.t1.get(i), w2 = hl && hl.t2.get(i);
    const hot = isHead || w1 || w2 || i === S.graphFocus;
    const grp = sv("g", { style: "cursor:pointer" });
    grp.append(sv("circle", { cx: p.x, cy: p.y, r: 12, fill: "transparent" }));
    grp.append(sv("circle", { cx: p.x, cy: p.y, r: w1 || w2 ? 4 + 4 * Math.max(w1 || 0, w2 || 0) : 4.5, fill: w2 ? c.s1d : w1 ? c.s1 : c.edge, stroke: isHead || i === S.graphFocus ? c.ink : c.surface, "stroke-width": 2 }));
    if (!dense || hot || near.has(i)) grp.append(sv("text", { x: p.x + 9, y: p.y - 7, class: "node-label" + (hot ? " hot" : ""), text: name + (hl && hl.answer === i ? "  ← answer" : "") }));
    const outs = g.triples.filter(([h]) => h === i).map(([, rr, t]) => `${g.relations[rr]} → ${g.entities[t]}`);
    hover(grp, () => [{ v: name, k: [isHead ? `head ${fmt(hl.heads.get(i), 2)}` : "", w1 ? `hop 1 ${fmt(w1, 2)}` : "", w2 ? `hop 2 ${fmt(w2, 2)}` : "", "click to focus"].filter(Boolean).join(" · ") }, ...outs.slice(0, 8).map((o) => ({ k: o }))]);
    grp.addEventListener("click", () => { S.graphFocus = S.graphFocus === i ? null : i; renderGraph(); });
    svg.append(grp);
  });
  card.append(el("div", { class: "graph-wrap" }, svg));
  if (S.run && S.run.trace.kg && S.run.graphKey !== key) card.append(el("div", { class: "notice warn", text: "The graph changed since the last run; run again to see its effect." }));
  card.append(graphEditor());
  card.append(el("div", { class: "status", id: "graph-status", role: "status" }));
  const comp = Object.entries(g.computed || {});
  if (comp.length) card.append(el("details", {}, el("summary", { text: `${comp.length} relations are C programs run on strandedcpu` }),
    table(["relation", "C (x is the head, y the tail)"], comp.map(([r, src]) => [r, src]))));
  card.append(el("details", {}, el("summary", { text: `All ${g.triples.length} edges (table)` }),
    el("div", { class: "table-scroll" }, table(["head", "relation", "tail"], g.triples.map(([h, rr, t]) => [g.entities[h], g.relations[rr], g.entities[t]])))));
}
async function graphOp(body, { rerun = true } = {}) {
  try {
    const before = S.run && S.run.req && S.run.req.mode === "ask" ? S.run.answer : null;
    const res = await api("/api/graph", body);
    S.state.graph = res.graph; renderPresets(); renderGraph();
    const what = { demo: "Loaded the demo graph.", strandedcpu: "Loaded the strandedcpu graph.", random: "Loaded a random world.", set: "Edge set.", add: "Edge added.", remove: "Edge removed.", add_entity: "Entity added.", add_relation: "Relation added." }[body.op];
    status("graph-status", what);
    if (rerun && S.run && S.run.req) {
      await run(S.run.req, { keepEdit: true });
      if (before !== null) S.edit = { before, after: S.run.answer, note: what };
      renderAnswers(); status("graph-status", `${what} Re-ran the last prompt: see the answer before and after at the top.`);
    }
  } catch (e) { status("graph-status", e.message, true); }
}
function graphEditor() {
  const g = S.state.graph;
  const opts = (names) => names.map((n) => el("option", { value: n, text: n }));
  const h = el("select", { "aria-label": "head entity" }, ...opts(g.entities));
  const r = el("select", { "aria-label": "relation" }, ...opts(g.relations));
  const t = el("select", { "aria-label": "tail entity" }, ...opts(g.entities));
  const seed = el("input", { type: "number", value: "4242", "aria-label": "random world seed", style: "width:72px" });
  const names = el("select", { "aria-label": "name pool" }, el("option", { value: "eval", text: "held-out names" }), el("option", { value: "train", text: "training names" }));
  const newName = el("input", { type: "text", placeholder: "new name", "aria-label": "new entity or relation name", style: "width:120px" });
  const btn = (text, fn) => el("button", { class: "ghost", type: "button", text, onclick: fn });
  return el("div", { class: "editor" },
    el("span", { class: "group" }, btn("Demo graph", () => graphOp({ op: "demo" })), btn("strandedcpu arithmetic", () => graphOp({ op: "strandedcpu" })), el("span", { class: "note", text: "random world" }), seed, names, btn("Load", () => graphOp({ op: "random", seed: +seed.value, names: names.value }))),
    el("span", { class: "group" }, h, r, t, btn("Set", () => graphOp({ op: "set", h: h.value, r: r.value, t: t.value })), btn("Add", () => graphOp({ op: "add", h: h.value, r: r.value, t: t.value })), btn("Remove", () => graphOp({ op: "remove", h: h.value, r: r.value, t: t.value }))),
    el("span", { class: "group" }, newName, btn("Add entity", () => graphOp({ op: "add_entity", name: newName.value })), btn("Add relation", () => graphOp({ op: "add_relation", name: newName.value }))));
}

// model / architecture
const DESCR = {
  norm: "RMSNorm on the hidden state coming out of the attach layer",
  kg: "knowledge-graph retrieval head", "kg.ent": "MentionLinker: per-token entity links + mention attention",
  "kg.rel": "one relation linker per hop", "kg.halt": "halting softmax over hop counts", "kg.vnorm": "RMSNorm on candidate entity keys",
  "kg.val": "candidate keys → value vector", pad: "DeltaScratchpad: delta-rule slot memory", "pad.ptr_w": "pointer attention: where write addresses come from",
  "pad.ptr_r": "pointer attention: where read addresses come from", "pad.addr": "shared address map", "pad.beta": "write strength β",
  "pad.v": "value to write", "pad.v_r": "read-modify-write path", "pad.alpha": "decay gate", out_kg: "graph value → residual stream (zero-initialised)",
  out_pad: "scratchpad read → residual stream (zero-initialised)", pad_readout: "scratchpad read → logits through the frozen LM head",
  read_norm: "RMSNorm on reads (scaled down for near-empty reads)", gate: "per-position gates for the two memories", copy_gate: "copy head gate λ (pointer vs LM)",
};
function renderArch() {
  const card = $("#card-arch"), a = S.state.arch, c = C();
  cardHead(card, null, "Model", `${a.model}, frozen (${(a.frozen_params / 1e6).toFixed(1)}M params). One trainable block (${(a.block.params / 1e6).toFixed(2)}M) sits after decoder layer ${a.layer_idx}; the copy head mixes at the logits.`);
  const n = a.n_layers, rowH = 10, g = 4, W = 120, padT = 4;
  const rowsAll = [{ label: "token embeddings", kind: "frozen" }];
  for (let i = 0; i < n; i++) {
    rowsAll.push({ label: `layer ${i}${a.layer_types[i] && !String(a.layer_types[i]).includes("full") ? " · sliding" : ""}`, kind: "frozen", i });
    if (i === a.layer_idx) rowsAll.push({ label: `MemoryBlock · ${(a.block.params / 1e6).toFixed(2)}M trainable`, kind: "block" });
  }
  rowsAll.push({ label: "LM head + scratchpad readout", kind: "frozen" }, { label: "copy head: (1−λ)·p_LM + λ·p_ptr", kind: "block" });
  const H = padT + rowsAll.length * (rowH + g) + 4;
  const svg = sv("svg", { width: W + 200, height: H, role: "img", "aria-label": `Layer stack: ${n} frozen layers with the memory block after layer ${a.layer_idx}` });
  rowsAll.forEach((r, j) => {
    const y = padT + j * (rowH + g), isB = r.kind === "block";
    svg.append(sv("rect", { x: 0, y, width: isB ? W : W * 0.82, height: rowH, rx: 2, fill: isB ? c.s1 : c.edge }));
    if (isB || r.i === undefined || r.i % 5 === 0 || r.i === n - 1 || r.i === a.layer_idx) svg.append(sv("text", { x: W + 8, y: y + 9, "font-size": 10, fill: isB ? c.ink : c.ink2, "font-weight": isB ? 600 : 400, text: r.label }));
  });
  const rowsData = [];
  for (const ch of a.block.children) {
    rowsData.push([ch.name, ch.params.toLocaleString(), DESCR[ch.name] || ch.type]);
    for (const gg of ch.children) rowsData.push([`  ${ch.name}.${gg.name}`, gg.params.toLocaleString(), DESCR[`${ch.name}.${gg.name}`] || gg.type]);
  }
  card.append(el("div", { class: "grid2" },
    el("div", {}, el("h3", { text: "Where the block sits" }), svg,
      el("p", { class: "note", text: `Backward runs only through layers ${a.layer_idx + 1}–${n - 1}. Layer types: ${[...new Set(a.layer_types)].join(", ")}.` })),
    el("div", {}, el("h3", { text: "Trainable components" }), table(["component", "params", "role"], rowsData),
      el("h3", { text: "Config" }), table(["key", "value"], Object.entries(a.cfg).map(([k, v]) => [k, String(v)])),
      el("h3", { text: "Adapter provenance" }),
      a.meta ? el("pre", { class: "meta", text: JSON.stringify(a.meta, null, 1) }) : el("p", { class: "note", text: `${a.adapter.split("/").pop()} has no training metadata.` }))));
}

// walkthrough
function buildTour(r) {
  const tr = r.trace, steps = [];
  const P = (v) => fmt(v, 2);
  if (r.mode === "remember") {
    tr.lanes.beta.forEach((b, t) => {
      if (b <= 0.5 || steps.length >= 4) return;
      const src = tr.pad.ptr_w[t][0], s = argmax(tr.pad.kw[t]);
      steps.push({ t, card: "card-pad-pos", text: `${q(tr.tokens[t])} is written with β = ${P(b)}. Its address comes from ${src ? q(tr.tokens[src[0]]) : "this token"}, the variable's name, ` +
        (tr.pad.cont ? "as a continuous key vector. Read/write key overlap is shown below." : `so it lands mostly in slot ${s} (weight ${P(tr.pad.kw[t][s])}).`) });
    });
    steps.push({ t: S.sel, card: "card-memory", text: "Writes committed to the saved scratchpad. Choose “Scratchpad 2 · Ask in a new session” to query that buffer without repeating the assignments." });
    return steps;
  }
  const t0 = r.mode === "ask" ? r.n_prompt - 1 : defaultSel(r), nx = tr.next[t0], k = tr.kg && tr.kg[t0];
  const actual = nx.actual !== null && nx.actual !== undefined ? tr.tokens[t0 + 1] : null;
  const base = [...nx.cands].sort((a, b) => b[1] - a[1])[0];
  steps.push({ t: t0, card: "card-timeline", text: `${r.mode === "ask" ? "The final prompt position predicts the first answer token." : "This position predicts the next token."} With the memory off, the model would continue with ${q(base[0])} (${P(base[1])}). With it, the next token is ${q(actual ?? "")} at ${P(tr.p_actual[t0])}, versus ${P(tr.p_actual_base[t0])} without memory.` });
  const lam = nx.lam ?? 0, ro = nx.readout ? nx.readout[0] : null;
  const graphAnswer = k && lam >= 0.3 && k.final.length && k.heads.length;
  if (graphAnswer) {
    const m = (tr.mention_att[t0] || [])[0] || [t0, 0];
    steps.push({ t: t0, card: "card-link", text: `Mention attention points at ${q(tr.tokens[m[0]])} @${m[0]} (${P(m[1])}). Its top head candidate is ${ent(k.heads[0][0])} (${P(k.heads[0][1])}).` });
    const h1 = k.hops[0], h2 = k.hops[1];
    const two = (k.halt[1] ?? 0) > 0.5;
    steps.push({ t: t0, card: "card-traversal", text: `Hop 1 follows “${rel(h1.rels[0][0])}” (${P(h1.rels[0][1])}) to ${h1.tails.length ? ent(h1.tails[0][0]) : "nothing"}.` +
      (two && h2 && h2.tails.length ? ` Hop 2 follows “${rel(h2.rels[0][0])}” to ${ent(h2.tails[0][0])}, and halting puts ${P(k.halt[1])} on the two-hop answer.` : ` Halting puts ${P(k.halt[0])} on stopping after one hop.`) +
      ` The top candidate is ${ent(k.final[0][0])}.` });
    steps.push({ t: t0, card: "card-next", text: `Copy gate λ = ${P(lam)}; top graph candidate: ${ent(k.final[0][0])}. The next token in this sequence is ${q(actual ?? "")}. After the hidden-state injection, the upper layers gave that token ${P(nx.cands.find((x) => x[5] === nx.actual)?.[2] ?? 0)}.` });
    if (r.mode === "ask" && r.n_prompt < tr.T - 1)
      steps.push({ t: t0 + 1, card: "card-next", text: `The next position can continue the same candidate name${tr.next[t0 + 1].carry ? " (this position is carrying it)" : ""}. Completion and line-ending behavior depend on the adapter configuration.` });
  } else if (tr.pad && ro && ro[1] > 2) {
    const src = tr.pad.ptr_r[t0][0], s = argmax(tr.pad.kr[t0]), ov = overlaps(t0)[0];
    steps.push({ t: t0, card: "card-pad-pos", text: `The scratchpad read address comes from ${src ? q(tr.tokens[src[0]]) : "this token"}` +
      (tr.pad.cont ? "." : ` and lands mostly in slot ${s} (${P(tr.pad.kr[t0][s])}).`) + (ov ? ` The write it overlaps most is ${ov.name} (${P(ov.v)}).` : "") });
    steps.push({ t: t0, card: "card-next", text: `${q(ro[0])} receives a scratchpad logit bias of +${fmt(ro[1], 1)}. Copy gate λ = ${fmt(lam, 3)}.` + (r.no_buffer_answer !== undefined ? ` Without the remembered state the answer would be ${q(r.no_buffer_answer)}.` : "") });
  } else {
    steps.push({ t: t0, card: "card-next", text: `Copy gate λ = ${fmt(lam, 3)}${ro ? `; strongest scratchpad bias +${fmt(ro[1], 1)}` : ""}. Compare the prediction stages for the hidden-state injection effect.` });
  }
  return steps;
}
function startTour() {
  if (!S.run) return;
  S.tour = { steps: buildTour(S.run), i: 0 };
  tourStep();
}
function tourStep() {
  const box = $("#tour"), T = S.tour;
  if (!T) return;
  const st = T.steps[T.i];
  select(st.t, true);
  box.replaceChildren(el("div", { class: "note", text: "Trace steps" }), el("p", { class: "text", text: st.text }),
    el("div", { class: "bar" },
      el("button", { class: "ghost", type: "button", text: "Back", disabled: T.i === 0 ? "disabled" : null, onclick: () => { T.i--; tourStep(); } }),
      T.i < T.steps.length - 1 ? el("button", { class: "primary", type: "button", text: "Next", onclick: () => { T.i++; tourStep(); } }) : el("button", { class: "primary", type: "button", text: "Done", onclick: endTour }),
      el("button", { class: "ghost", type: "button", text: "End", onclick: endTour }),
      el("span", { class: "count", text: `${T.i + 1} / ${T.steps.length}` })));
  box.classList.add("on");
  const target = document.getElementById(st.card);
  if (target) { target.classList.add("spot"); setTimeout(() => target.classList.remove("spot"), 1800); target.scrollIntoView({ block: "center", behavior: "auto" }); }
}
function endTour() { S.tour = null; $("#tour").classList.remove("on"); $("#tour").replaceChildren(); }

// boot
function renderAll() {
  if (!S.state) return;
  renderAnswers(); renderTimeline(); renderSummary(); renderLink(); renderTraversal(); renderPadPos(); renderNext(); renderPad();
  renderGraph(); renderMemory(); renderArch(); renderDiagram();
}

(async function init() {
  setupControls(); setMode("ask"); renderGlossary();
  // #preset=N runs that example on load, #theme=dark|light picks the theme, #tour=1 starts the walkthrough
  const h = new URLSearchParams(location.hash.slice(1));
  if (h.get("theme")) document.documentElement.dataset.theme = h.get("theme");
  try {
    S.state = await api("/api/state");
    renderStatic();
    const p = PRESETS[+h.get("preset")];
    if (p && h.get("preset") !== null) await applyPreset(p); else await run();
    if (h.get("tour")) startTour();
  } catch (e) { status("status", e.message, true); }
})();
