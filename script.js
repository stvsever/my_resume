/* =========================================================
   Stijn Van Severen - interaction + brain-network field
   ========================================================= */
document.documentElement.classList.add("js-enabled");
const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

/* =========================================================
   BRAIN FIELD
   Nodes are assigned to the 7 canonical functional networks
   (Yeo 2011) and laid out over a bilateral, anatomically
   plausible map. Edges and signal flow follow a between-
   network connectivity matrix, so the activity reads as a
   real connectome instead of random clusters.
   ========================================================= */
const BrainField = (() => {
  const canvas = document.getElementById("neural-field");
  if (!canvas) return null;
  const ctx = canvas.getContext("2d");

  // Yeo-7 networks: colour + anatomically plausible blobs in one
  // hemisphere (x: 0 lateral -> 0.5 midline ; y: 0 anterior -> 1 posterior)
  const NETS = [
    { col: [150, 78, 188], blobs: [[0.30, 0.90, 0.11], [0.42, 0.86, 0.08], [0.19, 0.85, 0.08]] }, // 0 Visual
    { col: [74, 142, 205], blobs: [[0.17, 0.46, 0.08], [0.28, 0.42, 0.07], [0.40, 0.40, 0.06]] },  // 1 Somatomotor
    { col: [56, 188, 110], blobs: [[0.22, 0.64, 0.07], [0.34, 0.60, 0.06], [0.37, 0.30, 0.05]] },  // 2 Dorsal Attention
    { col: [202, 96, 232], blobs: [[0.12, 0.52, 0.06], [0.44, 0.36, 0.05]] },                       // 3 Ventral Attention
    { col: [226, 206, 132], blobs: [[0.13, 0.36, 0.06], [0.40, 0.18, 0.05]] },                      // 4 Limbic
    { col: [236, 154, 64], blobs: [[0.18, 0.26, 0.06], [0.30, 0.24, 0.05], [0.26, 0.66, 0.06]] },   // 5 Frontoparietal
    { col: [220, 86, 100], blobs: [[0.44, 0.20, 0.06], [0.44, 0.74, 0.06], [0.30, 0.62, 0.05], [0.10, 0.46, 0.05]] }, // 6 Default
  ];
  // between-network functional connectivity weights (symmetric)
  const W = [
    [1.0, .30, .42, .20, .12, .18, .12],
    [.30, 1.0, .38, .36, .15, .20, .14],
    [.42, .38, 1.0, .40, .14, .48, .12],
    [.20, .36, .40, 1.0, .22, .50, .20],
    [.12, .15, .14, .22, 1.0, .20, .34],
    [.18, .20, .48, .50, .20, 1.0, .52],
    [.12, .14, .12, .20, .34, .52, 1.0],
  ];

  let W_ = 0, H = 0, dpr = 1;
  let nodes = [], pulses = [], raf = null;
  let nextEmit = 0, scrollY = 0, startTime = 0;
  const SPAWN_MS = 1600;
  const pointer = { x: -9999, y: -9999, active: false };
  const clamp = (v, a, b) => (v < a ? a : v > b ? b : v);
  const rnd = () => Math.random();
  const gauss = () => (rnd() + rnd() + rnd() - 1.5) / 1.5; // approx normal in [-1,1]

  function nodeCount() {
    const w = window.innerWidth;
    if (w < 620) return 180;
    if (w < 980) return 320;
    if (w < 1400) return 460;
    return 560;
  }
  function maxLinkDist() {
    const w = window.innerWidth;
    if (w < 620) return 96;
    if (w < 1400) return 112;
    return 122;
  }

  function toScreen(nx, ny) {
    return [ (0.05 + nx * 0.90) * W_, (0.08 + ny * 0.84) * H ];
  }

  function seed() {
    const count = nodeCount();
    // budget per network proportional to number of blobs
    let totalBlobs = 0;
    NETS.forEach((n) => (totalBlobs += n.blobs.length));
    nodes = [];
    let id = 0;
    for (let ni = 0; ni < NETS.length; ni++) {
      const net = NETS[ni];
      const budget = Math.round(count * (net.blobs.length / totalBlobs));
      for (let k = 0; k < budget; k++) {
        const blob = net.blobs[(rnd() * net.blobs.length) | 0];
        let nx = clamp(blob[0] + gauss() * blob[2] * 0.85, 0.02, 0.48);
        const ny = clamp(blob[1] + gauss() * blob[2] * 0.85, 0.02, 0.98);
        if (id % 2 === 0) nx = 1 - nx; // mirror to the other hemisphere
        const [x, y] = toScreen(nx, ny);
        nodes.push({
          x, y, ox: x, oy: y, vx: 0, vy: 0,
          size: 0.7 + rnd() * 1.7, net: ni, col: net.col,
          phase: rnd() * Math.PI * 2,
          wx: 0.00022 + rnd() * 0.0004, wy: 0.00018 + rnd() * 0.00035,
          amp: 0.09 + rnd() * 0.13, signal: rnd() * 0.25,
          depth: 0.4 + rnd() * 1.1, refractoryUntil: 0,
        });
        id++;
      }
    }
  }

  function resize() {
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    W_ = window.innerWidth; H = window.innerHeight;
    canvas.width = Math.floor(W_ * dpr); canvas.height = Math.floor(H * dpr);
    canvas.style.width = W_ + "px"; canvas.style.height = H + "px";
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    seed();
  }

  /* spatial grid */
  let grid = new Map(), cell = 110;
  function buildGrid() {
    grid = new Map(); cell = maxLinkDist();
    for (let i = 0; i < nodes.length; i++) {
      const n = nodes[i];
      const key = ((n.x / cell) | 0) + "," + ((n.y / cell) | 0);
      let b = grid.get(key); if (!b) { b = []; grid.set(key, b); } b.push(i);
    }
  }
  function around(x, y, range) {
    const gx = (x / cell) | 0, gy = (y / cell) | 0, span = Math.ceil(range / cell), out = [];
    for (let ix = gx - span; ix <= gx + span; ix++) for (let iy = gy - span; iy <= gy + span; iy++) {
      const b = grid.get(ix + "," + iy); if (b) for (const j of b) out.push(j);
    }
    return out;
  }

  /* signal emission follows the connectivity weights */
  function emit(node, energy, now) {
    if (reduceMotion) return;
    node.signal = Math.min(1.7, Math.max(node.signal, 1.25));
    node.refractoryUntil = now + 720;
    if (energy <= 0 || pulses.length > 80) return;
    const range = 165, cand = [];
    for (const j of around(node.x, node.y, range)) {
      const o = nodes[j]; if (o === node) continue;
      const dx = o.x - node.x, dy = o.y - node.y, d2 = dx * dx + dy * dy;
      if (d2 < range * range && d2 > 120) {
        const w = W[node.net][o.net];
        if (w < 0.16) continue;
        cand.push({ o, score: w / (Math.sqrt(d2) + 30), d: Math.sqrt(d2) });
      }
    }
    cand.sort((a, b) => b.score - a.score);
    const fan = Math.min(cand.length, 2 + ((rnd() * 2) | 0));
    for (let i = 0; i < fan; i++) {
      pulses.push({ a: node, b: cand[i].o, t: 0, v: 1.8 / Math.max(cand[i].d, 28), energy: energy - 1, col: node.col, w: W[node.net][cand[i].o.net] });
    }
  }
  function scheduleEmit(now) {
    if (reduceMotion || !nodes.length) return;
    if (!nextEmit) { nextEmit = now + 800 + rnd() * 1300; return; }
    if (now < nextEmit) return;
    emit(nodes[(rnd() * nodes.length) | 0], 1 + (rnd() < 0.45 ? 1 : 0), now);
    nextEmit = now + 1400 + rnd() * 2200;
  }

  function updateNode(n, t) {
    const osc = Math.sin(t * n.wx + n.phase) * n.amp + Math.cos(t * n.wy + n.phase * 1.6) * n.amp * 0.6;
    n.signal += ((0.15 + Math.abs(osc)) - n.signal) * 0.04;
    n.signal *= 0.99; if (n.signal > 1.7) n.signal = 1.7;
    const dx = Math.cos(t * n.wx + n.phase) * 0.16, dy = Math.sin(t * n.wy + n.phase) * 0.16;
    const parY = scrollY * 0.015 * n.depth;
    n.vx += (n.ox - n.x) * 0.0013 + dx * 0.01;
    n.vy += (n.oy + parY - n.y) * 0.0013 + dy * 0.01;
    if (pointer.active) {
      const ax = n.x - pointer.x, ay = n.y - pointer.y, d2 = ax * ax + ay * ay;
      if (d2 < 22000) { const d = Math.sqrt(d2) || 1, f = (1 - d / 148) * 0.55; n.vx += (ax / d) * f; n.vy += (ay / d) * f; }
    }
    n.vx *= 0.9; n.vy *= 0.9; n.x += n.vx; n.y += n.vy;
  }

  function drawLinks(maxDist) {
    const md2 = maxDist * maxDist;
    for (let i = 0; i < nodes.length; i++) {
      const a = nodes[i];
      const gx = (a.x / cell) | 0, gy = (a.y / cell) | 0;
      for (let ix = gx - 1; ix <= gx + 1; ix++) for (let iy = gy - 1; iy <= gy + 1; iy++) {
        const b = grid.get(ix + "," + iy); if (!b) continue;
        for (const j of b) {
          if (j <= i) continue;
          const o = nodes[j], ax = a.x - o.x, ay = a.y - o.y, d2 = ax * ax + ay * ay;
          if (d2 > md2) continue;
          const w = W[a.net][o.net];
          if (w < 0.16) continue;
          const d = Math.sqrt(d2);
          const k = 0.011 * (1 - d / maxDist) * w;
          const flow = (o.signal - a.signal) * k; a.signal += flow; o.signal -= flow;
          const s = Math.min(1.2, (a.signal + o.signal) * 0.5);
          const alpha = (1 - d / maxDist) * (0.04 + w * 0.11 + s * 0.08);
          let c;
          if (a.net === o.net) c = a.col;
          else c = [(a.col[0] + o.col[0]) / 2, (a.col[1] + o.col[1]) / 2, (a.col[2] + o.col[2]) / 2];
          ctx.strokeStyle = `rgba(${c[0] | 0},${c[1] | 0},${c[2] | 0},${alpha})`;
          ctx.lineWidth = 0.5 + w * 0.6;
          ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(o.x, o.y); ctx.stroke();
        }
      }
    }
  }

  function drawPulses(now) {
    pulses = pulses.filter((p) => p.t < 1);
    for (const p of pulses) {
      p.t += p.v;
      const x = p.a.x + (p.b.x - p.a.x) * p.t, y = p.a.y + (p.b.y - p.a.y) * p.t, c = p.col;
      ctx.beginPath(); ctx.arc(x, y, 2.2, 0, 6.2832); ctx.fillStyle = `rgba(${c[0]},${c[1]},${c[2]},0.92)`; ctx.fill();
      ctx.beginPath(); ctx.arc(x, y, 6.5, 0, 6.2832); ctx.fillStyle = `rgba(${c[0]},${c[1]},${c[2]},0.1)`; ctx.fill();
      if (p.t >= 1) {
        p.b.signal = Math.min(1.7, p.b.signal + 0.4);
        if (p.energy > 0 && now > p.b.refractoryUntil && rnd() < clamp(p.w * 0.6, 0.1, 0.6)) emit(p.b, p.energy, now);
      }
    }
  }

  function drawNodes() {
    for (const n of nodes) {
      const r = n.size * (1 + n.signal * 0.5), c = n.col;
      ctx.beginPath(); ctx.arc(n.x, n.y, r * 3.8, 0, 6.2832); ctx.fillStyle = `rgba(${c[0]},${c[1]},${c[2]},${0.022 + n.signal * 0.04})`; ctx.fill();
      ctx.beginPath(); ctx.arc(n.x, n.y, r, 0, 6.2832); ctx.fillStyle = `rgba(${c[0]},${c[1]},${c[2]},${0.42 + n.signal * 0.34})`; ctx.fill();
    }
  }

  function drawPointer() {
    if (!pointer.active) return;
    for (const j of around(pointer.x, pointer.y, 150)) {
      const n = nodes[j], dx = n.x - pointer.x, dy = n.y - pointer.y, d = Math.sqrt(dx * dx + dy * dy);
      if (d > 150) continue;
      ctx.beginPath(); ctx.moveTo(pointer.x, pointer.y); ctx.lineTo(n.x, n.y);
      ctx.strokeStyle = `rgba(124,255,191,${(1 - d / 150) * 0.4})`; ctx.lineWidth = 0.8; ctx.stroke();
    }
  }

  function frame(t) {
    if (!startTime) startTime = t;
    const el = t - startTime;
    ctx.clearRect(0, 0, W_, H);
    buildGrid();
    if (el < SPAWN_MS && !reduceMotion) {
      // settle from a contracted-but-distributed layout (keeps the grid sparse)
      const e = 1 - Math.pow(1 - el / SPAWN_MS, 3), cx = W_ * 0.5, cy = H * 0.46, k = 0.5 + 0.5 * e;
      for (const n of nodes) { n.x = cx + (n.ox - cx) * k; n.y = cy + (n.oy - cy) * k; n.signal = 0.18 + e * 0.32; }
    } else {
      scheduleEmit(t);
      for (const n of nodes) updateNode(n, t);
    }
    drawLinks(maxLinkDist());
    drawPulses(t);
    drawNodes();
    drawPointer();
    if (!reduceMotion) raf = requestAnimationFrame(frame);
  }

  function nearestEmit(x, y, energy) {
    let best = Infinity, idx = -1;
    for (const j of around(x, y, 220)) { const n = nodes[j], d = (n.x - x) ** 2 + (n.y - y) ** 2; if (d < best) { best = d; idx = j; } }
    if (idx >= 0) emit(nodes[idx], energy, performance.now());
  }

  function start() {
    resize();
    window.addEventListener("resize", () => { if (raf) cancelAnimationFrame(raf); startTime = 0; resize(); if (reduceMotion) frame(0); else raf = requestAnimationFrame(frame); });
    window.addEventListener("pointermove", (e) => { pointer.x = e.clientX; pointer.y = e.clientY; pointer.active = true; }, { passive: true });
    window.addEventListener("pointerleave", () => { pointer.active = false; });
    window.addEventListener("pointerdown", (e) => { buildGrid(); nearestEmit(e.clientX, e.clientY, 2); }, { passive: true });
    window.addEventListener("scroll", () => { scrollY = window.scrollY; }, { passive: true });
    if (reduceMotion) frame(0); else raf = requestAnimationFrame(frame);
  }
  return { start, ignite: (x, y) => { buildGrid(); nearestEmit(x, y, 2); } };
})();

/* =========================================================
   HERO PORTRAIT: pointer tilt + scroll recede
   ========================================================= */
function setupHeroCore() {
  const core = document.querySelector("[data-core]");
  const hero = document.querySelector(".hero");
  if (!core || reduceMotion) return;
  let px = 0, py = 0, ticking = false;
  const cl = (v, a, b) => (v < a ? a : v > b ? b : v);
  function schedule() { if (!ticking) { ticking = true; requestAnimationFrame(apply); } }
  function apply() {
    ticking = false;
    const prog = hero ? cl(window.scrollY / Math.max(hero.offsetHeight, 1), 0, 1) : 0;
    core.style.transform = `translateY(${prog * -30}px) perspective(1100px) rotateX(${-py * 4}deg) rotateY(${px * 5}deg) scale(${1 - prog * 0.12})`;
    core.style.opacity = String(1 - prog * 0.7);
  }
  window.addEventListener("pointermove", (e) => { px = e.clientX / window.innerWidth - 0.5; py = e.clientY / window.innerHeight - 0.5; schedule(); }, { passive: true });
  window.addEventListener("scroll", schedule, { passive: true });
  apply();
}

/* =========================================================
   FOCUS ROTATOR
   ========================================================= */
function setupRotator() {
  const rot = document.querySelector("[data-rotator]");
  if (!rot) return;
  const items = [...rot.children];
  if (items.length < 2) return;
  let i = 0;
  setInterval(() => {
    items[i].classList.remove("is-active"); items[i].classList.add("is-out");
    const prev = i; i = (i + 1) % items.length;
    items[i].classList.remove("is-out"); items[i].classList.add("is-active");
    setTimeout(() => items[prev].classList.remove("is-out"), 500);
  }, 2600);
}

/* =========================================================
   NAV + SCROLL PROGRESS
   ========================================================= */
function setupNav() {
  const nav = document.querySelector("[data-nav]");
  const progress = document.querySelector("[data-progress]");
  const links = [...document.querySelectorAll("[data-navlink]")];
  let lastY = window.scrollY;
  function onScroll() {
    const y = window.scrollY, docH = document.documentElement.scrollHeight - window.innerHeight;
    if (progress) progress.style.width = (docH > 0 ? (y / docH) * 100 : 0) + "%";
    if (nav) { nav.classList.toggle("is-scrolled", y > 20); if (y > lastY && y > 200) nav.classList.add("is-hidden"); else nav.classList.remove("is-hidden"); }
    lastY = y;
  }
  window.addEventListener("scroll", onScroll, { passive: true }); onScroll();
  const sections = links.map((l) => document.querySelector(l.getAttribute("href"))).filter(Boolean);
  if (sections.length) {
    const obs = new IntersectionObserver((entries) => entries.forEach((e) => { if (e.isIntersecting) links.forEach((l) => l.classList.toggle("is-active", l.getAttribute("href") === "#" + e.target.id)); }), { rootMargin: "-40% 0px -55% 0px" });
    sections.forEach((s) => obs.observe(s));
  }
}

/* =========================================================
   REVEAL
   ========================================================= */
function setupReveal() {
  const items = document.querySelectorAll("[data-reveal]");
  const obs = new IntersectionObserver((entries) => entries.forEach((e) => { if (e.isIntersecting) { e.target.classList.add("is-visible"); obs.unobserve(e.target); } }), { threshold: 0.12 });
  items.forEach((it) => obs.observe(it));
  setTimeout(() => items.forEach((it) => { const r = it.getBoundingClientRect(); if (r.top < window.innerHeight && r.bottom > 0) it.classList.add("is-visible"); }), 120);
}

/* =========================================================
   PROFILE MATRIX
   ========================================================= */
function setupMatrix() {
  const tabs = [...document.querySelectorAll(".m-tab")];
  const panels = [...document.querySelectorAll(".m-panel")];
  const indicator = document.querySelector("[data-indicator]");
  if (!tabs.length) return;
  function moveIndicator(tab) { if (indicator) { indicator.style.width = tab.offsetWidth + "px"; indicator.style.transform = `translateX(${tab.offsetLeft - 6}px)`; } }
  function activate(tab) {
    tabs.forEach((t) => { const on = t === tab; t.classList.toggle("is-active", on); t.setAttribute("aria-selected", String(on)); });
    const key = tab.dataset.profile;
    panels.forEach((p) => { const on = p.dataset.panel === key; p.hidden = !on; p.classList.toggle("is-active", on); });
    moveIndicator(tab);
  }
  tabs.forEach((tab) => tab.addEventListener("click", () => { activate(tab); const r = tab.getBoundingClientRect(); BrainField && BrainField.ignite(r.left + r.width / 2, r.top + r.height / 2); }));
  const initial = tabs.find((t) => t.classList.contains("is-active")) || tabs[0];
  requestAnimationFrame(() => moveIndicator(initial));
  window.addEventListener("resize", () => { const a = tabs.find((t) => t.classList.contains("is-active")); if (a) moveIndicator(a); });
}

/* =========================================================
   MANUSCRIPT RADAR
   ========================================================= */
const PAPERS = [
  { short: "Cyber-manipulation", year: 2026, kind: "Multi-agent simulation", cat: "cyber",
    title: "Inter-individual Differences in Susceptibility to Cyber-manipulation of Political Opinions",
    summary: "Ontology-based multi-agent simulation with private and exposure-network susceptibility modelling of how political opinions shift under manipulation.",
    tags: ["Cyberpsychology", "Opinion dynamics", "Ontology"],
    repo: "https://github.com/stvsever/research_paper_on_cybermanipulation_susceptibility", img: "assets/papers/susceptibility.png" },
  { short: "Fibromyalgia EMA", year: 2026, kind: "EMA network analysis", cat: "health",
    title: "\"Walk on\": Physical Activity Dynamics in Fibromyalgia Patients",
    summary: "Temporal network analysis combining 1,474 EMA assessments with wrist-accelerometer ENMO and multilevel VAR models.",
    tags: ["EMA", "Fibromyalgia", "mlVAR"],
    repo: "https://github.com/stvsever/research_paper_on_physical_activity_in_fibromyalgia_patients", img: "assets/papers/fibromyalgia.png" },
  { short: "PHOENIX Engine", year: 2026, kind: "Agentic engine", cat: "ai",
    title: "PHOENIX Engine: Next-Generation Mental Health Applications",
    summary: "Ontology-backed agentic workflow for idiographic modelling and adaptive mental-health intervention design.",
    tags: ["AI agent", "Mental health", "Ontology"],
    repo: "https://github.com/stvsever/ThesisMaster", img: "assets/papers/phoenix.png" },
  { short: "COMPASS Engine", year: 2026, kind: "Multimodal prediction", cat: "ai",
    title: "COMPASS Engine: Clinical Ontology-driven Multimodal Predictive Agentic Support",
    summary: "Multi-agent prediction system for multimodal phenotypic inference from hierarchical deviation maps and non-tabular records.",
    tags: ["Predictive AI", "Clinical NLP", "Neuroimaging"],
    repo: "https://github.com/stvsever/COMPASS-Engine", img: "assets/papers/compass.png" },
  { short: "Cognitive bias grid", year: 2026, kind: "Bias mapping", cat: "cyber",
    title: "Cognitive Bias Vulnerability Across Emotional States in Cyber-Relevant Decision Contexts",
    summary: "Emotion-conditioned cognitive-bias mapping across 239 emotions and 200 bias dimensions.",
    tags: ["Cognitive bias", "Emotion", "Cybersecurity"],
    repo: "https://github.com/stvsever/research_paper_on_cognitive_biases_across_emotional_states_cyberrelevant", img: "assets/papers/cognitive_bias.png" },
  { short: "Hedonic bias", year: 2026, kind: "Computational meta-analysis", cat: "health",
    title: "Hedonic Bias in the Conceptualisation of Patient Progress in Well-Being",
    summary: "Computational meta-analysis of clinical scales, embeddings, and Western mental-healthcare assessment.",
    tags: ["Well-being", "Scale development", "Embeddings"],
    repo: "https://github.com/stvsever/research_paper_on_eudaimonia_in_healthcare", img: "assets/papers/eudaimonia.png" },
  { short: "Maladaptive perseveration", year: 2025, kind: "Experimental study", cat: "cyber",
    title: "Maladaptive Perseveration: Irrefutable Belief Systems & Insightful Reasoning",
    summary: "Experimental psychographic and computational-neuroscience study with PsychoPy, raw data, and R/Python analysis.",
    tags: ["Psychology", "Neuroscience", "Reasoning"],
    repo: "https://github.com/stvsever/research_paper_on_unfalsifiability", img: "assets/papers/unfalsifiability.png" },
];
const CAT_COLOR = { cyber: "var(--violet)", health: "var(--mint)", ai: "var(--cyan)" };
const GH_SVG = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 2.5a9.5 9.5 0 0 0-3 18.52c.48.1.66-.2.66-.46v-1.72c-2.68.58-3.24-1.14-3.24-1.14-.44-1.1-1.06-1.4-1.06-1.4-.86-.58.06-.58.06-.58.96.08 1.46.98 1.46.98.84 1.46 2.22 1.04 2.76.8.08-.62.32-1.04.58-1.28-2.14-.24-4.4-1.08-4.4-4.76 0-1.06.38-1.92.98-2.6-.1-.24-.42-1.24.1-2.56 0 0 .8-.26 2.62.98A9 9 0 0 1 12 5.46c.82 0 1.64.1 2.42.32 1.82-1.24 2.62-.98 2.62-.98.52 1.32.2 2.32.1 2.56.62.68.98 1.54.98 2.6 0 3.7-2.26 4.52-4.42 4.76.34.3.66.9.66 1.82v2.7c0 .26.18.56.68.46A9.5 9.5 0 0 0 12 2.5Z"/></svg>';

function setupRadar() {
  const radar = document.querySelector("[data-radar]");
  const blipsWrap = document.querySelector("[data-blips]");
  const detail = document.querySelector("[data-detail]");
  if (!radar || !blipsWrap || !detail) return;

  const N = PAPERS.length, R = 47;
  const blips = PAPERS.map((p, i) => {
    const ang = -90 + i * (360 / N);
    const rad = (ang * Math.PI) / 180;
    const x = 50 + R * Math.cos(rad), y = 50 + R * Math.sin(rad);
    const btn = document.createElement("button");
    btn.className = "blip" + (x < 49 ? " left" : "");
    btn.type = "button";
    btn.style.left = x + "%"; btn.style.top = y + "%";
    btn.style.setProperty("--bc", CAT_COLOR[p.cat]);
    btn.style.setProperty("--pd", (-(((ang + 90) % 360) / 360) * 8).toFixed(2) + "s");
    btn.setAttribute("aria-label", p.title);
    btn.innerHTML = `<span class="blip-dot"></span><span class="blip-label">${p.short}</span>`;
    btn.addEventListener("click", () => select(i));
    blipsWrap.appendChild(btn);
    return btn;
  });

  function repoLabel(url) { return url.replace("https://github.com/stvsever/", ""); }

  function select(i) {
    const p = PAPERS[i];
    blips.forEach((b, k) => b.classList.toggle("is-active", k === i));
    detail.innerHTML =
      `<div class="detail-card" style="--bc:${CAT_COLOR[p.cat]}">
        <div class="detail-top"><span class="detail-year">${p.year}</span><span class="detail-kind">${p.kind}</span></div>
        <h3>${p.title}</h3>
        <p>${p.summary}</p>
        <div class="detail-fig"><img alt="" decoding="async"></div>
        <div class="detail-tags">${p.tags.map((t) => `<span>${t}</span>`).join("")}</div>
        <a class="repo-link" href="${p.repo}" target="_blank" rel="noreferrer">${GH_SVG}<span class="repo-path">github.com/stvsever/<b>${repoLabel(p.repo)}</b></span><svg class="repo-arrow" viewBox="0 0 24 24" aria-hidden="true"><path d="M7 17 17 7"/><path d="M8 7h9v9"/></svg></a>
      </div>`;
    const img = detail.querySelector(".detail-fig img");
    img.addEventListener("load", () => img.classList.add("loaded"));
    img.src = p.img;
    BrainField && BrainField.ignite(window.innerWidth * 0.5, window.innerHeight * 0.5);
  }
  select(0);
}

/* =========================================================
   CAPABILITIES
   ========================================================= */
function setupCapabilities() {
  const tabs = [...document.querySelectorAll(".cap-tab")];
  const panels = [...document.querySelectorAll(".cap-panel")];
  if (!tabs.length) return;
  tabs.forEach((tab) => tab.addEventListener("click", () => {
    const key = tab.dataset.cap;
    tabs.forEach((t) => { const on = t === tab; t.classList.toggle("is-active", on); t.setAttribute("aria-selected", String(on)); });
    panels.forEach((p) => { const on = p.dataset.capPanel === key; p.hidden = !on; p.classList.toggle("is-active", on); });
    const r = tab.getBoundingClientRect(); BrainField && BrainField.ignite(r.left + r.width / 2, r.top + r.height / 2);
  }));
}

/* =========================================================
   CONTACT
   ========================================================= */
function setupContact() {
  const overlay = document.querySelector("[data-contact-overlay]");
  const toggles = [...document.querySelectorAll("[data-contact-toggle]")];
  const closeBtn = document.querySelector("[data-contact-close]");
  const status = document.querySelector("[data-copy-status]");
  if (!overlay) return;
  let lastFocus = null;
  function setOpen(open) {
    overlay.hidden = !open;
    toggles.forEach((t) => t.setAttribute("aria-expanded", String(open)));
    document.body.style.overflow = open ? "hidden" : "";
    if (open) { lastFocus = document.activeElement; overlay.querySelector(".contact-card")?.focus(); BrainField && BrainField.ignite(window.innerWidth * 0.5, window.innerHeight * 0.4); }
    else if (lastFocus) lastFocus.focus();
  }
  toggles.forEach((t) => t.addEventListener("click", () => setOpen(overlay.hidden)));
  closeBtn?.addEventListener("click", () => setOpen(false));
  overlay.addEventListener("click", (e) => { if (e.target === overlay) setOpen(false); });
  window.addEventListener("keydown", (e) => { if (e.key === "Escape" && !overlay.hidden) setOpen(false); });
  [...document.querySelectorAll("[data-copy]")].forEach((btn) => btn.addEventListener("click", async () => {
    const val = btn.dataset.copy || ""; let ok = true;
    try { await navigator.clipboard.writeText(val); } catch { ok = false; }
    if (status && !overlay.hidden) status.textContent = ok ? `Copied ${val}` : val;
    const act = btn.querySelector(".cf-act");
    if (act) { act.textContent = "copied"; setTimeout(() => (act.textContent = btn.tagName === "A" ? "open" : "copy"), 1400); }
    if (btn.classList.contains("quick-copy")) { const t0 = btn.textContent; btn.textContent = "copied to clipboard"; setTimeout(() => (btn.textContent = t0), 1400); }
  }));
}

/* =========================================================
   INIT
   ========================================================= */
function init() {
  BrainField && BrainField.start();
  setupHeroCore();
  setupRotator();
  setupNav();
  setupReveal();
  setupMatrix();
  setupRadar();
  setupCapabilities();
  setupContact();
}
if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
else init();
