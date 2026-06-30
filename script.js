/* =========================================================
   Stijn Van Severen - interaction + 3D brain field
   ========================================================= */
document.documentElement.classList.add("js-enabled");
const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

/* =========================================================
   BRAIN FIELD (3D)
   A dense, MNI-like brain volume that rotates on its own.
   Nodes are parcellated into the 7 canonical functional
   networks (Yeo 2011). At rest the whole brain is grey.
   Activity originates inside one network (DMN biased at
   rest): its ROIs light up in the network colour, a 3D
   wavefront sweeps the network, and the activation disperses
   to functionally connected networks (per a between-network
   connectivity matrix), then decays back to grey.
   ========================================================= */
const BrainField = (() => {
  const canvas = document.getElementById("neural-field");
  if (!canvas) return null;
  const ctx = canvas.getContext("2d");

  const GREY = [148, 160, 184];
  const COL = [
    [150, 78, 196], [78, 146, 214], [60, 196, 110], [206, 104, 240],
    [232, 212, 140], [240, 152, 64], [228, 84, 102],
  ];
  // Approximate right-hemisphere parcel anchors in normalised brain space
  // (x right, y superior, z anterior), mirrored to the left hemisphere below.
  const PARCELS_R = [
    [[0.12, -0.02, -0.84], [0.32, -0.10, -0.67], [0.18, 0.18, -0.74]], // 0 Visual
    [[0.28, 0.55, -0.04], [0.42, 0.42, 0.10], [0.50, 0.02, -0.18]],   // 1 Somatomotor
    [[0.34, 0.43, -0.36], [0.40, 0.34, 0.38], [0.16, 0.48, -0.16]],   // 2 Dorsal attention
    [[0.50, 0.10, -0.06], [0.18, 0.34, 0.24], [0.44, 0.04, 0.32]],    // 3 Ventral attention / salience
    [[0.30, -0.46, 0.40], [0.14, -0.42, 0.52], [0.44, -0.22, 0.16]],  // 4 Limbic
    [[0.48, 0.34, 0.48], [0.46, 0.30, -0.42], [0.22, 0.22, 0.46], [0.30, 0.46, -0.16]], // 5 Frontoparietal
    [[0.08, 0.22, 0.66], [0.06, 0.26, -0.54], [0.46, 0.30, -0.48], [0.54, -0.18, 0.0], [0.20, -0.32, 0.28], [0.04, -0.08, 0.18]], // 6 Default
  ];
  const PARCELS = [];
  PARCELS_R.forEach((list, net) => {
    list.forEach((c, idx) => {
      PARCELS.push({ x: c[0], y: c[1], z: c[2], net, parcel: idx * 2 });
      PARCELS.push({ x: -c[0], y: c[1], z: c[2], net, parcel: idx * 2 + 1 });
    });
  });
  const W = [
    [1.0, .30, .42, .20, .12, .18, .12],
    [.30, 1.0, .38, .36, .15, .20, .14],
    [.42, .38, 1.0, .40, .14, .48, .12],
    [.20, .36, .40, 1.0, .22, .50, .20],
    [.12, .15, .14, .22, 1.0, .20, .34],
    [.18, .20, .48, .50, .20, 1.0, .52],
    [.12, .14, .12, .20, .34, .52, 1.0],
  ];
  const SPONT = [.11, .09, .12, .16, .10, .16, .26];

  let W_ = 0, H = 0, dpr = 1, RAD = 320, cx = 0, cy = 0;
  let nodes = [], edges = [], order = [];
  let pulses = [], sparks = [];
  let raf = null, scrollY = 0, startTime = 0, nextSpont = 0;
  const netAct = new Float32Array(7);
  const netSeed = new Array(7).fill(null);
  let pending = [];
  let pointerX = 0, pointerY = 0;
  const clamp = (v, a, b) => (v < a ? a : v > b ? b : v);
  const mix = (a, b, t) => a + (b - a) * t;
  const tiltX = -0.49, tiltZ = 0.075, FOCAL = 2.85;

  function nodeCount() {
    const w = window.innerWidth;
    if (w < 620) return 520;
    if (w < 980) return 860;
    if (w < 1400) return 1350;
    return 1900;
  }

  function inBrain(x, y, z) {
    const rz = 0.62;
    const taper = z > 0 ? 1 - 0.30 * (z / rz) : 1 + 0.05 * (z / rz);
    const rx = 0.52 * taper, ry = 0.43 * (z > 0 ? 1 - 0.12 * (z / rz) : 1);
    if ((x / rx) ** 2 + (y / ry) ** 2 + (z / rz) ** 2 > 1) return false;
    if (y > 0.12 && Math.abs(x) < 0.035) return false; // longitudinal fissure groove
    return true;
  }
  function assignParcel(x, y, z) {
    let best = Infinity, hit = PARCELS[PARCELS.length - 1];
    for (const p of PARCELS) {
      const d = (x - p.x) ** 2 + (y - p.y) ** 2 + (z - p.z) ** 2;
      if (d < best) { best = d; hit = p; }
    }
    return hit;
  }

  function build() {
    const count = nodeCount();
    nodes = []; let guard = 0;
    while (nodes.length < count && guard < count * 50) {
      guard++;
      const x = (Math.random() * 2 - 1) * 0.55, y = (Math.random() * 2 - 1) * 0.46, z = (Math.random() * 2 - 1) * 0.64;
      if (!inBrain(x, y, z)) continue;
      const p = assignParcel(x, y, z);
      nodes.push({ x, y, z, net: p.net, parcel: p.parcel, size: 0.42 + Math.random() * 0.82, act: 0, sx: 0, sy: 0, depth: 0, scale: 1 });
    }
    order = nodes.map((_, i) => i);
    // precompute rigid edges (3D knn within radius, filtered by connectivity)
    edges = [];
    const N = nodes.length, rE2 = 0.145 * 0.145, maxDeg = 8, seen = new Set();
    for (let i = 0; i < N; i++) {
      const a = nodes[i], cand = [];
      for (let j = 0; j < N; j++) {
        if (j === i) continue;
        const b = nodes[j];
        if (W[a.net][b.net] < 0.14) continue;
        const d2 = (a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2;
        if (d2 < rE2) cand.push([d2, j]);
      }
      cand.sort((u, v) => u[0] - v[0]);
      for (let k = 0; k < Math.min(cand.length, maxDeg); k++) {
        const j = cand[k][1], key = i < j ? i * N + j : j * N + i;
        if (seen.has(key)) continue; seen.add(key);
        edges.push({ i, j, w: W[a.net][nodes[j].net] });
      }
    }
  }

  function resize() {
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    W_ = window.innerWidth; H = window.innerHeight;
    canvas.width = Math.floor(W_ * dpr); canvas.height = Math.floor(H * dpr);
    canvas.style.width = W_ + "px"; canvas.style.height = H + "px";
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    RAD = Math.min(W_, H) * (W_ < 760 ? 0.72 : 0.82);
    cx = W_ * 0.5; cy = H * 0.47;
    build();
  }

  function projectPoint(x, y, z, t) {
    const ay = t * 0.000148 + pointerX * 0.5;
    const cosY = Math.cos(ay), sinY = Math.sin(ay), cosT = Math.cos(tiltX), sinT = Math.sin(tiltX);
    const cosZ = Math.cos(tiltZ), sinZ = Math.sin(tiltZ);
    const x1 = x * cosY + z * sinY, z1 = -x * sinY + z * cosY, y1 = y;
    const y2 = y1 * cosT - z1 * sinT, z2 = y1 * sinT + z1 * cosT;
    const x2 = x1 * cosZ - y2 * sinZ, y3 = x1 * sinZ + y2 * cosZ;
    const sc = FOCAL / (FOCAL - z2);
    return { sx: cx + x2 * sc * RAD, sy: cy + y3 * sc * RAD, depth: z2, scale: sc };
  }

  function project(t) {
    for (const n of nodes) {
      const p = projectPoint(n.x, n.y, n.z, t);
      n.sx = p.sx; n.sy = p.sy; n.depth = p.depth; n.scale = p.scale;
    }
  }

  /* activation */
  function addSparks(x, y, net, amt) {
    if (reduceMotion) return;
    const c = COL[net];
    for (let i = 0; i < amt; i++) { const a = Math.random() * 6.2832, s = 0.5 + Math.random() * 2.2; sparks.push({ x, y, vx: Math.cos(a) * s, vy: Math.sin(a) * s, life: 1, c }); }
  }
  function nearestInNet(net, x, y, z) {
    let best = Infinity, bn = null;
    for (const n of nodes) if (n.net === net) { const d = (n.x - x) ** 2 + (n.y - y) ** 2 + (n.z - z) ** 2; if (d < best) { best = d; bn = n; } }
    return bn;
  }
  function activate(net, pt, strength, now) {
    netAct[net] = Math.min(1.5, Math.max(netAct[net], strength));
    netSeed[net] = { x: pt.x, y: pt.y, z: pt.z, t0: now };
    addSparks(pt.sx || cx, pt.sy || cy, net, 5 + (strength * 7) | 0);
    if (!reduceMotion) {
      let fired = 0;
      for (const e of edges) {
        if (fired >= 5) break;
        const a = nodes[e.i], b = nodes[e.j];
        if (a.net !== net && b.net !== net) continue;
        const seedNode = (a.x === pt.x && a.y === pt.y) ? a : (b.x === pt.x && b.y === pt.y) ? b : null;
        if (!seedNode) continue;
        pulses.push({ a: e.i, b: e.j, t: 0, v: 0.04 + Math.random() * 0.02, c: COL[net] });
        fired++;
      }
    }
    if (strength < 0.2) return;
    for (let j = 0; j < 7; j++) {
      if (j === net) continue;
      const w = W[net][j]; if (w < 0.3) continue;
      const tn = nearestInNet(j, pt.x, pt.y, pt.z); if (!tn) continue;
      pending.push({ net: j, pt: tn, strength: strength * w * 0.82, fireAt: now + 200 + (1 - w) * 440 + Math.random() * 140 });
    }
  }
  function processPending(now) {
    for (let i = pending.length - 1; i >= 0; i--) if (now >= pending[i].fireAt) { const e = pending[i]; pending.splice(i, 1); activate(e.net, e.pt, e.strength, now); }
  }
  function spontaneous(now) {
    if (reduceMotion || !nodes.length) return;
    if (!nextSpont) { nextSpont = now + 450; return; }
    if (now < nextSpont) return;
    let tot = 0; for (const v of SPONT) tot += v;
    let r = Math.random() * tot, s = 0, net = 6;
    for (let i = 0; i < 7; i++) { s += SPONT[i]; if (r <= s) { net = i; break; } }
    let pick = null, cnt = 0;
    for (const n of nodes) if (n.net === net && Math.random() < 1 / (++cnt)) pick = n;
    if (pick) activate(net, pick, 1.0, now);
    nextSpont = now + 460 + Math.random() * 760;
  }
  function updateActivation(t) {
    for (let i = 0; i < 7; i++) netAct[i] *= 0.985;
    for (const n of nodes) {
      const seed = netSeed[n.net];
      let target = 0;
      if (seed) {
        const dist = Math.sqrt((n.x - seed.x) ** 2 + (n.y - seed.y) ** 2 + (n.z - seed.z) ** 2);
        const wf = (t - seed.t0) * 0.0013;
        const gate = dist <= wf ? 1 : Math.exp(-((dist - wf) ** 2) / (2 * 0.16 * 0.16));
        target = netAct[n.net] * gate;
      }
      n.act += (target - n.act) * 0.16; n.act *= 0.985;
      if (n.act > 1.5) n.act = 1.5; else if (n.act < 0) n.act = 0;
    }
  }

  /* draw */
  function drawParcels(t) {
    for (const p of PARCELS) {
      const q = projectPoint(p.x, p.y, p.z, t);
      const c = COL[p.net];
      const fog = 0.25 + 0.55 * clamp((q.depth + 0.7) / 1.4, 0, 1);
      const r = RAD * q.scale * (0.038 + (p.parcel % 3) * 0.006);
      const g = ctx.createRadialGradient(q.sx, q.sy, 0, q.sx, q.sy, r);
      g.addColorStop(0, `rgba(${c[0]},${c[1]},${c[2]},${0.035 * fog})`);
      g.addColorStop(0.58, `rgba(${c[0]},${c[1]},${c[2]},${0.012 * fog})`);
      g.addColorStop(1, `rgba(${c[0]},${c[1]},${c[2]},0)`);
      ctx.fillStyle = g;
      ctx.beginPath();
      ctx.arc(q.sx, q.sy, r, 0, 6.2832);
      ctx.fill();
    }
  }

  function drawEdges() {
    for (const e of edges) {
      const a = nodes[e.i], b = nodes[e.j];
      const act = a.act > b.act ? a.act : b.act;
      const depth = (a.depth + b.depth) * 0.5;
      const fog = 0.35 + 0.65 * clamp((depth + 0.7) / 1.4, 0, 1);
      const sameNet = a.net === b.net;
      const sameParcel = sameNet && a.parcel === b.parcel;
      let col, alpha;
      if (act > 0.05) {
        const c = a.act > b.act ? COL[a.net] : COL[b.net];
        col = [mix(GREY[0], c[0], act), mix(GREY[1], c[1], act), mix(GREY[2], c[2], act)];
        alpha = (0.05 + e.w * 0.05) * fog + act * 0.22;
      } else {
        const c = sameNet ? COL[a.net] : GREY;
        const tint = sameNet ? 0.12 : 0;
        col = [mix(GREY[0], c[0], tint), mix(GREY[1], c[1], tint), mix(GREY[2], c[2], tint)];
        alpha = (0.026 + e.w * 0.04) * fog * (sameNet ? 1 : 0.42) * (sameParcel ? 1.28 : 1);
      }
      ctx.strokeStyle = `rgba(${col[0] | 0},${col[1] | 0},${col[2] | 0},${alpha})`;
      ctx.lineWidth = 0.5 + act * 0.7;
      ctx.beginPath(); ctx.moveTo(a.sx, a.sy); ctx.lineTo(b.sx, b.sy); ctx.stroke();
    }
  }
  function drawNodes() {
    order.sort((u, v) => nodes[u].depth - nodes[v].depth);
    for (const k of order) {
      const n = nodes[k], a = n.act, c = COL[n.net];
      const fog = 0.35 + 0.65 * clamp((n.depth + 0.7) / 1.4, 0, 1);
      const baseTint = 0.075 + (n.parcel % 4) * 0.008;
      const tint = a > 0.05 ? Math.min(1, baseTint + a) : baseTint;
      const col = [mix(GREY[0], c[0], tint) | 0, mix(GREY[1], c[1], tint) | 0, mix(GREY[2], c[2], tint) | 0];
      const r = n.size * n.scale * (1 + a * 0.8);
      if (a > 0.05) { ctx.beginPath(); ctx.arc(n.sx, n.sy, r * 4, 0, 6.2832); ctx.fillStyle = `rgba(${col[0]},${col[1]},${col[2]},${a * 0.09 * fog})`; ctx.fill(); }
      ctx.beginPath(); ctx.arc(n.sx, n.sy, r, 0, 6.2832);
      ctx.fillStyle = `rgba(${col[0]},${col[1]},${col[2]},${(0.18 + a * 0.62) * fog})`; ctx.fill();
      if (a < 0.05 && n.parcel % 5 === 0) {
        ctx.beginPath(); ctx.arc(n.sx, n.sy, r * 1.9, 0, 6.2832);
        ctx.strokeStyle = `rgba(${c[0]},${c[1]},${c[2]},${0.055 * fog})`;
        ctx.lineWidth = 0.35;
        ctx.stroke();
      }
    }
  }
  function drawPulses() {
    pulses = pulses.filter((p) => p.t < 1);
    for (const p of pulses) {
      p.t += p.v;
      const a = nodes[p.a], b = nodes[p.b], c = p.c;
      const x = a.sx + (b.sx - a.sx) * p.t, y = a.sy + (b.sy - a.sy) * p.t;
      ctx.beginPath(); ctx.arc(x, y, 2, 0, 6.2832); ctx.fillStyle = `rgba(${c[0]},${c[1]},${c[2]},0.95)`; ctx.fill();
      ctx.beginPath(); ctx.arc(x, y, 6, 0, 6.2832); ctx.fillStyle = `rgba(${c[0]},${c[1]},${c[2]},0.12)`; ctx.fill();
    }
  }
  function drawSparks() {
    sparks = sparks.filter((s) => s.life > 0.04);
    for (const s of sparks) {
      s.x += s.vx; s.y += s.vy; s.vx *= 0.92; s.vy *= 0.92; s.life *= 0.9;
      ctx.beginPath(); ctx.arc(s.x, s.y, 1.8 * s.life, 0, 6.2832);
      ctx.fillStyle = `rgba(${s.c[0]},${s.c[1]},${s.c[2]},${s.life})`; ctx.fill();
    }
  }

  function frame(t) {
    if (!startTime) startTime = t;
    const el = t - startTime;
    ctx.clearRect(0, 0, W_, H);
    project(t);
    if (!reduceMotion) { spontaneous(t); processPending(t); }
    updateActivation(t);
    const intro = el < 1300 && !reduceMotion ? el / 1300 : 1;
    const dim = clamp(1 - (scrollY / (H * 1.3)) * 0.45, 0.55, 1);
    ctx.globalAlpha = intro * dim;
    drawParcels(t);
    drawEdges();
    drawNodes();
    drawPulses();
    drawSparks();
    ctx.globalAlpha = 1;
    if (!reduceMotion) raf = requestAnimationFrame(frame);
  }

  function igniteAt(x, y) {
    let best = Infinity, bn = null;
    for (const n of nodes) { const d = (n.sx - x) ** 2 + (n.sy - y) ** 2; if (d < best) { best = d; bn = n; } }
    if (bn) activate(bn.net, bn, 1.1, performance.now());
  }

  function start() {
    resize();
    window.addEventListener("resize", () => { if (raf) cancelAnimationFrame(raf); startTime = 0; resize(); if (reduceMotion) frame(0); else raf = requestAnimationFrame(frame); });
    window.addEventListener("pointermove", (e) => { pointerX = (e.clientX / window.innerWidth - 0.5) * 0.6; pointerY = e.clientY / window.innerHeight - 0.5; }, { passive: true });
    window.addEventListener("pointerdown", (e) => igniteAt(e.clientX, e.clientY), { passive: true });
    window.addEventListener("scroll", () => { scrollY = window.scrollY; }, { passive: true });
    if (reduceMotion) frame(0); else raf = requestAnimationFrame(frame);
  }
  return { start, ignite: (x, y) => igniteAt(x, y) };
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
   FOCUS ROTATOR (brackets hug the active word)
   ========================================================= */
function setupRotator() {
  const rot = document.querySelector("[data-rotator]");
  if (!rot) return;
  const items = [...rot.children];
  if (!items.length) return;
  let i = 0;
  const fit = () => { rot.style.width = (items[i].offsetWidth + 4) + "px"; };
  requestAnimationFrame(fit); setTimeout(fit, 400);
  window.addEventListener("resize", fit);
  if (items.length < 2 || reduceMotion) return;
  setInterval(() => {
    items[i].classList.remove("is-active"); items[i].classList.add("is-out");
    const prev = i; i = (i + 1) % items.length;
    items[i].classList.remove("is-out"); items[i].classList.add("is-active");
    fit();
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
    const ang = -90 + i * (360 / N), rad = (ang * Math.PI) / 180;
    const x = 50 + R * Math.cos(rad), y = 50 + R * Math.sin(rad);
    const btn = document.createElement("button");
    btn.className = "blip" + (x < 49 ? " left" : "");
    btn.type = "button"; btn.style.left = x + "%"; btn.style.top = y + "%";
    btn.style.setProperty("--bc", CAT_COLOR[p.cat]);
    btn.style.setProperty("--pd", (-(((ang + 90) % 360) / 360) * 8).toFixed(2) + "s");
    btn.setAttribute("aria-label", p.title);
    btn.innerHTML = `<span class="blip-dot"></span><span class="blip-label">${p.short}</span>`;
    btn.addEventListener("click", () => select(i));
    blipsWrap.appendChild(btn);
    return btn;
  });
  const repoLabel = (url) => url.replace("https://github.com/stvsever/", "");
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
    if (btn.classList.contains("quick-copy")) { const lbl = btn.querySelector(".cq-label"); if (lbl) { const t0 = lbl.textContent; lbl.textContent = "copied to clipboard"; setTimeout(() => (lbl.textContent = t0), 1400); } }
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
