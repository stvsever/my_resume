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

  const GREY = [164, 170, 184];
  const COL = [
    [150, 78, 196], [78, 146, 214], [60, 196, 110], [206, 104, 240],
    [232, 212, 140], [240, 152, 64], [228, 84, 102],
  ];
  // Schaefer2018 100-parcel atlas, 7-network order, FSLMNI152 2mm centroid RAS coordinates.
  // Source: Thomas Yeo Lab / CBIG. RAS is normalised to renderer space: x right, y superior, z anterior.
  // Network ids: 0 Visual, 1 Somatomotor, 2 Dorsal attention, 3 Ventral attention / salience,
  // 4 Limbic, 5 Frontoparietal control, 6 Default mode.
  const PARCELS = [
    { net: 0, parcel: 0, x: -0.2080, y: -0.1200, z: -0.2720 },
    { net: 0, parcel: 1, x: -0.2080, y: -0.0933, z: -0.6080 },
    { net: 0, parcel: 2, x: -0.1440, y: -0.0400, z: -0.4800 },
    { net: 0, parcel: 3, x: -0.2080, y: -0.0267, z: -0.7680 },
    { net: 0, parcel: 4, x: -0.0480, y: -0.0133, z: -0.7360 },
    { net: 0, parcel: 5, x: -0.0960, y: 0.0400, z: -0.5280 },
    { net: 0, parcel: 6, x: -0.3840, y: 0.0667, z: -0.5600 },
    { net: 0, parcel: 7, x: -0.2080, y: 0.1333, z: -0.7040 },
    { net: 0, parcel: 8, x: -0.0480, y: 0.1733, z: -0.6560 },
    { net: 1, parcel: 9, x: -0.4320, y: 0.0533, z: -0.1760 },
    { net: 1, parcel: 10, x: -0.2880, y: 0.1067, z: -0.1760 },
    { net: 1, parcel: 11, x: -0.4320, y: 0.0933, z: -0.0960 },
    { net: 1, parcel: 12, x: -0.4320, y: 0.2267, z: -0.0640 },
    { net: 1, parcel: 13, x: -0.3200, y: 0.3867, z: -0.1760 },
    { net: 1, parcel: 14, x: -0.0480, y: 0.4667, z: -0.2240 },
    { net: 2, parcel: 15, x: -0.3680, y: -0.0800, z: -0.4640 },
    { net: 2, parcel: 16, x: -0.4640, y: 0.2533, z: -0.1920 },
    { net: 2, parcel: 17, x: -0.1920, y: 0.3333, z: -0.5440 },
    { net: 2, parcel: 18, x: -0.3360, y: 0.3200, z: -0.2720 },
    { net: 2, parcel: 19, x: -0.0480, y: 0.3733, z: -0.4800 },
    { net: 2, parcel: 20, x: -0.1760, y: 0.4400, z: -0.4000 },
    { net: 2, parcel: 21, x: -0.3840, y: 0.1867, z: 0.0480 },
    { net: 2, parcel: 22, x: -0.2080, y: 0.4000, z: -0.0320 },
    { net: 3, parcel: 23, x: -0.4640, y: 0.2000, z: -0.3040 },
    { net: 3, parcel: 24, x: -0.3360, y: -0.0400, z: -0.0160 },
    { net: 3, parcel: 25, x: -0.3040, y: 0.0400, z: 0.0960 },
    { net: 3, parcel: 26, x: -0.2400, y: 0.2000, z: 0.3520 },
    { net: 3, parcel: 27, x: -0.0480, y: 0.2267, z: 0.1600 },
    { net: 3, parcel: 28, x: -0.0960, y: 0.3067, z: -0.2720 },
    { net: 3, parcel: 29, x: -0.0480, y: 0.4133, z: 0.0320 },
    { net: 4, parcel: 30, x: -0.1120, y: -0.1333, z: 0.2560 },
    { net: 4, parcel: 31, x: -0.2560, y: -0.2533, z: 0.0160 },
    { net: 4, parcel: 32, x: -0.4640, y: -0.1467, z: -0.2560 },
    { net: 5, parcel: 33, x: -0.3040, y: 0.3067, z: -0.4160 },
    { net: 5, parcel: 34, x: -0.3520, y: 0.1333, z: 0.2560 },
    { net: 5, parcel: 35, x: -0.0800, y: 0.2533, z: -0.5920 },
    { net: 5, parcel: 36, x: -0.0320, y: 0.2267, z: -0.2080 },
    { net: 6, parcel: 37, x: -0.4480, y: -0.1333, z: -0.0320 },
    { net: 6, parcel: 38, x: -0.4640, y: -0.0133, z: -0.2560 },
    { net: 6, parcel: 39, x: -0.4640, y: 0.0800, z: -0.4000 },
    { net: 6, parcel: 40, x: -0.3840, y: 0.2400, z: -0.5120 },
    { net: 6, parcel: 41, x: -0.2720, y: -0.0667, z: 0.1760 },
    { net: 6, parcel: 42, x: -0.3680, y: -0.0267, z: 0.2720 },
    { net: 6, parcel: 43, x: -0.0480, y: 0.0000, z: 0.3680 },
    { net: 6, parcel: 44, x: -0.1920, y: -0.0133, z: 0.4800 },
    { net: 6, parcel: 45, x: -0.0640, y: 0.2800, z: 0.3840 },
    { net: 6, parcel: 46, x: -0.3200, y: 0.3200, z: 0.1120 },
    { net: 6, parcel: 47, x: -0.2080, y: 0.3467, z: 0.1600 },
    { net: 6, parcel: 48, x: -0.0960, y: 0.0800, z: -0.4480 },
    { net: 6, parcel: 49, x: -0.0480, y: 0.2267, z: -0.4160 },
    { net: 0, parcel: 50, x: 0.2560, y: -0.1467, z: -0.2560 },
    { net: 0, parcel: 51, x: 0.2240, y: -0.0800, z: -0.5280 },
    { net: 0, parcel: 52, x: 0.4000, y: -0.0667, z: -0.4800 },
    { net: 0, parcel: 53, x: 0.1760, y: -0.0267, z: -0.7680 },
    { net: 0, parcel: 54, x: 0.0640, y: 0.0400, z: -0.6080 },
    { net: 0, parcel: 55, x: 0.1440, y: 0.0400, z: -0.4640 },
    { net: 0, parcel: 56, x: 0.2880, y: 0.1067, z: -0.6560 },
    { net: 0, parcel: 57, x: 0.0960, y: 0.2000, z: -0.6880 },
    { net: 1, parcel: 58, x: 0.4160, y: 0.0400, z: -0.1280 },
    { net: 1, parcel: 59, x: 0.3200, y: 0.1067, z: -0.1280 },
    { net: 1, parcel: 60, x: 0.4480, y: 0.0800, z: -0.0320 },
    { net: 1, parcel: 61, x: 0.4640, y: 0.2000, z: -0.0480 },
    { net: 1, parcel: 62, x: 0.3680, y: 0.3200, z: -0.0960 },
    { net: 1, parcel: 63, x: 0.3200, y: 0.4000, z: -0.1760 },
    { net: 1, parcel: 64, x: 0.2400, y: 0.4267, z: -0.3040 },
    { net: 1, parcel: 65, x: 0.0480, y: 0.4667, z: -0.2240 },
    { net: 2, parcel: 66, x: 0.4000, y: 0.1067, z: -0.4960 },
    { net: 2, parcel: 67, x: 0.4000, y: 0.2800, z: -0.1920 },
    { net: 2, parcel: 68, x: 0.3040, y: 0.3333, z: -0.3680 },
    { net: 2, parcel: 69, x: 0.2080, y: 0.3467, z: -0.5280 },
    { net: 2, parcel: 70, x: 0.1120, y: 0.4400, z: -0.4160 },
    { net: 2, parcel: 71, x: 0.3840, y: 0.1733, z: 0.0800 },
    { net: 2, parcel: 72, x: 0.2240, y: 0.4000, z: -0.0160 },
    { net: 3, parcel: 73, x: 0.4640, y: 0.0800, z: -0.3360 },
    { net: 3, parcel: 74, x: 0.4800, y: 0.1867, z: -0.2080 },
    { net: 3, parcel: 75, x: 0.3200, y: 0.0133, z: 0.0640 },
    { net: 3, parcel: 76, x: 0.0960, y: 0.3067, z: -0.2560 },
    { net: 3, parcel: 77, x: 0.0640, y: 0.3467, z: 0.0480 },
    { net: 4, parcel: 78, x: 0.0960, y: -0.1333, z: 0.2720 },
    { net: 4, parcel: 79, x: 0.3040, y: -0.2533, z: 0.0000 },
    { net: 5, parcel: 80, x: 0.4640, y: 0.2933, z: -0.3040 },
    { net: 5, parcel: 81, x: 0.3680, y: 0.3067, z: -0.4960 },
    { net: 5, parcel: 82, x: 0.2400, y: -0.0133, z: 0.4640 },
    { net: 5, parcel: 83, x: 0.3680, y: 0.1067, z: 0.3040 },
    { net: 5, parcel: 84, x: 0.2560, y: 0.2000, z: 0.3680 },
    { net: 5, parcel: 85, x: 0.3520, y: 0.2933, z: 0.1280 },
    { net: 5, parcel: 86, x: 0.0480, y: 0.2267, z: -0.2240 },
    { net: 5, parcel: 87, x: 0.0480, y: 0.2000, z: 0.2240 },
    { net: 5, parcel: 88, x: 0.0800, y: 0.2800, z: -0.5280 },
    { net: 6, parcel: 89, x: 0.4320, y: 0.2000, z: -0.4000 },
    { net: 6, parcel: 90, x: 0.4960, y: -0.1200, z: -0.1920 },
    { net: 6, parcel: 91, x: 0.4000, y: -0.1200, z: 0.0480 },
    { net: 6, parcel: 92, x: 0.4640, y: -0.0133, z: -0.2080 },
    { net: 6, parcel: 93, x: 0.2880, y: -0.1067, z: 0.2080 },
    { net: 6, parcel: 94, x: 0.4000, y: 0.0000, z: 0.2240 },
    { net: 6, parcel: 95, x: 0.0480, y: 0.0000, z: 0.3840 },
    { net: 6, parcel: 96, x: 0.0960, y: 0.2667, z: 0.4000 },
    { net: 6, parcel: 97, x: 0.2080, y: 0.3333, z: 0.1920 },
    { net: 6, parcel: 98, x: 0.0960, y: 0.0933, z: -0.4320 },
    { net: 6, parcel: 99, x: 0.0480, y: 0.2000, z: -0.4160 },
  ];
  const W = [
    [1.0, .30, .42, .20, .12, .18, .12],
    [.30, 1.0, .38, .36, .15, .20, .14],
    [.42, .38, 1.0, .40, .14, .48, .12],
    [.20, .36, .40, 1.0, .22, .50, .20],
    [.12, .15, .14, .22, 1.0, .20, .34],
    [.18, .20, .48, .50, .20, 1.0, .52],
    [.12, .14, .12, .20, .34, .52, 1.0],
  ];
  const SPONT = [.08, .07, .11, .16, .08, .22, .36];
  const NET_GAIN = [.84, .78, .95, 1.04, .86, 1.22, 1.45];
  const NET_SPREAD = [.095, .09, .115, .12, .10, .15, .17];
  const REST_BIAS = [.0014, .0012, .0018, .0024, .0016, .0042, .0060];

  let W_ = 0, H = 0, dpr = 1, RAD = 320, cx = 0, cy = 0;
  let nodes = [], nodesByNet = [], nodesByParcel = [], edges = [], edgesByNet = [], order = [];
  let nodeFlashes = [];
  let raf = null, scrollY = 0, startTime = 0, nextSpont = 0;
  const netAct = new Float32Array(7);
  const netSeed = Array.from({ length: 7 }, () => []);
  let pending = [];
  const clamp = (v, a, b) => (v < a ? a : v > b ? b : v);
  const mix = (a, b, t) => a + (b - a) * t;
  const tiltX = -0.49, tiltZ = 0.075, FOCAL = 2.85;

  function nodeCount() {
    const w = window.innerWidth;
    if (w < 620) return 940;
    if (w < 980) return 1700;
    if (w < 1400) return 3000;
    return 4200;
  }

  function inBrain(x, y, z) {
    const rz = 0.78;
    const taper = z > 0 ? 1 - 0.30 * (z / rz) : 1 + 0.05 * (z / rz);
    const rx = 0.55 * taper, ry = 0.50 * (z > 0 ? 1 - 0.12 * (z / rz) : 1);
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
      const x = (Math.random() * 2 - 1) * 0.57, y = (Math.random() * 2 - 1) * 0.52, z = (Math.random() * 2 - 1) * 0.80;
      if (!inBrain(x, y, z)) continue;
      const p = assignParcel(x, y, z);
      nodes.push({ id: nodes.length, x, y, z, net: p.net, parcel: p.parcel, hemi: x < 0 ? -1 : 1, size: 0.15 + Math.random() * 0.32, act: 0, sx: 0, sy: 0, depth: 0, scale: 1 });
    }
    order = nodes.map((_, i) => i);
    nodesByNet = Array.from({ length: 7 }, () => []);
    nodesByParcel = Array.from({ length: PARCELS.length }, () => []);
    nodes.forEach((n, i) => {
      nodesByNet[n.net].push(i);
      nodesByParcel[n.parcel].push(i);
    });

    edges = [];
    edgesByNet = Array.from({ length: 7 }, () => []);
    const N = nodes.length, cell = 0.126, rE2 = 0.126 * 0.126, maxDeg = 4, seen = new Set();
    const grid = new Map();
    const gkey = (x, y, z) => `${Math.floor(x / cell)},${Math.floor(y / cell)},${Math.floor(z / cell)}`;
    nodes.forEach((n, i) => {
      const key = gkey(n.x, n.y, n.z);
      if (!grid.has(key)) grid.set(key, []);
      grid.get(key).push(i);
    });
    function addEdge(i, j, long = false, wOverride = null, tract = "local") {
      if (i === j) return false;
      const key = i < j ? i * N + j : j * N + i;
      if (seen.has(key)) return false;
      seen.add(key);
      const ni = nodes[i], nj = nodes[j];
      const idx = edges.length;
      edges.push({ i, j, w: wOverride == null ? W[ni.net][nj.net] : wOverride, long, tract });
      edgesByNet[ni.net].push(idx);
      if (nj.net !== ni.net) edgesByNet[nj.net].push(idx);
      return true;
    }

    // Precompute local 3D k-nearest neighbours through a spatial grid, filtered by functional connectivity.
    for (let i = 0; i < N; i++) {
      const a = nodes[i], cand = [];
      const gx = Math.floor(a.x / cell), gy = Math.floor(a.y / cell), gz = Math.floor(a.z / cell);
      for (let dx = -1; dx <= 1; dx++) for (let dy = -1; dy <= 1; dy++) for (let dz = -1; dz <= 1; dz++) {
        const bucket = grid.get(`${gx + dx},${gy + dy},${gz + dz}`);
        if (!bucket) continue;
        for (const j of bucket) {
          if (j === i) continue;
          const b = nodes[j];
          if (a.hemi !== b.hemi) continue;
          if (W[a.net][b.net] < 0.14) continue;
          const d2 = (a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2;
          if (d2 < rE2) cand.push([d2, j]);
        }
      }
      cand.sort((u, v) => u[0] - v[0]);
      for (let k = 0; k < Math.min(cand.length, maxDeg); k++) addEdge(i, cand[k][1]);
    }

    function pickNode(net, hemi) {
      const pool = nodesByNet[net].filter((idx) => nodes[idx].hemi === hemi);
      return pool.length ? pool[(Math.random() * pool.length) | 0] : null;
    }
    function pickParcel(parcel) {
      const pool = nodesByParcel[parcel] || [];
      return pool.length ? pool[(Math.random() * pool.length) | 0] : null;
    }
    for (const hemi of [-1, 1]) {
      for (let a = 0; a < 7; a++) for (let b = a + 1; b < 7; b++) {
        const w = W[a][b];
        if (w < 0.34) continue;
        const repeats = Math.round(2 + w * 10 + (a === 6 || b === 6 ? 3 : 0) + (a === 5 || b === 5 ? 2 : 0));
        for (let k = 0; k < repeats; k++) {
          const ia = pickNode(a, hemi), ib = pickNode(b, hemi);
          if (ia == null || ib == null) continue;
          const A = nodes[ia], B = nodes[ib];
          const d2 = (A.x - B.x) ** 2 + (A.y - B.y) ** 2 + (A.z - B.z) ** 2;
          if (d2 > 0.06 && d2 < 0.85) addEdge(ia, ib, true, w, "association");
        }
      }
    }

    const rightParcels = PARCELS.filter((p) => p.x > 0.035);
    for (const lp of PARCELS) {
      if (lp.x > -0.035) continue;
      let best = Infinity, rp = null;
      for (const cand of rightParcels) {
        if (cand.net !== lp.net) continue;
        const d = (-lp.x - cand.x) ** 2 + (lp.y - cand.y) ** 2 + (lp.z - cand.z) ** 2;
        if (d < best) { best = d; rp = cand; }
      }
      if (!rp || best > 0.10) continue;
      if (lp.net !== 6 && lp.net !== 5 && lp.net !== 1 && Math.random() > 0.32) continue;
      const ia = pickParcel(lp.parcel), ib = pickParcel(rp.parcel);
      if (ia != null && ib != null) addEdge(ia, ib, true, W[lp.net][rp.net] * 0.55, "commissural");
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
    const ay = t * 0.000148;
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
  function addNodeFlashes(net, pt, amt, strength, delay = 0) {
    if (reduceMotion) return;
    const pool = nodesByNet[net] || [];
    if (!pool.length) return;
    const local = [];
    const radius2 = (net === 6 ? 0.115 : net === 5 ? 0.088 : net === 3 ? 0.074 : 0.062);
    for (const idx of pool) {
      const n = nodes[idx];
      const d2 = (n.x - pt.x) ** 2 + (n.y - pt.y) ** 2 + (n.z - pt.z) ** 2;
      if (n.parcel === pt.parcel || d2 < radius2) local.push([d2, idx]);
    }
    local.sort((a, b) => a[0] - b[0]);
    const sourceId = pt.id != null ? pt.id : local[0]?.[1];
    if (sourceId != null) local.unshift([0, sourceId]);
    const amount = Math.max(amt, Math.round((net === 6 ? 46 : net === 5 ? 36 : net === 3 ? 30 : 22) * clamp(strength, 0.35, 1.3)));
    const c = COL[net];
    for (let i = 0; i < amount; i++) {
      const pick = local.length ? local[Math.min(local.length - 1, (Math.random() ** 1.7 * local.length) | 0)] : [0, pool[(Math.random() * pool.length) | 0]];
      const d = Math.sqrt(pick[0] || 0);
      nodeFlashes.push({
        i: pick[1],
        t: -(delay + Math.min(0.34, d * (net === 6 ? 1.05 : 0.9)) + Math.random() * 0.16),
        v: 0.046 + Math.random() * 0.034,
        r: 0.85 + Math.random() * 1.55,
        phase: Math.random() * 6.2832,
        c
      });
    }
    if (nodeFlashes.length > 940) nodeFlashes.splice(0, nodeFlashes.length - 940);
  }
  function nearestInNet(net, x, y, z) {
    let best = Infinity, bn = null;
    for (const n of nodes) if (n.net === net) { const d = (n.x - x) ** 2 + (n.y - y) ** 2 + (n.z - z) ** 2; if (d < best) { best = d; bn = n; } }
    return bn;
  }
  function distantInNet(net, x, y, z) {
    const pool = nodesByNet[net] || [];
    let best = -1, bn = null;
    for (let k = 0; k < 30 && pool.length; k++) {
      const n = nodes[pool[(Math.random() * pool.length) | 0]];
      const d = (n.x - x) ** 2 + (n.y - y) ** 2 + (n.z - z) ** 2;
      if (d > best) { best = d; bn = n; }
    }
    return bn;
  }
  function relayWeight(from, to) {
    let w = W[from][to];
    if (from === 6 && to === 5) w += 0.24;       // DMN -> frontoparietal control
    if (from === 5 && (to === 2 || to === 3)) w += 0.13; // FP -> attention/salience
    if (from === 3 && (to === 5 || to === 1)) w += 0.10; // salience -> control/motor
    if (from === 2 && to === 0) w += 0.08;       // dorsal attention -> visual
    return Math.min(1, w);
  }
  function activate(net, pt, strength, now, hop = 0) {
    const gain = NET_GAIN[net];
    const seedStrength = Math.min(2.1, strength * gain);
    netAct[net] = Math.max(netAct[net], seedStrength);
    netSeed[net].push({ x: pt.x, y: pt.y, z: pt.z, t0: now, strength: seedStrength });
    if (netSeed[net].length > 7) netSeed[net].splice(0, netSeed[net].length - 7);
    if (strength > 0.48 && hop < 2) {
      const distant = distantInNet(net, pt.x, pt.y, pt.z);
      if (distant) {
        netSeed[net].push({
          x: distant.x,
          y: distant.y,
          z: distant.z,
          t0: now + 260 + Math.random() * 280,
          strength: seedStrength * (net === 6 ? 0.58 : 0.46),
        });
        addNodeFlashes(net, distant, 5 + (strength * (net === 6 ? 12 : 8)) | 0, seedStrength * 0.44, 0.22 + Math.random() * 0.18);
      }
    }
    addNodeFlashes(net, pt, 4 + (strength * (net === 6 ? 11 : 7)) | 0, strength);
    if (strength < 0.18 || hop >= 3) return;
    for (let j = 0; j < 7; j++) {
      if (j === net) continue;
      const w = relayWeight(net, j);
      if (w < 0.28) continue;
      const tn = nearestInNet(j, pt.x, pt.y, pt.z); if (!tn) continue;
      const relayBoost = net === 6 && j === 5 ? 1.08 : net === 5 ? 0.9 : 0.78;
      pending.push({ net: j, pt: tn, strength: strength * w * relayBoost, fireAt: now + 140 + (1 - w) * 470 + Math.random() * 150, hop: hop + 1 });
    }
    if (pending.length > 150) pending.splice(0, pending.length - 150);
  }
  function processPending(now) {
    for (let i = pending.length - 1; i >= 0; i--) if (now >= pending[i].fireAt) { const e = pending[i]; pending.splice(i, 1); activate(e.net, e.pt, e.strength, now, e.hop); }
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
    if (pick) activate(net, pick, net === 6 ? 1.18 : net === 5 ? 1.08 : 0.96, now);
    nextSpont = now + 250 + Math.random() * 540;
  }
  function updateActivation(t) {
    for (let i = 0; i < 7; i++) netAct[i] *= i === 6 ? 0.991 : i === 5 ? 0.988 : 0.983;
    for (let i = 0; i < 7; i++) netSeed[i] = netSeed[i].filter((seed) => t - seed.t0 < 2600);
    for (const n of nodes) {
      const seeds = netSeed[n.net];
      const rhythm = reduceMotion ? 0 : (Math.sin(t * (0.00034 + n.net * 0.000035) + n.net * 1.77 + n.parcel * 0.31) + 1) * 0.5;
      const rest = REST_BIAS[n.net] * (0.45 + rhythm * 0.55);
      let target = rest;
      for (const seed of seeds) {
        if (t < seed.t0) continue;
        const age = t - seed.t0;
        const dist = Math.sqrt((n.x - seed.x) ** 2 + (n.y - seed.y) ** 2 + (n.z - seed.z) ** 2);
        const wf = age * 0.00105;
        const spread = NET_SPREAD[n.net];
        const shell = Math.exp(-((dist - wf) ** 2) / (2 * spread * spread)) * Math.exp(-age / 1800);
        const local = Math.exp(-(dist * dist) / (2 * (spread * 0.75) ** 2)) * Math.exp(-age / 620);
        target = Math.max(target, seed.strength * Math.max(shell, local * 0.62));
      }
      n.act += (target - n.act) * (n.net === 6 ? 0.13 : 0.16);
      n.act *= n.net === 6 ? 0.991 : n.net === 5 ? 0.988 : 0.984;
      if (n.act > 2.0) n.act = 2.0; else if (n.act < 0) n.act = 0;
    }
  }

  /* draw */
  function drawParcels(t) {
    for (const p of PARCELS) {
      const q = projectPoint(p.x, p.y, p.z, t);
      const c = COL[p.net];
      const pc = [mix(GREY[0], c[0], 0.012), mix(GREY[1], c[1], 0.012), mix(GREY[2], c[2], 0.012)];
      const fog = 0.25 + 0.55 * clamp((q.depth + 0.7) / 1.4, 0, 1);
      const r = RAD * q.scale * (0.038 + (p.parcel % 3) * 0.006);
      const g = ctx.createRadialGradient(q.sx, q.sy, 0, q.sx, q.sy, r);
      g.addColorStop(0, `rgba(${pc[0] | 0},${pc[1] | 0},${pc[2] | 0},${0.090 * fog})`);
      g.addColorStop(0.58, `rgba(${pc[0] | 0},${pc[1] | 0},${pc[2] | 0},${0.034 * fog})`);
      g.addColorStop(1, `rgba(${pc[0] | 0},${pc[1] | 0},${pc[2] | 0},0)`);
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
      if (e.long && act < 0.09) continue;
      if (act > 0.05) {
        const c = a.act > b.act ? COL[a.net] : COL[b.net];
        col = [mix(GREY[0], c[0], act), mix(GREY[1], c[1], act), mix(GREY[2], c[2], act)];
        alpha = (0.024 + e.w * 0.022) * fog + act * 0.095;
      } else {
        const c = sameNet ? COL[a.net] : GREY;
        const tint = sameNet ? 0.010 : 0;
        col = [mix(GREY[0], c[0], tint), mix(GREY[1], c[1], tint), mix(GREY[2], c[2], tint)];
        alpha = (0.020 + e.w * 0.026) * fog * (sameNet ? 1.0 : 0.22) * (sameParcel ? 1.18 : 1);
      }
      alpha *= e.long ? (e.tract === "commissural" ? 0.22 : 0.34) : 1;
      ctx.strokeStyle = `rgba(${col[0] | 0},${col[1] | 0},${col[2] | 0},${alpha})`;
      ctx.lineWidth = (e.long ? 0.16 : 0.24) + act * (e.long ? 0.22 : 0.30);
      ctx.beginPath(); ctx.moveTo(a.sx, a.sy); ctx.lineTo(b.sx, b.sy); ctx.stroke();
    }
  }
  function drawNodes() {
    order.sort((u, v) => nodes[u].depth - nodes[v].depth);
    for (const k of order) {
      const n = nodes[k], a = n.act, c = COL[n.net];
      const fog = 0.35 + 0.65 * clamp((n.depth + 0.7) / 1.4, 0, 1);
      const baseTint = 0.004 + (n.parcel % 4) * 0.0010;
      const tint = a > 0.05 ? Math.min(1, baseTint + a) : baseTint;
      const col = [mix(GREY[0], c[0], tint) | 0, mix(GREY[1], c[1], tint) | 0, mix(GREY[2], c[2], tint) | 0];
      const r = n.size * n.scale * (1 + a * 0.55);
      if (a > 0.05) { ctx.beginPath(); ctx.arc(n.sx, n.sy, r * 3.4, 0, 6.2832); ctx.fillStyle = `rgba(${col[0]},${col[1]},${col[2]},${a * 0.075 * fog})`; ctx.fill(); }
      ctx.beginPath(); ctx.arc(n.sx, n.sy, r, 0, 6.2832);
      ctx.fillStyle = `rgba(${col[0]},${col[1]},${col[2]},${(0.43 + a * 0.46) * fog})`; ctx.fill();
      if (a < 0.05 && n.parcel % 7 === 0) {
        ctx.beginPath(); ctx.arc(n.sx, n.sy, r * 1.9, 0, 6.2832);
        ctx.strokeStyle = `rgba(${GREY[0]},${GREY[1]},${GREY[2]},${0.098 * fog})`;
        ctx.lineWidth = 0.22;
        ctx.stroke();
      }
    }
  }
  function drawNodeFlashes() {
    nodeFlashes = nodeFlashes.filter((p) => p.t < 1);
    for (const p of nodeFlashes) {
      p.t += p.v;
      const n = nodes[p.i];
      if (!n) continue;
      if (p.t < 0) continue;
      const life = 1 - p.t;
      const fog = 0.35 + 0.65 * clamp((n.depth + 0.7) / 1.4, 0, 1);
      const shimmer = 1 + Math.sin(p.t * 14 + p.phase) * 0.12;
      const r = (n.size * n.scale + p.r) * (0.82 + p.t * 1.9) * shimmer;
      ctx.beginPath();
      ctx.arc(n.sx, n.sy, r, 0, 6.2832);
      ctx.fillStyle = `rgba(${p.c[0]},${p.c[1]},${p.c[2]},${0.20 * life * fog})`;
      ctx.fill();
      ctx.beginPath();
      ctx.arc(n.sx, n.sy, Math.max(0.75, n.size * n.scale * 1.7), 0, 6.2832);
      ctx.fillStyle = `rgba(${p.c[0]},${p.c[1]},${p.c[2]},${0.96 * life * fog})`;
      ctx.fill();
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
    drawNodeFlashes();
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
