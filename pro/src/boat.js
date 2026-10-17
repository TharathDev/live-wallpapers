// Boat geometry for the realistic renderer.
//
// A boat is a list of parts { pos, nrm, uv, idx, mat } in boat-local metres: +x is the bow,
// +y up, +z starboard, and y = 0 is the waterline. Without a 3D model, a motor yacht is built
// procedurally from smooth lofted surfaces; a glTF model (via three.js GLTFLoader) is converted
// into the same parts.

// Materials. `mode` picks procedural detail in the boat shader (see BOAT_RENDER):
// 0 plain, 1 hull paint (antifouling, boot stripe, hull windows), 2 superstructure (window
// band), 3 deck (teak aft, white non-skid forward).
export const MATS = {
  hull:   { base: [0.86, 0.87, 0.88, 1], metal: 0, rough: 0.22, coat: 1, mode: 1 },
  deck:   { base: [0.82, 0.82, 0.80, 1], metal: 0, rough: 0.55, coat: 0, mode: 3 },
  super:  { base: [0.88, 0.89, 0.90, 1], metal: 0, rough: 0.25, coat: 1, mode: 2 },
  glass:  { base: [0.015, 0.02, 0.026, 1], metal: 0, rough: 0.04, coat: 1, mode: 0 },
  teak:   { base: [0.42, 0.27, 0.14, 1], metal: 0, rough: 0.7, coat: 0, mode: 0 },
  chrome: { base: [0.92, 0.93, 0.95, 1], metal: 1, rough: 0.12, coat: 0, mode: 0 },
  cushion:{ base: [0.86, 0.83, 0.76, 1], metal: 0, rough: 0.9, coat: 0, mode: 0 },
  dark:   { base: [0.05, 0.055, 0.06, 1], metal: 0, rough: 0.4, coat: 0.5, mode: 0 },
};

const smooth = (a, b, x) => { const t = Math.min(1, Math.max(0, (x - a) / (b - a))); return t * t * (3 - 2 * t); };

// ---------------------------------------------------------------- mesh builders
class Part {
  constructor(mat) { this.mat = mat; this.p = []; this.n = []; this.t = []; this.i = []; }
  vert(p, n, uv = [0, 0]) { this.p.push(...p); this.n.push(...n); this.t.push(...uv); return this.p.length / 3 - 1; }
  tri(a, b, c) { this.i.push(a, b, c); }
  done() { return { pos: new Float32Array(this.p), nrm: new Float32Array(this.n), uv: new Float32Array(this.t), idx: new Uint32Array(this.i), mat: this.mat }; }
}
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm = v => { const l = Math.hypot(...v); return l > 1e-9 ? [v[0] / l, v[1] / l, v[2] / l] : [0, 1, 0]; };

// A grid surface S(i, j) with smooth normals from its own parametric derivatives.
// `flip` reverses the winding/normal direction.
function gridSurface(part, ni, nj, S, flip = false) {
  const base = part.p.length / 3;
  for (let i = 0; i <= ni; i++) {
    for (let j = 0; j <= nj; j++) {
      const p = S(i, j);
      const du = sub(S(Math.min(ni, i + 1), j), S(Math.max(0, i - 1), j));
      const dv = sub(S(i, Math.min(nj, j + 1)), S(i, Math.max(0, j - 1)));
      let n = norm(cross(du, dv));
      if (flip) n = n.map(v => -v);
      if (!isFinite(n[0])) n = [0, 1, 0];
      part.vert(p, n, [i / ni, j / nj]);
    }
  }
  for (let i = 0; i < ni; i++) {
    for (let j = 0; j < nj; j++) {
      const a = base + i * (nj + 1) + j, b = a + 1, c = a + nj + 1, d = c + 1;
      if (flip) { part.tri(a, b, d); part.tri(a, d, c); } else { part.tri(a, d, b); part.tri(a, c, d); }
    }
  }
}
// Mirror a part across z = 0 (the centreline) and append it.
function mirrorZ(part) {
  const nv = part.p.length / 3, ni = part.i.length;
  for (let v = 0; v < nv; v++) part.vert([part.p[v * 3], part.p[v * 3 + 1], -part.p[v * 3 + 2]], [part.n[v * 3], part.n[v * 3 + 1], -part.n[v * 3 + 2]], [part.t[v * 2], part.t[v * 2 + 1]]);
  for (let k = 0; k < ni; k += 3) part.tri(part.i[k] + nv, part.i[k + 2] + nv, part.i[k + 1] + nv);
}
// Fan cap over a closed outline (outline points in order, centre given).
function fanCap(part, centre, outline, normal) {
  const c = part.vert(centre, normal);
  const vs = outline.map(p => part.vert(p, normal));
  for (let k = 0; k < vs.length - 1; k++) part.tri(c, vs[k], vs[k + 1]);
}
// Rounded box (half extents h, corner radius r), centred at c.
function roundedBox(part, c, h, r, segs = 6) {
  const inner = h.map(v => Math.max(v - r, 1e-4));
  const faces = [[0, 1, 2, 1], [0, 1, 2, -1], [1, 2, 0, 1], [1, 2, 0, -1], [2, 0, 1, 1], [2, 0, 1, -1]];
  for (const [a, b, n, s] of faces) {
    gridSurface(part, segs, segs, (i, j) => {
      const q = [0, 0, 0];
      q[a] = (-1 + 2 * i / segs) * h[a]; q[b] = (-1 + 2 * j / segs) * h[b]; q[n] = s * h[n];
      const cl = q.map((v, k) => Math.max(-inner[k], Math.min(inner[k], v)));
      const d = norm(sub(q, cl));
      return [c[0] + cl[0] + d[0] * r, c[1] + cl[1] + d[1] * r, c[2] + cl[2] + d[2] * r];
    }, s < 0);
  }
}
// Tube along a polyline.
function tube(part, pts, r, segs = 8) {
  for (let k = 0; k < pts.length - 1; k++) {
    const a = pts[k], b = pts[k + 1], ax = norm(sub(b, a));
    const t0 = Math.abs(ax[1]) < 0.9 ? [0, 1, 0] : [1, 0, 0];
    const u = norm(cross(ax, t0)), v = cross(ax, u);
    gridSurface(part, 1, segs, (i, j) => {
      const p = i ? b : a, th = j / segs * Math.PI * 2;
      return [0, 1, 2].map(q => p[q] + (u[q] * Math.cos(th) + v[q] * Math.sin(th)) * r);
    }, true);
  }
}

// ---------------------------------------------------------------- the procedural motor yacht
export const YACHT_DIMS = { LB: 10, LS: 9, HB: 2.6, bowPow: 1.45 };
export function buildYacht() {
  const { LB, LS, HB, bowPow } = YACHT_DIMS;
  const hb = x => (x > 0 ? HB * (1 - Math.pow(Math.min(x / LB, 1), bowPow)) : HB * (1 - 0.06 * Math.pow(Math.min(-x / LS, 1), 2)));
  const sheer = x => 1.6 + 0.9 * Math.pow(smooth(-LS, LB, x), 2);
  const keel = x => -0.9 + Math.pow(Math.max(x - 1, 0), 2) * 0.03;
  const NX = 72, NU = 18;
  const stationX = i => { const s = i / NX; return -LS + (LB + LS) * (1 - Math.pow(1 - s, 1.25)); };   // denser at the bow

  // hull side: keel -> chine -> flared topside -> sheer, starboard half
  const hullPart = new Part(MATS.hull);
  const section = (x, u) => {
    const b = Math.max(hb(x), 0.002), yk = keel(x), ys = sheer(x);
    const bc = b * 0.9, yc = Math.min(yk + bc * 0.32, ys - 0.25);
    if (u < 0.45) {                        // V bottom, slightly convex
      const t = u / 0.45, z = bc * t;
      return [x, yk + (yc - yk) * t - 0.05 * Math.sin(Math.PI * t) * Math.min(1, b), z];
    }
    if (u < 0.55) {                        // rounded chine
      const t = (u - 0.45) / 0.1, z = bc + (b * 0.97 - bc) * Math.sin(t * Math.PI / 2);
      return [x, yc + (0.12) * (1 - Math.cos(t * Math.PI / 2)), z];
    }
    const t = (u - 0.55) / 0.45;           // flared topside up to the sheer
    const y0 = yc + 0.12;
    return [x, y0 + (ys - y0) * t, b * (0.97 + 0.03 * Math.sin(t * Math.PI / 2))];
  };
  gridSurface(hullPart, NX, NU, (i, j) => section(stationX(i), j / NU), true);
  mirrorZ(hullPart);
  // transom
  const tr = [];
  for (let j = 0; j <= NU; j++) tr.push(section(-LS, j / NU));
  const trAll = [...tr.slice().reverse().map(p => [p[0], p[1], -p[2]]), ...tr];
  fanCap(hullPart, [-LS, (keel(-LS) + sheer(-LS)) / 2, 0], trAll, [-1, 0, 0]);

  // deck: from the sheer line in to the centreline, with a little camber
  const deckPart = new Part(MATS.deck);
  gridSurface(deckPart, NX, 6, (i, j) => {
    const x = stationX(i), b = Math.max(hb(x), 0.002), v = j / 6;
    return [x, sheer(x) + 0.07 * Math.sin(v * Math.PI / 2), b * (1 - v)];
  }, false);
  mirrorZ(deckPart);
  // deck edge cap at the transom
  const dk = [];
  for (let j = 0; j <= 6; j++) { const b = hb(-LS), v = j / 6; dk.push([-LS, sheer(-LS) + 0.07 * Math.sin(v * Math.PI / 2), b * (1 - v)]); }
  fanCap(deckPart, [-LS, sheer(-LS) - 0.01, 0], [...dk, ...dk.slice().reverse().map(p => [p[0], p[1], -p[2]])], [-1, 0, 0]);

  // saloon: a lofted rounded-rectangle tube with a raked windscreen
  const superPart = new Part(MATS.super);
  const sx0 = -5.7, sx1 = 4.6, SN = 40, RN = 20;
  const sw = x => Math.min(hb(x) - 0.4, 2.15);
  const stop = x => 3.25 - Math.max(x - 1.6, 0) * 0.62;
  const ring = (x, u) => {                 // u around the half section: deck edge -> side -> roof -> centre
    const w = Math.max(sw(x), 0.05), y0 = sheer(x) + 0.02, y1 = Math.max(stop(x), y0 + 0.08), r = Math.min(0.32, (y1 - y0) * 0.45, w * 0.4);
    const sideLen = (y1 - r) - y0, arc = r * Math.PI / 2, roofLen = w - r, L = sideLen + arc + roofLen;
    let d = u * L;
    if (d < sideLen) return [x, y0 + d, w - 0.04 * (d / sideLen)];                 // slight tumblehome
    d -= sideLen;
    if (d < arc) { const a = d / r; return [x, y1 - r + Math.sin(a) * r, w - 0.04 - r + Math.cos(a) * r]; }
    d -= arc;
    return [x, y1 + 0.03 * (d / roofLen), w - 0.04 - r - d];
  };
  gridSurface(superPart, SN, RN, (i, j) => ring(sx0 + (sx1 - sx0) * i / SN, j / RN), true);
  mirrorZ(superPart);
  for (const [x, nx] of [[sx0, -1], [sx1, 1]]) {
    const ol = []; for (let j = 0; j <= RN; j++) ol.push(ring(x, j / RN));
    fanCap(superPart, [x, (sheer(x) + stop(x)) / 2, 0], [...ol.slice().reverse().map(p => [p[0], p[1], -p[2]]), ...ol], [nx, 0, 0]);
  }
  // flybridge: deck, seating, windscreen, hardtop on posts, radar and mast
  const flyPart = new Part(MATS.super);
  roundedBox(flyPart, [-2.2, 3.36, 0], [3.0, 0.1, 1.95], 0.08);
  roundedBox(flyPart, [-3.0, 4.62, 0], [1.75, 0.07, 1.85], 0.06);
  roundedBox(flyPart, [-3.0, 4.75, 0], [0.32, 0.08, 0.32], 0.08);
  const glassPart = new Part(MATS.glass);
  roundedBox(glassPart, [-3.0, 4.7, 0], [1.35, 0.03, 1.4], 0.03);              // tinted hardtop panel
  roundedBox(glassPart, [2.6, 3.22, 0], [0.9, 0.04, 1.2], 0.04);               // saloon skylight
  roundedBox(glassPart, [0.55, 3.62, 0], [0.12, 0.2, 1.7], 0.05);
  const cushPart = new Part(MATS.cushion);
  roundedBox(cushPart, [-3.3, 3.62, -0.9], [0.9, 0.18, 0.7], 0.12);
  roundedBox(cushPart, [-3.3, 3.62, 0.9], [0.9, 0.18, 0.7], 0.12);
  roundedBox(cushPart, [-0.9, 3.6, 0], [0.45, 0.16, 1.2], 0.12);
  roundedBox(cushPart, [6.2, sheer(6.2) + 0.16, 0], [1.6, 0.13, 1.1], 0.12);      // foredeck sunpad
  roundedBox(cushPart, [-8.3, sheer(-8.3) + 0.3, 0], [0.4, 0.2, 1.9], 0.12);    // cockpit bench
  roundedBox(cushPart, [-6.4, sheer(-6.4) + 0.3, -1.5], [0.9, 0.2, 0.35], 0.12);
  roundedBox(cushPart, [-6.4, sheer(-6.4) + 0.3, 1.5], [0.9, 0.2, 0.35], 0.12);
  const chromePart = new Part(MATS.chrome);
  for (const s of [-1, 1]) {
    tube(chromePart, [[-1.4, 3.45, s * 1.75], [-1.5, 4.55, s * 1.65]], 0.06);
    tube(chromePart, [[-4.6, 3.45, s * 1.75], [-4.5, 4.55, s * 1.65]], 0.06);
    // bow rail with stanchions
    const rail = []; for (let k = 0; k <= 12; k++) { const x = 2.4 + (9.6 - 2.4) * k / 12; rail.push([x, sheer(x) + 0.62, s * Math.max(hb(x) - 0.12, 0.05)]); }
    tube(chromePart, rail, 0.022, 6);
    for (let k = 0; k <= 12; k += 3) tube(chromePart, [[rail[k][0], rail[k][1] - 0.62, rail[k][2]], rail[k]], 0.018, 6);
  }
  tube(chromePart, [[-3.6, 4.7, 0], [-3.6, 5.6, 0]], 0.03, 6);
  // swim platform and cockpit sole in teak
  const teakPart = new Part(MATS.teak);
  roundedBox(teakPart, [-LS - 0.85, 0.42, 0], [0.85, 0.06, HB * 0.88], 0.03);
  roundedBox(teakPart, [-7.0, sheer(-7.0) + 0.42, 0], [0.7, 0.04, 0.55], 0.03);   // cockpit table
  roundedBox(teakPart, [-2.2, 3.48, 0], [0.5, 0.03, 0.45], 0.03);               // flybridge table
  return { parts: [hullPart, deckPart, superPart, flyPart, glassPart, cushPart, chromePart, teakPart].map(p => p.done()), dims: { ...YACHT_DIMS, XB: 1 + Math.sqrt(0.9 / 0.03) } };
}

// ---------------------------------------------------------------- glTF models
// Converts a three.js scene (from GLTFLoader) into boat parts: oriented bow +x, scaled to
// `length`, sunk to `draft` x its hull depth. Returns the parts plus the measured hull.
export function fromGLTF(THREE, scene, opts) {
  scene.updateMatrixWorld(true);
  const raw = [];
  scene.traverse(o => {
    if (!o.isMesh || !o.geometry.attributes.position) return;
    const g = o.geometry.index ? o.geometry : o.geometry;
    const pos = g.attributes.position, nrm = g.attributes.normal, uv = g.attributes.uv;
    const m = o.matrixWorld, nm = new THREE.Matrix3().getNormalMatrix(m);
    const P = new Float32Array(pos.count * 3), Nn = new Float32Array(pos.count * 3), T = new Float32Array(pos.count * 2);
    const v = new THREE.Vector3();
    for (let i = 0; i < pos.count; i++) {
      v.fromBufferAttribute(pos, i).applyMatrix4(m); P.set([v.x, v.y, v.z], i * 3);
      if (nrm) { v.fromBufferAttribute(nrm, i).applyMatrix3(nm).normalize(); Nn.set([v.x, v.y, v.z], i * 3); }
      if (uv) T.set([uv.getX(i), uv.getY(i)], i * 2);
    }
    let idx = g.index ? Uint32Array.from(g.index.array) : Uint32Array.from({ length: pos.count }, (_, i) => i);
    if (m.determinant() < 0) for (let k = 0; k < idx.length; k += 3) [idx[k + 1], idx[k + 2]] = [idx[k + 2], idx[k + 1]];
    if (!nrm) computeNormals(P, idx, Nn);
    const mat = [].concat(o.material)[0] || {};
    raw.push({
      pos: P, nrm: Nn, uv: T, idx,
      mat: {
        base: [mat.color?.r ?? 1, mat.color?.g ?? 1, mat.color?.b ?? 1, mat.opacity ?? 1],
        metal: mat.metalness ?? 0, rough: mat.roughness ?? 0.6, coat: mat.clearcoat ?? 0, mode: 0,
        emissive: mat.emissive ? [mat.emissive.r, mat.emissive.g, mat.emissive.b].map(c => c * (mat.emissiveIntensity ?? 1)) : [0, 0, 0],
        map: mat.map?.image || null,
      },
    });
  });
  if (!raw.length) throw new Error('the model has no meshes');
  // orientation: long axis along x, bow (narrow end of the hull) towards +x
  let bb = bounds(raw);
  if (bb.size[2] > bb.size[0]) transform(raw, (p) => [p[2], p[1], -p[0]]);
  bb = bounds(raw);
  const s = opts.length / bb.size[0];
  transform(raw, p => [(p[0] - bb.centre[0]) * s, (p[1] - bb.min[1]) * s, (p[2] - bb.centre[2]) * s], true);
  bb = bounds(raw);
  const L = bb.size[0], yCut = Math.min(bb.size[1] * 0.3, 4);
  let wF = 0, wB = 0;
  eachVert(raw, p => { if (p[1] > yCut) return; const w = Math.abs(p[2]); if (p[0] > bb.max[0] - 0.15 * L) wF = Math.max(wF, w); if (p[0] < bb.min[0] + 0.15 * L) wB = Math.max(wB, w); });
  if ((wF > wB) !== !!opts.flip) transform(raw, p => [-p[0], p[1], -p[2]]);
  // waterline: a fraction of the hull depth (keel to gunwale, amidships, outboard)
  bb = bounds(raw);
  let halfW = 0, gun = 0;
  const mid = p => Math.abs(p[0]) < 0.15 * L && p[1] < 0.3 * L;
  eachVert(raw, p => { if (mid(p)) halfW = Math.max(halfW, Math.abs(p[2])); });
  eachVert(raw, p => { if (mid(p) && Math.abs(p[2]) > 0.85 * halfW) gun = Math.max(gun, p[1]); });
  const depth = gun > 0 ? gun : bb.size[1] * 0.3;
  const water = opts.waterline ?? opts.draft * depth;
  transform(raw, p => [p[0], p[1] - water, p[2]]);
  // measure the hull at the waterline for the wake and the boat's motion
  let xb = -1e9, xs = 1e9, hbw = 0;
  eachVert(raw, p => { if (p[1] < 0.1) { xb = Math.max(xb, p[0]); xs = Math.min(xs, p[0]); } if (Math.abs(p[1]) < 0.6) hbw = Math.max(hbw, Math.abs(p[2])); });
  if (xb < -1e8) { xb = L * 0.35; xs = -L * 0.45; }
  return { parts: raw, dims: { LB: xb + 0.3, LS: -xs, HB: hbw || bb.size[2] / 2, bowPow: 1.6, XB: xb }, waterline: water };
}
function computeNormals(P, idx, N) {
  for (let k = 0; k < idx.length; k += 3) {
    const a = idx[k] * 3, b = idx[k + 1] * 3, c = idx[k + 2] * 3;
    const n = cross(sub([P[b], P[b + 1], P[b + 2]], [P[a], P[a + 1], P[a + 2]]), sub([P[c], P[c + 1], P[c + 2]], [P[a], P[a + 1], P[a + 2]]));
    for (const v of [a, b, c]) { N[v] += n[0]; N[v + 1] += n[1]; N[v + 2] += n[2]; }
  }
  for (let v = 0; v < N.length; v += 3) { const n = norm([N[v], N[v + 1], N[v + 2]]); N.set(n, v); }
}
function eachVert(parts, f) { for (const pt of parts) for (let i = 0; i < pt.pos.length; i += 3) f([pt.pos[i], pt.pos[i + 1], pt.pos[i + 2]]); }
function transform(parts, f, scaleOnly = false) {
  for (const pt of parts) {
    for (let i = 0; i < pt.pos.length; i += 3) {
      const q = f([pt.pos[i], pt.pos[i + 1], pt.pos[i + 2]]); pt.pos.set(q, i);
      if (!scaleOnly) { const o = f([0, 0, 0]), n = f([pt.nrm[i], pt.nrm[i + 1], pt.nrm[i + 2]]); pt.nrm.set(norm(sub(n, o)), i); }
    }
  }
}
function bounds(parts) {
  const min = [1e9, 1e9, 1e9], max = [-1e9, -1e9, -1e9];
  eachVert(parts, p => { for (let k = 0; k < 3; k++) { min[k] = Math.min(min[k], p[k]); max[k] = Math.max(max[k], p[k]); } });
  return { min, max, size: max.map((v, k) => v - min[k]), centre: max.map((v, k) => (v + min[k]) / 2) };
}
