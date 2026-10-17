(() => {
  // ../pro/src/wgsl/common.js
  var N = 256;
  var LOGN = 8;
  var CASCADES = 6;
  var SKY_W = 256;
  var SKY_H = 144;
  var ENV_SIZE = 256;
  var ENV_MIPS = 9;
  var FRAME_WGSL = (
    /* wgsl */
    `
struct Frame {
  viewProj: mat4x4f,   // camera-relative view-projection (reverse-Z, infinite)
  camPos:   vec4f,     // xyz metres, w = time (s)
  camRight: vec4f,     // xyz * tan(fovX/2)
  camUp:    vec4f,     // xyz * tan(fovY/2)
  camFwd:   vec4f,     // xyz
  sunDir:   vec4f,     // xyz towards sun, w = solar illuminance scale
  screen:   vec4f,     // w, h, 1/w, 1/h
  cascL:    array<vec4f, 2>,  // cascade patch sizes (m), index [c>>2][c&3]
  cascInvL: array<vec4f, 2>,
  windP:    array<vec4f, 2>,  // [0].xy = wind direction (unit), [0].z = wind speed
  lodP:     vec4f,     // CDLOD base range, grid quads per node, -, -
  post:     vec4f,     // exposure, bloom strength, frame index, -
  atmo:     vec4f,     // haze (mie multiplier), camera altitude km, -, -
  waterCol: vec4f,     // body scatter colour rgb, unresolved slope variance
  sssCol:   vec4f,     // sss colour rgb, sss strength
  foamP:    vec4f,     // J threshold, J gain, foam intensity, lace sharpness
  fogExt:   vec4f,     // rgb extinction per metre near sea level
  cloudP:   vec4f,     // altitude km, coverage, optical density, wind time
};
`
  );
  var ATMO_CORE_WGSL = (
    /* wgsl */
    `
const PI: f32 = 3.14159265358979;
const RG: f32 = 6360.0;
const RT: f32 = 6460.0;
const RAY_S: vec3f = vec3f(5.802e-3, 13.558e-3, 33.1e-3);
const MIE_S: f32 = 3.996e-3;
const MIE_E: f32 = 4.40e-3;
const OZO_A: vec3f = vec3f(0.650e-3, 1.881e-3, 0.085e-3);
const SUN_ANG_R: f32 = 0.004675;
const TRANS_W: f32 = 256.0;
const TRANS_H: f32 = 64.0;
const MS_SIZE: f32 = 32.0;

struct Medium { ext: vec3f, rayS: vec3f, mieS: f32 };

fn medium(h: f32, haze: f32) -> Medium {
  let hh = max(h, 0.0);
  let dR = exp(-hh / 8.0);
  // aerosols: regular 1.2 km layer plus a dense marine boundary layer
  let dM = exp(-hh / 1.2);
  let dO = max(0.0, 1.0 - abs(hh - 25.0) / 15.0);
  var m: Medium;
  m.rayS = RAY_S * dR;
  m.mieS = MIE_S * dM * haze;
  m.ext = m.rayS + vec3f(MIE_E * dM * haze) + OZO_A * dO;
  return m;
}

fn raySphere(ro: vec3f, rd: vec3f, R: f32) -> vec2f {
  let l = length(ro);
  let b = dot(ro, rd);
  let c = (l - R) * (l + R);
  let d = b * b - c;
  if (d < 0.0) { return vec2f(-1.0, -1.0); }
  let s = sqrt(d);
  return vec2f(-b - s, -b + s);
}

fn rhoOf(r: f32) -> f32 { return sqrt(max(0.0, (r - RG) * (r + RG))); }

fn transRMuToUV(r: f32, mu: f32) -> vec2f {
  let H = sqrt(RT * RT - RG * RG);
  let rho = rhoOf(r);
  let disc = r * r * (mu * mu - 1.0) + RT * RT;
  let d = max(0.0, -r * mu + sqrt(max(disc, 0.0)));
  let dMin = RT - r;
  let dMax = rho + H;
  let x = (d - dMin) / (dMax - dMin);
  let y = rho / H;
  return vec2f(x, y);
}

fn transUVToRMu(uv: vec2f) -> vec2f {
  let H = sqrt(RT * RT - RG * RG);
  let rho = H * uv.y;
  let r = sqrt(rho * rho + RG * RG);
  let dMin = RT - r;
  let dMax = rho + H;
  let d = dMin + uv.x * (dMax - dMin);
  var mu = 1.0;
  if (d > 0.0) { mu = (H * H - rho * rho - d * d) / (2.0 * r * d); }
  return vec2f(r, clamp(mu, -1.0, 1.0));
}

fn fromUnitToSubUvs(u: f32, res: f32) -> f32 { return (u + 0.5 / res) * (res / (res + 1.0)); }
fn fromSubUvsToUnit(u: f32, res: f32) -> f32 { return (u - 0.5 / res) * (res / (res - 1.0)); }

fn phaseRayleigh(c: f32) -> f32 { return 3.0 / (16.0 * PI) * (1.0 + c * c); }
fn phaseMie(c: f32, g: f32) -> f32 {
  let g2 = g * g;
  return 3.0 / (8.0 * PI) * ((1.0 - g2) * (1.0 + c * c)) / ((2.0 + g2) * pow(max(1.0 + g2 - 2.0 * g * c, 1e-4), 1.5));
}
fn phaseHG(c: f32, g: f32) -> f32 {
  let g2 = g * g;
  return (1.0 - g2) / (4.0 * PI * pow(max(1.0 + g2 - 2.0 * g * c, 1e-4), 1.5));
}

// true when the sun is below the geometric horizon of a point at radius r
fn sunBelowHorizon(r: f32, mu: f32) -> bool {
  return mu < -rhoOf(r) / r;
}
`
  );
  var NOISE_WGSL = (
    /* wgsl */
    `
fn pcgHash(v: u32) -> u32 {
  let s = v * 747796405u + 2891336453u;
  let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  return (w >> 22u) ^ w;
}
fn hashGrad(i: vec2i) -> vec2f {
  let h = pcgHash(u32(i.x) * 1664525u + pcgHash(u32(i.y) + 1013904223u));
  let a = f32(h) * (6.28318530718 / 4294967296.0);
  return vec2f(cos(a), sin(a));
}
fn gnoise(p: vec2f) -> f32 {
  let i = vec2i(floor(p));
  let f = p - floor(p);
  let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
  let a = dot(hashGrad(i), f);
  let b = dot(hashGrad(i + vec2i(1, 0)), f - vec2f(1.0, 0.0));
  let c = dot(hashGrad(i + vec2i(0, 1)), f - vec2f(0.0, 1.0));
  let d = dot(hashGrad(i + vec2i(1, 1)), f - vec2f(1.0, 1.0));
  return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}
`
  );
  var PERIODIC_WGSL = (
    /* wgsl */
    `
fn pnoise(p: vec2f, period: i32) -> f32 {
  let i = vec2i(floor(p));
  let f = p - floor(p);
  let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
  let i00 = (i % period + period) % period;
  let i10 = ((i + vec2i(1, 0)) % period + period) % period;
  let i01 = ((i + vec2i(0, 1)) % period + period) % period;
  let i11 = ((i + vec2i(1, 1)) % period + period) % period;
  let a = dot(hashGrad(i00), f);
  let b = dot(hashGrad(i10), f - vec2f(1.0, 0.0));
  let c = dot(hashGrad(i01), f - vec2f(0.0, 1.0));
  let d = dot(hashGrad(i11), f - vec2f(1.0, 1.0));
  return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}
fn pfbm(p: vec2f, period: i32, oct: i32) -> f32 {
  var s = 0.0; var a = 0.5; var q = p; var per = period;
  for (var o = 0; o < oct; o++) {
    s += a * pnoise(q, per);
    q *= 2.0; per *= 2; a *= 0.5;
  }
  return s;
}
`
  );
  var BOAT_WGSL = (
    /* wgsl */
    `
struct Boat {
  model:    mat4x4f,  // boat-local -> world (absolute metres; the boat heads +x, +z is starboard)
  invModel: mat4x4f,
  dims:     vec4f,    // bow x, stern distance, half beam, x where the bow meets the water
  wake:     vec4f,    // speed m/s, wake strength, hull displacement on/off, twin screws
  plan:     vec4f,    // bow plan power, dt, wake foam decay 1/s, -
};
fn hullHW(x: f32) -> f32 {
  if (x > 0.0) { return B.dims.z * (1.0 - pow(clamp(x / B.dims.x, 0.0, 1.0), B.plan.x)); }
  return B.dims.z * (1.0 - 0.06 * pow(clamp(-x / B.dims.y, 0.0, 1.0), 2.0));
}
fn boatLocal(p: vec2f) -> vec2f { return (B.invModel * vec4f(p.x, 0.0, p.y, 1.0)).xz; }
// Bow wave, Kelvin wake arms and transverse wake waves, as a height (m) in boat-local x/z.
fn kelvin(lp: vec2f) -> f32 {
  let LS = B.dims.y; let HB = B.dims.z; let XB = B.dims.w; let V = max(B.wake.x, 0.5);
  let sb = XB - lp.x;
  if (sb < -2.5 || sb > 250.0) { return 0.0; }
  let az = abs(lp.y);
  let arm = HB * 0.85 + max(sb, 0.0) * mix(0.33, 0.2, smoothstep(6.0, 12.0, V));
  let armW = 0.45 + max(sb, 0.0) * 0.045;
  let decay = exp(-max(sb, 0.0) * 0.025) * smoothstep(-2.5, 0.5, sb);
  var h = 0.32 * exp(-pow((az - arm) / armW, 2.0)) * decay;
  h -= 0.15 * exp(-pow((az - arm + armW * 1.9) / (armW * 1.3), 2.0)) * decay * smoothstep(0.0, 3.0, sb);
  let inside = smoothstep(arm, arm - 4.0, az);
  let lam = 6.2831853 * V * V / 9.81;
  h += 0.1 * cos(sb * 6.2831853 / lam) * inside * exp(-sb * 0.02) * smoothstep(5.0, 14.0, sb);
  let ss = -LS - lp.x;
  if (ss > -1.0) { h += 0.15 * exp(-pow(lp.y / (HB * 0.8 + max(ss, 0.0) * 0.06), 2.0)) * exp(-max(ss, 0.0) * 0.07) * smoothstep(-1.0, 1.0, ss); }
  return h * min(B.wake.y, 1.3);
}
// The boat's white water, fixed in the boat's frame: x = churn along the hull and in the prop
// wash (dense), y = the long fading trail and the bow-wave lines (laced by the foam texture),
// z = air-filled turquoise water under it. The boat moves at constant speed, so the trail's age
// is just its distance behind the boat divided by the speed.
fn wakeFoam(lp: vec2f) -> vec3f {
  if (B.wake.y <= 0.0) { return vec3f(0.0); }
  let LB = B.dims.x; let LS = B.dims.y; let HB = B.dims.z; let XB = B.dims.w; let V = max(B.wake.x, 0.5);
  let az = abs(lp.y);
  var churn = 0.0; var trail = 0.0; var aer = 0.0;
  if (lp.x > -LS - 0.5 && lp.x < XB + 0.6) {
    // white water pushed off the hull, strongest at the bow
    let d = az - hullHW(clamp(lp.x, -LS, XB)) * 0.92;
    churn = exp(-max(d, 0.0) * 5.0) * step(-0.5, d) * mix(0.25, 0.9, smoothstep(-LS, XB, lp.x));
  }
  let ss = -LS - lp.x;
  if (ss > -0.5) {
    let age = max(ss, 0.0) / V;                              // seconds since the boat passed
    let fade = exp(-age * B.plan.z);
    let twin = B.wake.w > 0.5;
    let sep = select(0.0, HB * (0.42 + 0.004 * min(max(ss, 0.0), 80.0)), twin);
    let cz = abs(az - sep);
    let w = select(HB * 0.55, HB * 0.24, twin) + max(ss, 0.0) * 0.028;
    let core = exp(-pow(cz / w, 2.0)) * smoothstep(-0.5, 0.4, ss);
    // the prop wash is solid white just behind the transom, then breaks up into a lacy trail
    churn = max(churn, core * exp(-max(ss, 0.0) * 0.2));
    trail = core * fade * 0.75;
    // the whole wake band (both streams and between) stays aerated a while longer
    let band = exp(-pow(az / (HB * 1.1 + max(ss, 0.0) * 0.045), 2.0)) * smoothstep(-0.5, 0.4, ss);
    aer = band * exp(-age * B.plan.z * 0.6);
  }
  // thin foam lines along the bow-wave arms
  let sb = XB - lp.x;
  if (sb > 0.0 && sb < 120.0) {
    let arm = HB * 0.9 + sb * mix(0.33, 0.2, smoothstep(6.0, 12.0, V));
    let line = exp(-pow((az - arm) / (0.25 + sb * 0.02), 2.0)) * exp(-sb / V * B.plan.z * 3.0);
    trail = max(trail, line * 0.42);
  }
  return vec3f(churn, trail, aer) * min(B.wake.y, 1.3);
}
`
  );

  // ../pro/src/wgsl/sim.js
  var SIM_STRUCT = (
    /* wgsl */
    `
const N: u32 = ${N}u;
const LOGN: u32 = ${LOGN}u;
const PI: f32 = 3.14159265358979;
const GRAV: f32 = 9.81;
const SURF: f32 = 7.4e-5;   // surface tension / density

struct Sim {
  L:          array<vec4f, 2>,  // patch size per cascade ([c>>2][c&3])
  kMin:       array<vec4f, 2>,
  kMax:       array<vec4f, 2>,
  foamDecay:  array<vec4f, 2>,  // 1/s per cascade
  foamThresh: array<vec4f, 2>,  // Jacobian threshold per cascade
  wind:       vec4f,  // U10 m/s, direction rad, fetch m, gamma
  swell:      vec4f,  // Hs m, direction rad, peak omega, spreading exponent
  misc:       vec4f,  // choppiness, short wave cut (m), loop period s, seed
  timeP:      vec4f,  // time (looped), dt, foam gain, spectrum gain
};
@group(0) @binding(0) var<uniform> S: Sim;

fn waveVec(x: u32, y: u32, L: f32) -> vec2f {
  let nx = select(i32(x), i32(x) - i32(N), x >= N / 2u);
  let ny = select(i32(y), i32(y) - i32(N), y >= N / 2u);
  return vec2f(f32(nx), f32(ny)) * (2.0 * PI / L);
}
fn omegaOf(k: f32) -> f32 { return sqrt(GRAV * k + SURF * k * k * k); }
fn dOmegaOf(k: f32) -> f32 { return (GRAV + 3.0 * SURF * k * k) / (2.0 * omegaOf(k)); }
fn cmul(a: vec2f, b: vec2f) -> vec2f { return vec2f(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x); }
fn mulI(a: vec2f) -> vec2f { return vec2f(-a.y, a.x); }
`
  );
  var SIM_INIT = SIM_STRUCT + NOISE_WGSL + /* wgsl */
  `
@group(0) @binding(1) var<storage, read_write> h0: array<vec4f>;

fn rand01(s: u32) -> f32 { return (f32(pcgHash(s) >> 8u) + 0.5) / 16777216.0; }
fn gauss2(seed: u32) -> vec2f {
  let u1 = rand01(seed * 2u + 17u);
  let u2 = rand01(seed * 2u + 18u + pcgHash(seed));
  let r = sqrt(-2.0 * log(u1));
  let a = 2.0 * PI * u2;
  return r * vec2f(cos(a), sin(a));
}
fn wrapAngle(a: f32) -> f32 {
  var x = a;
  x = x - 2.0 * PI * floor((x + PI) / (2.0 * PI));
  return x;
}
fn jonswap(w: f32, wp: f32, alpha: f32, gamma: f32) -> f32 {
  let sigma = select(0.09, 0.07, w <= wp);
  let d = (w - wp) / (sigma * wp);
  let r = exp(-0.5 * d * d);
  let q = wp / w;
  let q2 = q * q;
  return alpha * GRAV * GRAV / (w * w * w * w * w) * exp(-1.25 * q2 * q2) * pow(gamma, r);
}
// Donelan-Banner directional spreading (normalised over [-pi, pi])
fn donelanBanner(theta: f32, w: f32, wp: f32) -> f32 {
  let ratio = w / wp;
  var beta: f32;
  if (ratio < 0.95) {
    beta = 2.61 * pow(ratio, 1.3);
  } else if (ratio < 1.6) {
    beta = 2.28 * pow(ratio, -1.3);
  } else {
    let eps = -0.4 + 0.8393 * exp(-0.567 * log(ratio * ratio));
    beta = pow(10.0, eps);
  }
  let sh = 1.0 / cosh(clamp(beta * theta, -20.0, 20.0));
  return beta / (2.0 * tanh(beta * PI)) * sh * sh;
}
// cos^2s(theta/2) spreading, normalised
fn cosSpread(theta: f32, s: f32) -> f32 {
  let q = sqrt(s + 0.25) / (2.0 * sqrt(PI));
  return q * pow(abs(cos(theta * 0.5)), 2.0 * s);
}

fn spectrum(k: vec2f) -> f32 {
  let kl = length(k);
  let w = omegaOf(kl);
  let theta = atan2(k.y, k.x);
  // fetch-limited wind sea (JONSWAP)
  let U = S.wind.x;
  let F = S.wind.z;
  var Sw = 0.0;
  if (U > 0.1) {
    let alpha = 0.076 * pow(U * U / (F * GRAV), 0.22);
    let wp = 22.0 * pow(GRAV * GRAV / (U * F), 1.0 / 3.0);
    Sw = jonswap(w, wp, alpha, S.wind.w) * donelanBanner(wrapAngle(theta - S.wind.y), w, wp);
  }
  // swell: narrow, long-crested JONSWAP
  if (S.swell.x > 0.001) {
    let ws = S.swell.z;
    let gs = 6.0;
    let hs4 = S.swell.x * 0.25;
    let alphaS = hs4 * hs4 * 5.0 * ws * ws * ws * ws / (GRAV * GRAV) * (1.0 - 0.287 * log(gs));
    Sw += jonswap(w, ws, alphaS, gs) * cosSpread(wrapAngle(theta - S.swell.y), S.swell.w);
  }
  // damp the very shortest waves
  let sc = S.misc.y;
  Sw *= exp(-kl * kl * sc * sc);
  return Sw * dOmegaOf(kl) / kl * S.timeP.w;
}

fn amplitude(k: vec2f, c: u32) -> f32 {
  let kl = length(k);
  if (kl < 1e-5 || kl < S.kMin[c >> 2u][c & 3u] || kl >= S.kMax[c >> 2u][c & 3u]) { return 0.0; }
  let dk = 2.0 * PI / S.L[c >> 2u][c & 3u];
  // the finest cascades are toned down (they otherwise read as crumpled foil)
  let g = select(1.0, select(0.85, 0.6, c == 5u), c >= 4u);
  return sqrt(max(spectrum(k), 0.0)) * dk * 0.5 * g;
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let c = id.z;
  let L = S.L[c >> 2u][c & 3u];
  let k = waveVec(id.x, id.y, L);
  let mx = (N - id.x) % N;
  let my = (N - id.y) % N;
  let seed = u32(S.misc.w) * 7919u + c * 1000003u;
  let a = gauss2(seed + (id.y * N + id.x)) * amplitude(k, c);
  let b = gauss2(seed + (my * N + mx)) * amplitude(-k, c);
  h0[(c * N + id.y) * N + id.x] = vec4f(a, b.x, -b.y);
}
`;
  var SIM_EVOLVE = SIM_STRUCT + /* wgsl */
  `
@group(0) @binding(1) var<storage, read> h0: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> specA: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> specB: array<vec4f>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let c = id.z;
  let idx = (c * N + id.y) * N + id.x;
  let k = waveVec(id.x, id.y, S.L[c >> 2u][c & 3u]);
  let kl = length(k);
  if (kl < 1e-5) {
    specA[idx] = vec4f(0.0);
    specB[idx] = vec4f(0.0);
    return;
  }
  let w0 = 2.0 * PI / S.misc.z;
  let w = floor(omegaOf(kl) / w0) * w0;
  let ph = w * S.timeP.x;
  let e = vec2f(cos(ph), sin(ph));
  let H0 = h0[idx];
  // h0(k) travels along +k: h(k,t) = h0(k) e^{-iwt} + conj(h0(-k)) e^{iwt}
  let h = cmul(H0.xy, vec2f(e.x, -e.y)) + cmul(H0.zw, e);
  let ih = mulI(h);
  let lam = S.misc.x;
  let ik = 1.0 / kl;
  let Dx = ih * (lam * k.x * ik);
  let Dz = ih * (lam * k.y * ik);
  let jxz = h * (-lam * k.x * k.y * ik);
  let sx = ih * k.x;
  let sz = ih * k.y;
  let jxx = h * (-lam * k.x * k.x * ik);
  let jzz = h * (-lam * k.y * k.y * ik);
  specA[idx] = vec4f(Dx + mulI(Dz), h + mulI(jxz));
  specB[idx] = vec4f(sx + mulI(sz), jxx + mulI(jzz));
}
`;
  var FFT_CORE = (
    /* wgsl */
    `
const N: u32 = ${N}u;
const LOGN: u32 = ${LOGN}u;
const PI: f32 = 3.14159265358979;
var<workgroup> wa: array<vec4f, ${N}>;
var<workgroup> wb: array<vec4f, ${N}>;
fn cmul(a: vec2f, b: vec2f) -> vec2f { return vec2f(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x); }
fn bitrev(i: u32) -> u32 { return reverseBits(i) >> (32u - LOGN); }
// In-place radix-2 inverse FFT over shared memory (input already bit-reversed).
fn butterflies(t: u32) {
  for (var s = 0u; s < LOGN; s++) {
    let half = 1u << s;
    let pos = t & (half - 1u);
    let i0 = ((t >> s) << (s + 1u)) + pos;
    let i1 = i0 + half;
    let ang = PI * f32(pos) / f32(half);
    let w = vec2f(cos(ang), sin(ang));
    let a0 = wa[i0]; let a1 = wa[i1];
    let b0 = wb[i0]; let b1 = wb[i1];
    let ta = vec4f(cmul(a1.xy, w), cmul(a1.zw, w));
    let tb = vec4f(cmul(b1.xy, w), cmul(b1.zw, w));
    wa[i0] = a0 + ta; wa[i1] = a0 - ta;
    wb[i0] = b0 + tb; wb[i1] = b0 - tb;
    workgroupBarrier();
  }
}
`
  );
  var FFT_ROWS = FFT_CORE + /* wgsl */
  `
@group(0) @binding(0) var<storage, read> inA: array<vec4f>;
@group(0) @binding(1) var<storage, read> inB: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> outA: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> outB: array<vec4f>;

@compute @workgroup_size(${N / 2})
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) t: u32) {
  let base = (wg.z * N + wg.x) * N;
  let e0 = t;
  let e1 = t + N / 2u;
  wa[bitrev(e0)] = inA[base + e0];
  wa[bitrev(e1)] = inA[base + e1];
  wb[bitrev(e0)] = inB[base + e0];
  wb[bitrev(e1)] = inB[base + e1];
  workgroupBarrier();
  butterflies(t);
  outA[base + e0] = wa[e0];
  outA[base + e1] = wa[e1];
  outB[base + e0] = wb[e0];
  outB[base + e1] = wb[e1];
}
`;
  var FFT_COLS = FFT_CORE + /* wgsl */
  `
@group(0) @binding(0) var<storage, read> inA: array<vec4f>;
@group(0) @binding(1) var<storage, read> inB: array<vec4f>;
@group(0) @binding(2) var dispOut: texture_storage_2d_array<rgba16float, write>;
@group(0) @binding(3) var derivOut: texture_storage_2d_array<rgba16float, write>;
@group(0) @binding(4) var momOut: texture_storage_2d_array<rgba16float, write>;

fn emit(x: u32, y: u32, c: u32, a: vec4f, b: vec4f) {
  // a = (Dx, Dz, Dy, jxz), b = (sx, sz, jxx, jzz)
  let p = vec2u(x, y);
  textureStore(dispOut, p, c, vec4f(a.x, a.z, a.y, a.w));
  textureStore(derivOut, p, c, b);
  textureStore(momOut, p, c, vec4f(b.x * b.x, b.y * b.y, b.x * b.y, 0.0));
}

@compute @workgroup_size(${N / 2})
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) t: u32) {
  let x = wg.x;
  let c = wg.z;
  let e0 = t;
  let e1 = t + N / 2u;
  let i0 = (c * N + e0) * N + x;
  let i1 = (c * N + e1) * N + x;
  wa[bitrev(e0)] = inA[i0];
  wa[bitrev(e1)] = inA[i1];
  wb[bitrev(e0)] = inB[i0];
  wb[bitrev(e1)] = inB[i1];
  workgroupBarrier();
  butterflies(t);
  emit(x, e0, c, wa[e0], wb[e0]);
  emit(x, e1, c, wa[e1], wb[e1]);
}
`;
  var SIM_MIPS = (
    /* wgsl */
    `
@group(0) @binding(0) var srcA: texture_2d_array<f32>;
@group(0) @binding(1) var srcB: texture_2d_array<f32>;
@group(0) @binding(2) var srcC: texture_2d_array<f32>;
@group(0) @binding(3) var dstA: texture_storage_2d_array<rgba16float, write>;
@group(0) @binding(4) var dstB: texture_storage_2d_array<rgba16float, write>;
@group(0) @binding(5) var dstC: texture_storage_2d_array<rgba16float, write>;

fn box(t: texture_2d_array<f32>, p: vec2i, l: u32) -> vec4f {
  return 0.25 * (textureLoad(t, p, l, 0) + textureLoad(t, p + vec2i(1, 0), l, 0) +
                 textureLoad(t, p + vec2i(0, 1), l, 0) + textureLoad(t, p + vec2i(1, 1), l, 0));
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let size = textureDimensions(dstA);
  if (id.x >= size.x || id.y >= size.y) { return; }
  let p = vec2i(id.xy) * 2;
  textureStore(dstA, id.xy, id.z, box(srcA, p, id.z));
  textureStore(dstB, id.xy, id.z, box(srcB, p, id.z));
  textureStore(dstC, id.xy, id.z, box(srcC, p, id.z));
}
`
  );
  var FOAM_TEX = NOISE_WGSL + PERIODIC_WGSL + /* wgsl */
  `
@group(0) @binding(0) var dst: texture_storage_2d<rgba8unorm, write>;
const SIZE: f32 = 512.0;

fn h2(i: vec2i, salt: u32) -> vec2f {
  let h = pcgHash((u32(i.x) * 73856093u) ^ pcgHash(u32(i.y) * 19349663u + salt));
  let h2v = pcgHash(h + 0x9e3779b9u);
  return vec2f(f32(h >> 8u), f32(h2v >> 8u)) / 16777216.0;
}
// "swiss cheese" foam: 1 in foam material, 0 inside bubbles; per-bubble radius
fn bubbles(uv: vec2f, period: i32, salt: u32, rmin: f32, rmax: f32) -> f32 {
  let w = vec2f(pfbm(uv * f32(period) * 0.5, period / 2, 3), pfbm(uv * f32(period) * 0.5 + 17.0, period / 2, 3));
  let p = uv * f32(period) + w * 0.9;
  let ip = vec2i(floor(p));
  let fp = p - floor(p);
  var v = 9.0;
  for (var j = -1; j <= 1; j++) {
    for (var i = -1; i <= 1; i++) {
      let o = vec2i(i, j);
      let cell = ((ip + o) % period + period) % period;
      let hh = h2(cell, salt);
      let r = vec2f(o) + hh * 0.9 + 0.05 - fp;
      let rad = mix(rmin, rmax, h2(cell, salt + 7u).x);
      v = min(v, length(r) / rad);
    }
  }
  return smoothstep(0.8, 1.25, v);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let uv = (vec2f(id.xy) + 0.5) / SIZE;
  // irregular holes from warped fbm (big sheets with ragged openings) ...
  let w1 = vec2f(pfbm(uv * 5.0, 5, 3), pfbm(uv * 5.0 + vec2f(3.7, 9.1), 5, 3));
  let m1 = smoothstep(-0.14, 0.06, pfbm(uv * 10.0 + w1 * 1.4, 10, 5));
  let w2 = vec2f(pfbm(uv * 12.0 + 1.3, 12, 2), pfbm(uv * 12.0 + 7.9, 12, 2));
  let m2 = smoothstep(-0.16, 0.04, pfbm(uv * 27.0 + w2 * 0.9, 27, 4));
  // ... plus small round bubbles
  let b3 = bubbles(uv, 29, 37u, 0.06, 0.45);
  let b4 = bubbles(uv, 61, 53u, 0.05, 0.42);
  let dens = pfbm(uv * 4.0, 4, 5) * 0.5 + 0.5;
  let fineD = pfbm(uv * 16.0, 16, 3) * 0.5 + 0.5;
  let big = m1 * (0.55 + 0.45 * m2) * (0.75 + 0.25 * b3);
  let fine = 0.45 + 0.55 * b3 * (0.7 + 0.3 * b4);
  let streak = pfbm(uv * vec2f(4.0, 16.0), 4, 5) * 0.5 + 0.5;
  textureStore(dst, id.xy, vec4f(clamp(big * 0.6 + dens * 0.4, 0.0, 1.0), clamp(fine * 0.8 + fineD * 0.2, 0.0, 1.0), dens, streak));
}
`;
  var RGBA8_MIPS = (
    /* wgsl */
    `
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var dst: texture_storage_2d<rgba8unorm, write>;
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let size = textureDimensions(dst);
  if (id.x >= size.x || id.y >= size.y) { return; }
  let p = vec2i(id.xy) * 2;
  let v = 0.25 * (textureLoad(src, p, 0) + textureLoad(src, p + vec2i(1, 0), 0) +
                  textureLoad(src, p + vec2i(0, 1), 0) + textureLoad(src, p + vec2i(1, 1), 0));
  textureStore(dst, id.xy, v);
}
`
  );
  var FOAM_RES = 1024;
  var FOAM_UPDATE = (
    /* wgsl */
    `
const RES: i32 = ${FOAM_RES};
struct FoamU {
  snap: array<vec4i, 2>,     // per level: current camera texel (x, z), previous (x, z)
  p0: vec4f,                 // texel size level 0, level 1, dt, decay rate
  p1: vec4f,                 // J threshold, gain, growth, -
  cascL: array<vec4f, 2>,
  cascInvL: array<vec4f, 2>,
  drift: vec4f,              // foam drift velocity (m/s) xz
};
@group(0) @binding(0) var<uniform> U: FoamU;
@group(0) @binding(1) var prevTex: texture_2d_array<f32>;
@group(0) @binding(2) var dst: texture_storage_2d_array<rgba16float, write>;
@group(0) @binding(3) var dispTex: texture_2d_array<f32>;
@group(0) @binding(4) var derivTex: texture_2d_array<f32>;
@group(0) @binding(5) var samp: sampler;
@group(0) @binding(6) var sampPrev: sampler;

fn wrapIdx(i: i32, lo: i32) -> i32 { return lo + (((i - lo) % RES) + RES) % RES; }

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let lvl = id.z;
  let sn = U.snap[lvl];
  let ts = select(U.p0.x, U.p0.y, lvl == 1u);
  let lo = sn.xy - vec2i(RES / 2);
  let g = vec2i(wrapIdx(i32(id.x), lo.x), wrapIdx(i32(id.y), lo.y));
  let plo = sn.zw - vec2i(RES / 2);
  let inPrev = all(g >= plo) && all(g < plo + vec2i(RES));
  let p = (vec2f(g) + 0.5) * ts;
  var jxx = 0.0; var jzz = 0.0; var jxz = 0.0;
  for (var c = 0u; c < ${CASCADES}u; c++) {
    let texel = U.cascL[c >> 2u][c & 3u] / ${N}.0;
    let lod = log2(max(ts / texel, 1.0));
    if (lod < 7.5) {
      let uv = p * U.cascInvL[c >> 2u][c & 3u];
      let d = textureSampleLevel(derivTex, samp, uv, c, lod);
      let a = textureSampleLevel(dispTex, samp, uv, c, lod);
      jxx += d.z; jzz += d.w; jxz += a.w;
    }
  }
  let J = (1.0 + jxx) * (1.0 + jzz) - jxz * jxz;
  var prev = vec4f(0.0);
  if (inPrev) {
    // semi-Lagrangian advection: foam slowly drifts downwind (and spreads a little)
    let back = (vec2f(id.xy) + 0.5 - U.drift.xy * U.p0.z / ts) / f32(RES);
    prev = textureSampleLevel(prevTex, sampPrev, back, lvl, 0.0);
  }
  let dt = U.p0.z;
  let inject = clamp((U.p1.x - J) * U.p1.y, 0.0, 1.0);
  var f = prev.r * exp(-dt * U.p0.w);
  f = max(f, inject);
  f = min(f + inject * dt * U.p1.z, 1.6);
  // g: freshness (recently injected) used for brighter, denser crest foam
  let fresh = max(prev.g * exp(-dt * 1.6), inject);
  textureStore(dst, id.xy, lvl, vec4f(f, fresh, J, 1.0));
}
`
  );
  var ARRAY16_MIPS = (
    /* wgsl */
    `
@group(0) @binding(0) var src: texture_2d_array<f32>;
@group(0) @binding(1) var dst: texture_storage_2d_array<rgba16float, write>;
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let size = textureDimensions(dst);
  if (id.x >= size.x || id.y >= size.y) { return; }
  let p = vec2i(id.xy) * 2;
  let v = 0.25 * (textureLoad(src, p, id.z, 0) + textureLoad(src, p + vec2i(1, 0), id.z, 0) +
                  textureLoad(src, p + vec2i(0, 1), id.z, 0) + textureLoad(src, p + vec2i(1, 1), id.z, 0));
  textureStore(dst, id.xy, id.z, v);
}
`
  );

  // ../pro/src/wgsl/sky.js
  var ATMO_UNIFORM = (
    /* wgsl */
    `
struct AtmoP { haze: vec4f };   // x = mie multiplier
@group(0) @binding(0) var<uniform> A: AtmoP;
`
  );
  var TRANS_LUT = ATMO_CORE_WGSL + ATMO_UNIFORM + /* wgsl */
  `
@group(0) @binding(1) var dst: texture_storage_2d<rgba16float, write>;
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3u) {
  if (f32(id.x) >= TRANS_W || f32(id.y) >= TRANS_H) { return; }
  let uv = (vec2f(id.xy) + 0.5) / vec2f(TRANS_W, TRANS_H);
  let rm = transUVToRMu(uv);
  let ro = vec3f(0.0, rm.x, 0.0);
  let rd = vec3f(sqrt(max(0.0, 1.0 - rm.y * rm.y)), rm.y, 0.0);
  let tMax = raySphere(ro, rd, RT).y;
  var od = vec3f(0.0);
  let STEPS = 64u;
  let dt = tMax / f32(STEPS);
  for (var i = 0u; i < STEPS; i++) {
    let p = ro + rd * ((f32(i) + 0.5) * dt);
    od += medium(length(p) - RG, A.haze.x).ext * dt;
  }
  textureStore(dst, id.xy, vec4f(exp(-od), 1.0));
}
`;
  var MS_LUT = ATMO_CORE_WGSL + ATMO_UNIFORM + /* wgsl */
  `
@group(0) @binding(1) var transLUT: texture_2d<f32>;
@group(0) @binding(2) var sampC: sampler;
@group(0) @binding(3) var dst: texture_storage_2d<rgba16float, write>;

fn sampleTrans(r: f32, mu: f32) -> vec3f {
  return textureSampleLevel(transLUT, sampC, transRMuToUV(r, mu), 0.0).rgb;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3u) {
  if (f32(id.x) >= MS_SIZE || f32(id.y) >= MS_SIZE) { return; }
  var uv = (vec2f(id.xy) + 0.5) / MS_SIZE;
  uv = vec2f(fromSubUvsToUnit(uv.x, MS_SIZE), fromSubUvsToUnit(uv.y, MS_SIZE));
  let cosSun = uv.x * 2.0 - 1.0;
  let h = max(uv.y * (RT - RG), 0.01);
  let r = RG + h;
  let ro = vec3f(0.0, r, 0.0);
  let sun = vec3f(sqrt(max(0.0, 1.0 - cosSun * cosSun)), cosSun, 0.0);
  var Lsum = vec3f(0.0);
  var Fsum = vec3f(0.0);
  let SQ = 8u;
  let STEPS = 20u;
  for (var a = 0u; a < SQ; a++) {
    for (var b = 0u; b < SQ; b++) {
      let th = acos(1.0 - 2.0 * (f32(b) + 0.5) / f32(SQ));
      let ph = 2.0 * PI * (f32(a) + 0.5) / f32(SQ);
      let rd = vec3f(sin(th) * cos(ph), cos(th), sin(th) * sin(ph));
      let tTop = raySphere(ro, rd, RT);
      let tGr = raySphere(ro, rd, RG);
      var tMax = tTop.y;
      let hitG = tGr.x > 0.0;
      if (hitG) { tMax = tGr.x; }
      let dt = tMax / f32(STEPS);
      var T = vec3f(1.0);
      var L = vec3f(0.0);
      var Fms = vec3f(0.0);
      for (var s = 0u; s < STEPS; s++) {
        let p = ro + rd * ((f32(s) + 0.5) * dt);
        let pr = length(p);
        let m = medium(pr - RG, A.haze.x);
        let scat = m.rayS + vec3f(m.mieS);
        let Ts = exp(-m.ext * dt);
        let smu = dot(p, sun) / pr;
        var sunT = sampleTrans(pr, smu);
        if (sunBelowHorizon(pr, smu)) { sunT = vec3f(0.0); }
        let ext = max(m.ext, vec3f(1e-9));
        let S = scat * sunT / (4.0 * PI);
        L += T * (S - S * Ts) / ext;
        Fms += T * (scat - scat * Ts) / ext;
        T *= Ts;
      }
      if (hitG) {
        let pg = ro + rd * tMax;
        let ng = normalize(pg);
        let smu = dot(ng, sun);
        L += T * sampleTrans(RG, smu) * max(smu, 0.0) * 0.06 / PI;
      }
      Lsum += L;
      Fsum += Fms;
    }
  }
  let invN = 1.0 / f32(SQ * SQ);
  let L2 = Lsum * invN;
  let fms = Fsum * invN;
  textureStore(dst, id.xy, vec4f(L2 / (1.0 - fms), 1.0));
}
`;
  var SKYVIEW_LUT = FRAME_WGSL + ATMO_CORE_WGSL + /* wgsl */
  `
@group(0) @binding(0) var<uniform> F: Frame;
@group(0) @binding(1) var transLUT: texture_2d<f32>;
@group(0) @binding(2) var msLUT: texture_2d<f32>;
@group(0) @binding(3) var sampC: sampler;
@group(0) @binding(4) var dst: texture_storage_2d<rgba16float, write>;
const SKY_W: f32 = ${SKY_W}.0;
const SKY_H: f32 = ${SKY_H}.0;

fn sampleTrans(r: f32, mu: f32) -> vec3f {
  return textureSampleLevel(transLUT, sampC, transRMuToUV(r, mu), 0.0).rgb;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3u) {
  if (f32(id.x) >= SKY_W || f32(id.y) >= SKY_H) { return; }
  var uv = (vec2f(id.xy) + 0.5) / vec2f(SKY_W, SKY_H);
  uv = vec2f(fromSubUvsToUnit(uv.x, SKY_W), fromSubUvsToUnit(uv.y, SKY_H));
  let haze = F.atmo.x;
  let r = RG + F.atmo.y;
  let vHorizon = rhoOf(r);
  let cosBeta = vHorizon / r;
  let beta = acos(clamp(cosBeta, -1.0, 1.0));
  let zha = PI - beta;
  var vza: f32;
  if (uv.y < 0.5) {
    var coord = 2.0 * uv.y;
    coord = 1.0 - coord;
    coord *= coord;
    coord = 1.0 - coord;
    vza = zha * coord;
  } else {
    var coord = uv.y * 2.0 - 1.0;
    coord *= coord;
    vza = zha + beta * coord;
  }
  let vzc = cos(vza);
  let vzs = sin(vza);
  let cx = uv.x * uv.x;
  let lvc = -(cx * 2.0 - 1.0);
  let lvs = sqrt(max(0.0, 1.0 - lvc * lvc));
  let rd = vec3f(vzs * lvc, vzc, vzs * lvs);
  let smu = F.sunDir.y;
  let sun = vec3f(sqrt(max(0.0, 1.0 - smu * smu)), smu, 0.0);
  let ro = vec3f(0.0, r, 0.0);

  let tTop = raySphere(ro, rd, RT);
  let tGr = raySphere(ro, rd, RG);
  var tMax = tTop.y;
  if (tGr.x > 0.0) { tMax = tGr.x; }
  let cosT = dot(rd, sun);
  let pR = phaseRayleigh(cosT);
  let pM = phaseMie(cosT, 0.84);
  var L = vec3f(0.0);
  var T = vec3f(1.0);
  let STEPS = 48u;
  var tPrev = 0.0;
  for (var i = 0u; i < STEPS; i++) {
    let fa = (f32(i) + 1.0) / f32(STEPS);
    let tn = tMax * fa * fa;
    let dt = tn - tPrev;
    let t = tPrev + dt * 0.3;
    tPrev = tn;
    let p = ro + rd * t;
    let pr = length(p);
    let h = pr - RG;
    let m = medium(h, haze);
    let pmu = dot(p, sun) / pr;
    var sunT = sampleTrans(pr, pmu);
    if (sunBelowHorizon(pr, pmu)) { sunT = vec3f(0.0); }
    let msuv = vec2f(fromUnitToSubUvs(pmu * 0.5 + 0.5, MS_SIZE), fromUnitToSubUvs(clamp(h / (RT - RG), 0.0, 1.0), MS_SIZE));
    let ms = textureSampleLevel(msLUT, sampC, msuv, 0.0).rgb;
    let S = sunT * (m.rayS * pR + vec3f(m.mieS * pM)) + ms * (m.rayS + vec3f(m.mieS));
    let Ts = exp(-m.ext * dt);
    let ext = max(m.ext, vec3f(1e-9));
    L += T * (S - S * Ts) / ext;
    T *= Ts;
  }
  textureStore(dst, id.xy, vec4f(L, 1.0));
}
`;
  var SKY_RENDER_FUNCS = NOISE_WGSL + /* wgsl */
  `
const SKY_W: f32 = ${SKY_W}.0;
const SKY_H: f32 = ${SKY_H}.0;

fn camR() -> f32 { return RG + F.atmo.y; }

fn sampleTrans(r: f32, mu: f32) -> vec3f {
  return textureSampleLevel(transLUT, sampC, transRMuToUV(r, mu), 0.0).rgb;
}

fn sunIlluminance() -> vec3f {
  return sampleTrans(camR(), F.sunDir.y) * F.sunDir.w;
}

fn skyViewUV(dir: vec3f) -> vec2f {
  let r = camR();
  let vHorizon = rhoOf(r);
  let cosBeta = vHorizon / r;
  let beta = acos(clamp(cosBeta, -1.0, 1.0));
  let zha = PI - beta;
  let vzc = clamp(dir.y, -1.0, 1.0);
  var uv: vec2f;
  if (vzc >= -cosBeta) {
    var coord = acos(vzc) / zha;
    coord = 1.0 - coord;
    coord = sqrt(max(coord, 0.0));
    coord = 1.0 - coord;
    uv.y = coord * 0.5;
  } else {
    var coord = (acos(vzc) - zha) / beta;
    coord = sqrt(max(coord, 0.0));
    uv.y = coord * 0.5 + 0.5;
  }
  let dl = length(dir.xz);
  let sl = length(F.sunDir.xz);
  var lvc = 1.0;
  if (dl > 1e-6 && sl > 1e-6) { lvc = clamp(dot(dir.xz, F.sunDir.xz) / (dl * sl), -1.0, 1.0); }
  uv.x = sqrt(max(-lvc * 0.5 + 0.5, 0.0));
  return vec2f(fromUnitToSubUvs(uv.x, SKY_W), fromUnitToSubUvs(uv.y, SKY_H));
}

fn skyAtmo(dir: vec3f) -> vec3f {
  return textureSampleLevel(skyLUT, sampC, skyViewUV(dir), 0.0).rgb * F.sunDir.w;
}
`;
  var SKY_CLOUD_FUNCS = (
    /* wgsl */
    `

// ---- clouds -------------------------------------------------------------
// Two thin, light-aware layers: broken low cumulus with cauliflower edges (2 km) and
// high cirrus wisps (7 km). Each is shaded with self-shadowing towards the sun, bright
// sun-facing rims, silver linings on thin back-lit parts and warm multiple scattering.
const CLOUD_ROT: mat2x2f = mat2x2f(0.8, 0.6, -0.6, 0.8);
// fbm whose octaves fade once smaller than the pixel footprint fp (noise units);
// 'billow' turns the small octaves into bulgy |noise| for cauliflower-like edges
fn cloudFbm(p: vec2f, fp: f32, oct: i32, billow: f32) -> f32 {
  var s = 0.0;
  var a = 0.5;
  var q = p;
  var f = 1.0;
  for (var i = 0; i < oct; i++) {
    // fade octaves well before they reach pixel size (half-res buffer) -> no shimmer in motion
    let w = 1.0 - smoothstep(0.08, 0.28, fp * f);
    var n = gnoise(q);
    if (i >= 2) { n = mix(n, abs(n) * 2.0 - 0.38, billow); }
    s += a * n * w;
    q = CLOUD_ROT * q * 2.03 + vec2f(13.1, 7.7);
    f *= 2.03;
    a *= 0.5;
  }
  return s;
}
// Signed "edge" functions (cloud where > 0) and their density ramps. The edge value is
// what the sky pass differentiates (fwidth) to anti-alias the cloud silhouettes analytically.
fn lowEdge(q: vec2f, fp: f32, oct: i32) -> f32 {
  let t = F.cloudP.w;
  var p = q * 0.38 + vec2f(t * 0.004, t * 0.0012);
  let w = vec2f(gnoise(p * 0.3 + vec2f(1.7, 9.2)), gnoise(p * 0.3 + vec2f(8.3, 2.8)));
  p += w * 1.4;
  // slightly elongated cells (cloud streets), mildly billowy small scales for ragged edges
  let base = cloudFbm(p * vec2f(1.0, 1.7), fp * 0.65, oct, 0.35) * 1.3 + 0.5;
  let large = gnoise(q * 0.05 + vec2f(3.1, 1.3)) * 0.7;
  return base + large - (1.0 - F.cloudP.y);
}
fn lowSoft(fp: f32) -> f32 { return 0.38 + min(fp * 0.65, 1.0) * 0.3; }
fn lowDensity(q: vec2f, fp: f32, oct: i32) -> f32 {
  return clamp(lowEdge(q, fp, oct) / lowSoft(fp), 0.0, 1.0);
}
fn highEdge(q: vec2f, fp: f32, oct: i32) -> f32 {
  let t = F.cloudP.w;
  // altostratus deck: smooth, slightly fibrous
  let p = CLOUD_ROT * (q * vec2f(0.09, 0.20)) + vec2f(t * 0.0015, 0.0);
  let n = cloudFbm(p, fp * 0.2, oct, 0.0) * 1.25 + 0.5;
  let big = gnoise(q * 0.012 + vec2f(7.0, 3.0)) * 0.6;
  return n + big - (1.0 - F.atmo.z);
}
fn highSoft(fp: f32) -> f32 { return 0.85 + min(fp * 0.2, 1.0) * 0.3; }
fn highDensity(q: vec2f, fp: f32, oct: i32) -> f32 {
  return clamp(highEdge(q, fp, oct) / highSoft(fp), 0.0, 1.0);
}

struct CloudHit { t: f32, q: vec2f, mu: f32, up: vec3f };
fn hitLayer(dir: vec3f, hc: f32) -> CloudHit {
  var h: CloudHit;
  let ro = vec3f(0.0, camR(), 0.0);
  h.t = raySphere(ro, dir, RG + hc).y;
  let p = ro + dir * h.t;
  h.up = p / length(p);
  h.q = p.xz;   // Ocean Run: camera travels with the boat; keep the clouds fixed so the loop closes
  h.mu = max(dot(h.up, dir), 0.03);
  return h;
}

// d: density at the hit; s1..s3: densities 80 m / 250 m / 600 m towards the sun
fn shadeLayer(dir: vec3f, h: CloudHit, hc: f32, d: f32, s1: f32, s2: f32, s3: f32, sigma: f32, g: f32) -> vec4f {
  let tau = d * sigma / h.mu;
  let alpha = 1.0 - exp(-tau);
  let tauS = (d * 0.5 + s1 * 0.6 + s2 + s3 * 0.7) * sigma * 0.55;
  let Ts = exp(-tauS);
  let sunC = sampleTrans(RG + hc, dot(h.up, F.sunDir.xyz)) * F.sunDir.w;
  let cosT = dot(dir, F.sunDir.xyz);
  let ph = mix(phaseHG(cosT, g), phaseHG(cosT, -0.2), 0.25);
  let fwd = smoothstep(-0.2, 0.7, cosT);
  // single scattering: reflected when front-lit, transmitted through thin parts when back-lit
  let single = sunC * Ts * ph * mix(alpha * 0.55, tau * exp(-tau) * 2.2, fwd) * 3.0;
  // edges where density falls off towards the sun catch direct light
  let rim = clamp((d - s1) * 2.5, 0.0, 1.0);
  // multiple scattering: a low sun lights cloud tops and sun-facing flanks, not the bases we
  // see from below, so diffused sunlight shows mostly when looking towards the sun
  let toward = smoothstep(-0.3, 0.9, cosT);
  let ms = sunC * (0.025 + 0.22 * rim) * mix(1.0, Ts, 0.5) * alpha * (0.25 + 0.75 * toward);
  // bases are lit by the sky from the sides/below (horizon band + sea), cooler than the tops
  let amb = (skyAtmo(normalize(vec3f(dir.x, 0.03, dir.z))) * 0.26 + skyAtmo(vec3f(0.0, 1.0, 0.0)) * 0.42)
            * alpha * (1.0 - 0.35 * d);
  return vec4f(single + ms + amb, alpha);
}

// returns (premultiplied radiance, fraction of the background sky hidden)
// aa: per-layer edge change per pixel (fwidth of the edge function); negative = estimate it
// from the pixel footprint (compute passes, which have no derivatives)
fn cloudLayers(dir: vec3f, pixAng: f32, aa: vec2f, sunT: ptr<function, f32>) -> vec4f {
  *sunT = 1.0;
  if (dir.y <= 0.0) { return vec4f(0.0); }
  let sxz = F.sunDir.xz / max(length(F.sunDir.xz), 1e-4);
  var lo = vec4f(0.0);
  var loVis = 1.0;
  if (F.cloudP.y > 0.001) {
    let hc = F.cloudP.x;
    let h = hitLayer(dir, hc);
    // clear band near the horizon, where the low sun shines under the clouds
    let far = 1.0 - smoothstep(18.0, 36.0, h.t);
    if (far > 0.0) {
      let fp = h.t * pixAng / h.mu;
      let e = lowEdge(h.q, fp, 7);
      let d = clamp(e / lowSoft(fp), 0.0, 1.0) * far;
      if (d > 0.002) {
        let s1 = lowDensity(h.q + sxz * 0.08, fp, 5) * far;
        let s2 = lowDensity(h.q + sxz * 0.25, fp, 4) * far;
        let s3 = lowDensity(h.q + sxz * 0.6, fp, 3) * far;
        lo = shadeLayer(dir, h, hc, d, s1, s2, s3, F.cloudP.z, 0.75);
        // opacity saturates within a sliver of the density ramp, making a hard silhouette:
        // spread it over ~1.5 pixels
        let aw = select(fp * 0.38 * 2.5, aa.x, aa.x >= 0.0);
        lo *= smoothstep(0.0, aw * 1.5 + 1e-6, e);
        loVis = exp(-h.t * 0.02 * (0.6 + 0.4 * F.atmo.x));
      }
    }
  }
  var hi = vec4f(0.0);
  var hiVis = 1.0;
  if (F.atmo.z > 0.001) {
    let hc = 4.5;
    let h = hitLayer(dir, hc);
    let far = 1.0 - smoothstep(30.0, 70.0, h.t);
    if (far > 0.0) {
      let fp = h.t * pixAng / h.mu;
      let e = highEdge(h.q, fp, 6);
      let d = clamp(e / highSoft(fp), 0.0, 1.0) * far;
      if (d > 0.002) {
        let s1 = highDensity(h.q + sxz * 0.15, fp, 4) * far;
        let s2 = highDensity(h.q + sxz * 0.5, fp, 3) * far;
        hi = shadeLayer(dir, h, hc, d, s1, s2, s2, F.atmo.w, 0.80);
        let aw = select(fp * 0.2 * 2.5, aa.y, aa.y >= 0.0);
        hi *= smoothstep(0.0, aw * 1.5 + 1e-6, e);
        hiVis = exp(-h.t * 0.010 * (0.6 + 0.4 * F.atmo.x));
      }
    }
  }
  *sunT = (1.0 - lo.a) * (1.0 - hi.a);
  // low layer in front; aerial perspective fades each layer into the haze without making
  // it transparent to the layer behind
  let col = lo.rgb * loVis + hi.rgb * hiVis * (1.0 - lo.a);
  let hidden = lo.a * loVis + hi.a * hiVis * (1.0 - lo.a);
  return vec4f(col, hidden);
}

fn sunDisk(dir: vec3f) -> vec3f {
  let c = dot(dir, F.sunDir.xyz);
  let ang = sqrt(max(2.0 * (1.0 - c), 0.0));
  let pix = F.screen.w * 2.0 * length(F.camUp.xyz) * 0.75;
  let edge = 1.0 - smoothstep(SUN_ANG_R - pix, SUN_ANG_R + pix, ang);
  if (edge <= 0.0) { return vec3f(0.0); }
  let x = min(ang / SUN_ANG_R, 1.0);
  let mu = sqrt(max(1.0 - x * x, 0.0));
  let limb = vec3f(1.0) - vec3f(0.397, 0.503, 0.652) * (1.0 - mu);
  let rad = F.sunDir.w / (PI * SUN_ANG_R * SUN_ANG_R);
  return rad * limb * edge * sampleTrans(camR(), max(dir.y, -0.01));
}

fn skyRadiance(dirIn: vec3f, withSun: bool, pixAng: f32) -> vec3f {
  var dir = dirIn;
  dir.y = max(dir.y, 0.0005);
  dir = normalize(dir);
  var L = skyAtmo(dir);
  var sunT = 1.0;
  let cl = cloudLayers(dir, pixAng, vec2f(-1.0), &sunT);
  L = L * (1.0 - cl.a) + cl.rgb;
  if (withSun) { L += sunDisk(dir) * sunT; }
  return L;
}
`
  );
  var ENV_CUBE = FRAME_WGSL + ATMO_CORE_WGSL + /* wgsl */
  `
@group(0) @binding(0) var<uniform> F: Frame;
@group(0) @binding(1) var transLUT: texture_2d<f32>;
@group(0) @binding(2) var skyLUT: texture_2d<f32>;
@group(0) @binding(3) var sampC: sampler;
@group(0) @binding(4) var dst: texture_storage_2d_array<rgba16float, write>;
` + SKY_RENDER_FUNCS + SKY_CLOUD_FUNCS + /* wgsl */
  `
const ENV: f32 = ${ENV_SIZE}.0;
fn faceDir(face: u32, uv: vec2f) -> vec3f {
  let u = uv.x * 2.0 - 1.0;
  let v = uv.y * 2.0 - 1.0;
  switch (face) {
    case 0u: { return vec3f(1.0, -v, -u); }
    case 1u: { return vec3f(-1.0, -v, u); }
    case 2u: { return vec3f(u, 1.0, v); }
    case 3u: { return vec3f(u, -1.0, -v); }
    case 4u: { return vec3f(u, -v, 1.0); }
    default: { return vec3f(-u, -v, -1.0); }
  }
}
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3u) {
  if (f32(id.x) >= ENV || f32(id.y) >= ENV) { return; }
  // faces are refreshed round-robin: face = (frame + z) % 6
  let face = (u32(F.post.z) + id.z) % 6u;
  let dir = normalize(faceDir(face, (vec2f(id.xy) + 0.5) / ENV));
  var L: vec3f;
  if (dir.y >= 0.0) {
    L = skyRadiance(dir, false, 1.5708 / ENV * 1.5);
  } else {
    // below the horizon: what a reflected ray would see -> far, hazy ocean
    let hz = skyRadiance(normalize(vec3f(dir.x, 0.004, dir.z)), false, 1.5708 / ENV * 1.5);
    let body = F.waterCol.rgb * (sunIlluminance() * max(F.sunDir.y, 0.0) + skyAtmo(vec3f(0.0, 1.0, 0.0)) * 2.0);
    L = mix(hz * 0.75, body, clamp(-dir.y * 6.0, 0.0, 1.0));
  }
  textureStore(dst, id.xy, face, vec4f(L, 1.0));
}
`;
  var CUBE_MIPS = (
    /* wgsl */
    `
@group(0) @binding(0) var src: texture_2d_array<f32>;
@group(0) @binding(1) var dst: texture_storage_2d_array<rgba16float, write>;
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let size = textureDimensions(dst);
  if (id.x >= size.x || id.y >= size.y) { return; }
  let srcSize = vec2i(textureDimensions(src));
  let p = vec2i(id.xy) * 2;
  // 4x4 tent-ish filter for smoother roughness blur
  var acc = vec4f(0.0);
  var wsum = 0.0;
  for (var j = -1; j <= 2; j++) {
    for (var i = -1; i <= 2; i++) {
      let q = clamp(p + vec2i(i, j), vec2i(0), srcSize - 1);
      let w = select(1.0, 3.0, i == 0 || i == 1) * select(1.0, 3.0, j == 0 || j == 1);
      acc += textureLoad(src, q, id.z, 0) * w;
      wsum += w;
    }
  }
  textureStore(dst, id.xy, id.z, acc / wsum);
}
`
  );
  var SUN_VIS = FRAME_WGSL + ATMO_CORE_WGSL + /* wgsl */
  `
@group(0) @binding(0) var<uniform> F: Frame;
@group(0) @binding(1) var transLUT: texture_2d<f32>;
@group(0) @binding(2) var skyLUT: texture_2d<f32>;
@group(0) @binding(3) var sampC: sampler;
@group(0) @binding(4) var<storage, read_write> sunVis: array<f32, 4>;
` + SKY_RENDER_FUNCS + SKY_CLOUD_FUNCS + /* wgsl */
  `
var<workgroup> part: array<f32, 8>;
// one ray per thread (7 rays across the disc), then a tiny reduction
@compute @workgroup_size(8)
fn main(@builtin(local_invocation_index) k: u32) {
  let sd = F.sunDir.xyz;
  let tx = normalize(cross(sd, vec3f(0.0, 1.0, 0.0)));
  let ty = cross(tx, sd);
  var t = 1.0;
  if (k < 7u) {
    let a = f32(k) * 0.8975979;
    let o = select(vec2f(cos(a), sin(a)) * 0.65, vec2f(0.0), k == 0u);
    let dir = normalize(sd + (tx * o.x + ty * o.y) * SUN_ANG_R);
    let cl = cloudLayers(dir, 0.0, vec2f(-1.0), &t);
  }
  part[k] = t;
  workgroupBarrier();
  if (k == 0u) {
    var acc = 0.0;
    for (var i = 0u; i < 7u; i++) { acc += part[i]; }
    sunVis[0] = acc / 7.0;
  }
}
`;

  // ../pro/src/wgsl/ocean.js
  var RENDER_BINDINGS = (
    /* wgsl */
    `
@group(0) @binding(0) var<uniform> F: Frame;
@group(0) @binding(1) var transLUT: texture_2d<f32>;
@group(0) @binding(2) var skyLUT: texture_2d<f32>;
@group(0) @binding(3) var sampC: sampler;
`
  );
  var SKY_BG = FRAME_WGSL + ATMO_CORE_WGSL + RENDER_BINDINGS + SKY_RENDER_FUNCS + SKY_CLOUD_FUNCS + /* wgsl */
  `
struct VO { @builtin(position) pos: vec4f, @location(0) ndc: vec2f };
@vertex fn vs(@builtin(vertex_index) i: u32) -> VO {
  let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u)) * 2.0 - 1.0;
  var o: VO;
  o.pos = vec4f(p, 0.0, 1.0);
  o.ndc = p;
  return o;
}
@fragment fn fs(in: VO) -> @location(0) vec4f {
  let dir = normalize(F.camFwd.xyz + in.ndc.x * F.camRight.xyz + in.ndc.y * F.camUp.xyz);
  let pixAng = 2.0 * length(F.camUp.xyz) * F.screen.w;
  var d2 = dir;
  d2.y = max(d2.y, 0.0005);
  d2 = normalize(d2);
  // edge functions at this pixel (uniform control flow, so fwidth is valid)
  let hl = hitLayer(d2, F.cloudP.x);
  let el = lowEdge(hl.q, hl.t * pixAng / hl.mu, 7);
  let hh = hitLayer(d2, 4.5);
  let eh = highEdge(hh.q, hh.t * pixAng / hh.mu, 6);
  let aa = vec2f(fwidth(el), fwidth(eh));
  var sunT = 1.0;
  let cl = cloudLayers(d2, pixAng, aa, &sunT);
  let L = skyAtmo(d2) * (1.0 - cl.a) + cl.rgb + sunDisk(dir) * sunT;
  // Ocean Run: the sun disc can exceed half-float range (65504) when it is a few degrees up;
  // Inf would then spread through the bloom and wash out the whole frame
  return vec4f(min(L, vec3f(30000.0)), 1.0);
}
`;
  var OCEAN = FRAME_WGSL + ATMO_CORE_WGSL + RENDER_BINDINGS + /* wgsl */
  `
@group(0) @binding(4) var dispTex: texture_2d_array<f32>;
@group(0) @binding(5) var derivTex: texture_2d_array<f32>;
@group(0) @binding(6) var momTex: texture_2d_array<f32>;
@group(0) @binding(7) var foamTex: texture_2d<f32>;
@group(0) @binding(8) var envCube: texture_cube<f32>;
@group(0) @binding(9) var sampR: sampler;
@group(0) @binding(10) var foamWorld: texture_2d_array<f32>;
@group(0) @binding(11) var<storage, read> sunVis: array<f32, 4>;
@group(0) @binding(12) var<uniform> B: Boat;
` + BOAT_WGSL + SKY_RENDER_FUNCS + /* wgsl */
  `
const ENV_MAXMIP: f32 = ${ENV_MIPS - 1}.0;
const ENV_TEXELS_PER_RAD: f32 = ${ENV_SIZE}.0 / 1.5708;
const TEX_N: f32 = ${N}.0;

struct VSOut {
  @builtin(position) pos: vec4f,
  @location(0) rel: vec3f,     // displaced position relative to the camera
  @location(1) xz: vec2f,      // undisplaced (lagrangian) coordinates
  @location(2) height: f32,
};

@vertex fn vs(@builtin(vertex_index) vi: u32, @location(0) inst: vec4f) -> VSOut {
  let n = u32(F.lodP.y);
  let g = vec2f(f32(vi % (n + 1u)), f32(vi / (n + 1u)));
  let spacing = inst.z / f32(n);
  var xz = inst.xy + g * spacing;
  // CDLOD morph towards the next coarser level
  let R = F.lodP.x * exp2(inst.w);
  let d = distance(vec3f(xz.x, 0.0, xz.y), F.camPos.xyz);
  let mk = clamp((d - 0.8 * R) / (0.2 * R), 0.0, 1.0);
  xz -= fract(g * 0.5) * 2.0 * spacing * mk;
  let eff = spacing * (1.0 + mk);
  var disp = vec3f(0.0);
  for (var c = 0u; c < ${CASCADES}u; c++) {
    let texel = F.cascL[c >> 2u][c & 3u] / TEX_N;
    let lod = log2(max(eff / texel, 1.0)) + 0.5;
    if (lod < 8.0) {
      disp += textureSampleLevel(dispTex, sampR, xz * F.cascInvL[c >> 2u][c & 3u], c, lod).xyz;
    }
  }
  // Ocean Run: near the hull the sea is pushed onto the boat's waterline plane, and the boat
  // raises a bow wave and a Kelvin wake (both fixed to the boat)
  if (B.wake.z > 0.5) {
    let lp = boatLocal(xz);
    let dx = max(max(lp.x - B.dims.w - 1.2, -B.dims.y - 0.3 - lp.x), 0.0);
    let dz = max(abs(lp.y) - hullHW(clamp(lp.x, -B.dims.y, B.dims.w)) * 0.9, 0.0);
    let m = exp(-(dx * dx + dz * dz) * 0.25);
    let up = B.model[1].xyz;
    let o0 = B.model[3].xyz;
    let planeY = o0.y - (up.x * (xz.x - o0.x) + up.z * (xz.y - o0.z)) / max(up.y, 0.2) - 0.04;
    disp = mix(disp, vec3f(0.0, planeY, 0.0), min(m * 1.1, 1.0));
    disp.y += kelvin(lp) * (1.0 - m);
  }
  var o: VSOut;
  let world = vec3f(xz.x + disp.x, disp.y, xz.y + disp.z);
  o.rel = world - F.camPos.xyz;
  o.pos = F.viewProj * vec4f(o.rel, 1.0);
  o.xz = xz;
  o.height = disp.y;
  return o;
}

fn fresnelWater(c: f32) -> f32 {
  // Schlick with F0 for n = 1.333
  let x = 1.0 - clamp(c, 0.0, 1.0);
  let x2 = x * x;
  return 0.02 + 0.98 * x2 * x2 * x;
}

fn lambdaSmith(w: vec3f, cov: vec3f) -> f32 {
  let xz2 = max(dot(w.xz, w.xz), 1e-8);
  let il = inverseSqrt(xz2);
  let az = w.xz * il;
  let sig2 = az.x * az.x * cov.x + az.y * az.y * cov.y + 2.0 * az.x * az.y * cov.z;
  let cotT = w.y * il;
  let a = max(cotT, 1e-3) / sqrt(2.0 * max(sig2, 1e-7));
  if (a >= 1.6) { return 0.0; }
  return (1.0 - 1.259 * a + 0.396 * a * a) / (3.535 * a + 2.181 * a * a);
}

fn foamPattern(xz: vec2f, t: f32) -> f32 {
  // Ocean Run: scales are snapped (FOAM_S*) so the pattern repeats over one loop's travel
  let a = textureSample(foamTex, sampR, xz * FOAM_S0).r;
  let b = textureSample(foamTex, sampR, xz * FOAM_S1 + vec2f(0.37, 0.71)).g;
  return a * 0.78 + b * 0.22;
}

@fragment fn fs(in: VSOut) -> @location(0) vec4f {
  let xz = in.xz;
  var s = vec2f(0.0);
  var jxx = 0.0;
  var jzz = 0.0;
  var jxz = 0.0;
  var cov = vec3f(0.0);
  // wind gusts ("cat's paws"): patches where the short ripples are stronger or calmer
  let gustN = textureSample(foamTex, sampR, xz * FOAM_S3).b;
  let gust = mix(0.35, 1.2, smoothstep(0.3, 0.72, gustN));
  for (var c = 0u; c < ${CASCADES}u; c++) {
    let uv = xz * F.cascInvL[c >> 2u][c & 3u];
    let gk = mix(1.0, gust, select(0.0, select(0.5, 1.0, c >= 4u), c >= 3u));
    let dv = textureSample(derivTex, sampR, uv, c) * gk;
    let mv = textureSample(momTex, sampR, uv, c) * (gk * gk);
    let av = textureSample(dispTex, sampR, uv, c) * gk;
    s += dv.xy;
    jxx += dv.z;
    jzz += dv.w;
    jxz += av.w;
    cov += vec3f(max(mv.x - dv.x * dv.x, 0.0), max(mv.y - dv.y * dv.y, 0.0), mv.z - dv.x * dv.y);
  }
  let dist = length(in.rel);
  let V = -in.rel / dist;
  let Ld = F.sunDir.xyz;

  // slope statistics (LEAN): add unresolved capillary roughness + sun disk size
  // unresolved roughness; grows with distance to reach Cox-Munk-like totals in the far sheen
  let unres = F.waterCol.w + 0.0035 * smoothstep(30.0, 500.0, dist);
  cov.x += unres;
  cov.y += unres;
  let jx = max(1.0 + jxx, 0.15);
  let jz = max(1.0 + jzz, 0.15);
  let sl = vec2f(s.x / jx, s.y / jz);
  let nrm = normalize(vec3f(-sl.x, 1.0, -sl.y));
  // specular anti-aliasing: slope variation across the pixel widens the lobe
  let dsx = dpdx(sl);
  let dsy = dpdy(sl);
  let aa = vec3f(dsx.x * dsx.x + dsy.x * dsy.x, dsx.y * dsx.y + dsy.y * dsy.y, dsx.x * dsx.y + dsy.x * dsy.y) * 0.5;
  cov += aa * min(1.0, 0.03 / max(max(aa.x, aa.y), 1e-8));
  let J = (1.0 + jxx) * (1.0 + jzz) - jxz * jxz;

  let sunE = sunIlluminance() * sunVis[0];
  let skyUp = textureSampleLevel(envCube, sampC, vec3f(0.0, 1.0, 0.0), ENV_MAXMIP).rgb;
  // average radiance of the whole upper hemisphere (incl. the bright horizon band)
  let skyHemi = skyUp * 0.4 + 0.15 * (
      textureSampleLevel(envCube, sampC, vec3f(0.94, 0.34, 0.0), ENV_MAXMIP - 1.0).rgb +
      textureSampleLevel(envCube, sampC, vec3f(-0.94, 0.34, 0.0), ENV_MAXMIP - 1.0).rgb +
      textureSampleLevel(envCube, sampC, vec3f(0.0, 0.34, 0.94), ENV_MAXMIP - 1.0).rgb +
      textureSampleLevel(envCube, sampC, vec3f(0.0, 0.34, -0.94), ENV_MAXMIP - 1.0).rgb);
  let skyN = textureSampleLevel(envCube, sampC, nrm, ENV_MAXMIP - 1.0).rgb;

  // ---- sun glitter: anisotropic Gaussian slope distribution ----
  let Hv = normalize(V + Ld);
  let hs = -Hv.xz / max(Hv.y, 1e-3);
  let dsl = hs - sl;
  let det = max(cov.x * cov.y - cov.z * cov.z, 1e-10);
  let e = (dsl.x * dsl.x * cov.y + dsl.y * dsl.y * cov.x - 2.0 * dsl.x * dsl.y * cov.z) / det;
  let pdf = exp(-0.5 * e) / (2.0 * PI * sqrt(det));
  let hy2 = Hv.y * Hv.y;
  let Dm = pdf / max(hy2 * hy2, 1e-6);
  let G = 1.0 / (1.0 + lambdaSmith(V, cov) + lambdaSmith(Ld, cov));
  let Fh = fresnelWater(dot(V, Hv));
  let spec = sunE * (Fh * Dm * G / (4.0 * max(V.y, 0.02))) * step(0.0, Ld.y);

  // ---- sky reflection ----
  // At grazing angles the *visible* facets are biased towards the viewer (projected-area
  // weighting + masking), so the effective reflector tilts up and sees higher, greyer sky.
  let sig = sqrt(max(cov.x + cov.y, 1e-6) * 0.5);
  let vh = normalize(vec3f(V.x, 0.0, V.z) + vec3f(1e-5, 0.0, 0.0));
  let sigV = sqrt(max(vh.x * vh.x * cov.x + vh.z * vh.z * cov.y + 2.0 * vh.x * vh.z * cov.z, 1e-6));
  let aV = max(V.y, 1e-3) / max(sqrt(1.0 - V.y * V.y), 1e-3) / (1.4142 * sigV);
  let tilt = 0.32 * sigV * (1.0 - smoothstep(0.0, 2.5, aV));
  let nV = normalize(nrm + vh * tilt);
  var R = reflect(-V, nV);
  R.y = abs(R.y) * 0.4 + max(R.y, 0.0) * 0.6;
  R = normalize(R);
  let envLod = clamp(log2(max(2.0 * sig * ENV_TEXELS_PER_RAD, 1.0)), 0.0, ENV_MAXMIP);
  let refl = textureSampleLevel(envCube, sampC, R, envLod).rgb;
  let NdV = max(dot(nV, V), 0.0);
  // Fresnel from a partially smoothed normal: the long-wave facet plus the grazing-angle
  // statistics of everything finer (visible facets at grazing incidence stay highly reflective)
  let nF = normalize(mix(nrm, vec3f(0.0, 1.0, 0.0), 0.45));
  let Fr = fresnelWater(max(dot(nF, V), 0.0));

  // ---- water body: upwelling scatter + backlit subsurface glow ----
  let ambient = skyUp * 1.3 + skyN * 0.5;
  let NdL = max(dot(nrm, Ld), 0.0);
  // upwelling light is diffuse: light it with smooth terms only (no per-pixel normal noise)
  var body = F.waterCol.rgb * (skyHemi * 1.6 + sunE * max(Ld.y, 0.0) * 0.8);
  // light transmitted through the thin upper part of backlit crests
  let crest = clamp((in.height - 0.05) / 0.9, 0.0, 1.0);
  let toSun = max(dot(Ld, -V), 0.0);
  let thin = pow(clamp(0.5 - 0.5 * dot(Ld, nrm), 0.0, 1.0), 3.0);
  let k1 = F.sssCol.w * crest * crest * pow(toSun, 4.0) * thin * exp(-dist / 600.0);
  body += F.sssCol.rgb * sunE * k1;

  // ---- foam ----
  // persistent foam from the world-space clipmaps (near: fine, far: coarse), plus
  // instantaneous crest foam from the per-pixel Jacobian for anything beyond them
  let dc = abs(xz - F.camPos.xz);
  let cheb = max(dc.x, dc.y);
  let w0 = 1.0 - smoothstep(0.30 * F.lodP.z, 0.45 * F.lodP.z, cheb);
  let w1 = 1.0 - smoothstep(0.30 * F.lodP.w, 0.45 * F.lodP.w, cheb);
  let fw0 = textureSample(foamWorld, sampR, xz / F.lodP.z, 0u);
  let fw1 = textureSample(foamWorld, sampR, xz / F.lodP.w, 1u);
  let fwl = mix(fw1, fw0, w0) * w1;
  let fresh = clamp((F.foamP.x - J) * F.foamP.y, 0.0, 1.0);
  var cover = clamp((max(fwl.r, fresh * (1.0 - w1 * 0.7)) + fwl.g * 0.6) * F.foamP.z, 0.0, 1.5);
  // Ocean Run: the boat's wake (churn is dense white, the trail is laced by the foam texture)
  let wk = wakeFoam(boatLocal(xz));
  let trail = wk.z;
  cover = max(cover, clamp(wk.x * 1.3 + wk.y, 0.0, 1.4));
  // aged foam breaks into wind-aligned streaks; fresh crest foam stays dense
  let wd = F.windP[0].xy;
  let ws = vec2f(dot(xz, wd), dot(xz, vec2f(-wd.y, wd.x)));
  let streak = textureSample(foamTex, sampR, ws * vec2f(FOAM_S4, 0.11)).a;
  let freshK = clamp(fwl.g * 2.0 + fresh + wk.x * 1.5, 0.0, 1.0);
  cover *= mix(smoothstep(0.25, 0.75, streak) * 1.5, 1.0, freshK);
  let pat = foamPattern(xz, F.camPos.w);
  let fw = fwidth(pat) + 0.03;
  // soft threshold: thin foam is a translucent film, dense foam is opaque white
  // dense foam has crisp edges, thin decaying foam a soft translucent margin
  let soft = 0.04 + 0.14 * (1.0 - clamp(cover, 0.0, 1.0));
  let fm = smoothstep(1.0 - cover - fw - soft, 1.0 - cover + fw + soft, pat);
  var foamMask = clamp(fm * smoothstep(0.0, 0.5, cover), 0.0, 1.0) * (0.68 + 0.32 * smoothstep(0.3, 0.9, cover));
  // freshly broken crests: dense, almost opaque core
  foamMask = max(foamMask, smoothstep(0.35, 0.9, max(fwl.g, wk.x)) * smoothstep(0.25, 0.6, pat + 0.25) * 0.95);
  // submerged bubbles brighten and tint the water just beneath foam
  body += F.sssCol.rgb * (sunE * 0.05 + ambient * 0.35) * clamp(cover, 0.0, 1.0);
  body += vec3f(0.10, 0.32, 0.30) * skyUp * fwl.g * 0.35;
  // churned, air-filled water in the wake glows turquoise between the foam
  body += vec3f(0.06, 0.30, 0.30) * (skyHemi * 1.2 + sunE * max(Ld.y, 0.0) * 0.25) * clamp(trail, 0.0, 1.0);
  // bubbly brightness variation inside the foam
  let bub = textureSample(foamTex, sampR, xz * FOAM_S2 + vec2f(0.21, 0.53)).g;
  let foamAlb = vec3f(0.82, 0.87, 0.92) * (0.78 + 0.3 * bub);
  // diffuse: albedo * (sky irradiance / pi + sun * cos / pi); sky dominates at sunset
  // foam scatters light from the whole hemisphere (incl. the cool zenith): use a less saturated ambient
  let foamAmb = skyHemi * 1.0 + skyN * 0.3;
  let foamAmbN = mix(foamAmb, vec3f(dot(foamAmb, vec3f(0.2126, 0.7152, 0.0722))), 0.85);
  // grazing sun is largely self-shadowed by the foam's lumpy micro-structure
  let foamLit = foamAlb * (sunE * max(dot(nrm, Ld), 0.0) * 0.318 * 0.5 + foamAmbN * 1.55);

  if (F.post.w > 0.5) {
    // debug: r = foam cover, g = clamp(J), b = world foam level-0 weight
    if (F.post.w < 1.5) { return vec4f(cover, clamp(J * 0.5, 0.0, 1.0), w0 * 0.3, 1.0) * 0.2; }
    if (F.post.w > 2.5 && F.post.w < 3.5) { return vec4f(refl, 1.0); }
    if (F.post.w > 3.5 && F.post.w < 4.5) { return vec4f(vec3f(Fr), 1.0) * 0.5; }
    if (F.post.w > 4.5 && F.post.w < 5.5) { return vec4f(body * (1.0 - Fr), 1.0); }
    if (F.post.w > 5.5 && F.post.w < 6.5) { return vec4f(foamLit, 1.0); }
    if (F.post.w > 6.5 && F.post.w < 7.5) { return vec4f(skyUp, 1.0); }
    return vec4f(fwl.r, fwl.g, fresh, 1.0) * 0.2;
  }
  var col = body * (1.0 - Fr) + refl * Fr + spec;
  col = mix(col, foamLit, foamMask * 0.95);

  // ---- aerial perspective ----
  // aerial perspective: atmosphere extinction plus a marine haze floor
  let T = exp(-max(F.fogExt.rgb, vec3f(F.fogExt.w)) * dist);
  let hz = normalize(vec3f(-V.x, 0.002, -V.z));
  let fogCol = skyAtmo(hz);
  col = col * T + fogCol * (1.0 - T);
  return vec4f(min(col, vec3f(30000.0)), 1.0);
}
`;
  var HEIGHT_PROBE = FRAME_WGSL + /* wgsl */
  `
@group(0) @binding(0) var<uniform> F: Frame;
@group(0) @binding(1) var dispTex: texture_2d_array<f32>;
@group(0) @binding(2) var samp: sampler;
@group(0) @binding(3) var<storage, read_write> outH: array<f32, 4>;
fn dispAt(xz: vec2f) -> vec3f {
  var d = vec3f(0.0);
  for (var c = 0u; c < ${CASCADES}u; c++) {
    d += textureSampleLevel(dispTex, samp, xz * F.cascInvL[c >> 2u][c & 3u], c, 0.0).xyz;
  }
  return d;
}
@compute @workgroup_size(1)
fn main() {
  let goal = F.camPos.xz;
  var x0 = goal;
  for (var i = 0; i < 4; i++) {
    let d = dispAt(x0);
    x0 = goal - d.xz;
  }
  outH[0] = dispAt(x0).y;
}
`;

  // ../pro/src/wgsl/post.js
  var FS_TRI = (
    /* wgsl */
    `
struct VO { @builtin(position) pos: vec4f, @location(0) uv: vec2f };
@vertex fn vs(@builtin(vertex_index) i: u32) -> VO {
  let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
  var o: VO;
  o.pos = vec4f(p * 2.0 - 1.0, 0.0, 1.0);
  o.uv = vec2f(p.x, 1.0 - p.y);
  return o;
}
`
  );
  var BLOOM_DOWN = FS_TRI + /* wgsl */
  `
struct P { texel: vec4f };   // xy = 1/src size, z = first pass (karis + threshold)
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var<uniform> U: P;
fn luma(c: vec3f) -> f32 { return dot(c, vec3f(0.2126, 0.7152, 0.0722)); }
fn tap(uv: vec2f) -> vec3f { return textureSampleLevel(src, samp, uv, 0.0).rgb; }
fn karis(c: vec3f) -> f32 { return 1.0 / (1.0 + luma(c) * 0.25); }
@fragment fn fs(in: VO) -> @location(0) vec4f {
  let t = U.texel.xy;
  let uv = in.uv;
  let a = tap(uv + t * vec2f(-2.0, -2.0));
  let b = tap(uv + t * vec2f(0.0, -2.0));
  let c = tap(uv + t * vec2f(2.0, -2.0));
  let d = tap(uv + t * vec2f(-2.0, 0.0));
  let e = tap(uv);
  let f = tap(uv + t * vec2f(2.0, 0.0));
  let g = tap(uv + t * vec2f(-2.0, 2.0));
  let h = tap(uv + t * vec2f(0.0, 2.0));
  let i = tap(uv + t * vec2f(2.0, 2.0));
  let j = tap(uv + t * vec2f(-1.0, -1.0));
  let k = tap(uv + t * vec2f(1.0, -1.0));
  let l = tap(uv + t * vec2f(-1.0, 1.0));
  let m = tap(uv + t * vec2f(1.0, 1.0));
  var o: vec3f;
  if (U.texel.z > 0.5) {
    // Karis average on the first downsample to tame fireflies from sun glints
    let g0 = (a + b + d + e) * 0.25;
    let g1 = (b + c + e + f) * 0.25;
    let g2 = (d + e + g + h) * 0.25;
    let g3 = (e + f + h + i) * 0.25;
    let g4 = (j + k + l + m) * 0.25;
    let w0 = karis(g0) * 0.125; let w1 = karis(g1) * 0.125; let w2 = karis(g2) * 0.125;
    let w3 = karis(g3) * 0.125; let w4 = karis(g4) * 0.5;
    o = (g0 * w0 + g1 * w1 + g2 * w2 + g3 * w3 + g4 * w4) / (w0 + w1 + w2 + w3 + w4);
  } else {
    o = e * 0.125 + (a + c + g + i) * 0.03125 + (b + d + f + h) * 0.0625 + (j + k + l + m) * 0.125;
  }
  return vec4f(min(o, vec3f(6.0e4)), 1.0);
}
`;
  var BLOOM_UP = FS_TRI + /* wgsl */
  `
struct P { texel: vec4f };   // xy = 1/src size, z = radius, w = weight
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var<uniform> U: P;
fn tap(uv: vec2f) -> vec3f { return textureSampleLevel(src, samp, uv, 0.0).rgb; }
@fragment fn fs(in: VO) -> @location(0) vec4f {
  let t = U.texel.xy * U.texel.z;
  let uv = in.uv;
  var o = tap(uv) * 4.0;
  o += (tap(uv + vec2f(-t.x, 0.0)) + tap(uv + vec2f(t.x, 0.0)) + tap(uv + vec2f(0.0, -t.y)) + tap(uv + vec2f(0.0, t.y))) * 2.0;
  o += tap(uv + vec2f(-t.x, -t.y)) + tap(uv + vec2f(t.x, -t.y)) + tap(uv + vec2f(-t.x, t.y)) + tap(uv + vec2f(t.x, t.y));
  return vec4f(o * (U.texel.w / 16.0), 1.0);
}
`;
  var COMPOSITE = FS_TRI + /* wgsl */
  `
struct P { p: vec4f, q: vec4f };  // p: exposure, bloom, frame, vignette; q: saturation, contrast, warmth, grain
@group(0) @binding(0) var hdr: texture_2d<f32>;
@group(0) @binding(1) var bloom: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;
@group(0) @binding(3) var<uniform> U: P;

fn agxContrast(x: vec3f) -> vec3f {
  let x2 = x * x;
  let x4 = x2 * x2;
  return 15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4 - 6.868 * x2 * x + 0.4298 * x2 + 0.1191 * x - 0.00232;
}
fn agx(v: vec3f) -> vec3f {
  let m = mat3x3f(0.842479062253094, 0.0423282422610123, 0.0423756549057051,
                  0.0784335999999992, 0.878468636469772, 0.0784336,
                  0.0792237451477643, 0.0791661274605434, 0.879142973793104);
  let minEv = -12.47393;
  let maxEv = 4.026069;
  var x = m * max(v, vec3f(1e-10));
  x = clamp(log2(x), vec3f(minEv), vec3f(maxEv));
  x = (x - minEv) / (maxEv - minEv);
  return agxContrast(x);
}
fn agxEotf(v: vec3f) -> vec3f {
  let mi = mat3x3f(1.19687900512017, -0.0528968517574562, -0.0529716355144438,
                   -0.0980208811401368, 1.15190312990417, -0.0980434501171241,
                   -0.0990297440797205, -0.0989611768448433, 1.15107367264116);
  return mi * v;
}
fn agxLook(v: vec3f, sat: f32, power: f32) -> vec3f {
  let l = dot(v, vec3f(0.2126, 0.7152, 0.0722));
  let p = pow(max(v, vec3f(0.0)), vec3f(power));
  return l + sat * (p - l);
}
fn ign(p: vec2f) -> f32 { return fract(52.9829189 * fract(dot(p, vec2f(0.06711056, 0.00583715)))); }

@fragment fn fs(in: VO) -> @location(0) vec4f {
  let dims = vec2f(textureDimensions(hdr));
  let pc = vec2i(in.pos.xy);
  var c = textureLoad(hdr, pc, 0).rgb;
  // the up-chain accumulates 8 blur levels
  let b = textureSampleLevel(bloom, samp, in.uv, 0.0).rgb * (1.0 / 8.0);
  c = mix(c, b, U.p.y);
  c *= U.p.x;
  // subtle vignette
  let d = (in.uv - 0.5) * vec2f(dims.x / dims.y, 1.0);
  c *= 1.0 - U.p.w * dot(d, d);
  c *= vec3f(1.0 + U.q.z, 1.0, 1.0 - U.q.z);
  // film/sensor-like highlight desaturation: very bright glints and the sun go towards white
  let lum = dot(c, vec3f(0.2126, 0.7152, 0.0722));
  c = mix(c, vec3f(lum), smoothstep(0.7, 6.0, lum) * 0.55);
  var o = agx(c);
  o = agxLook(o, U.q.x, U.q.y);
  o = agxEotf(o);
  // triangular dither against banding in the sky gradients
  let n = ign(in.pos.xy + U.p.z * 5.588238) + ign(in.pos.xy * 1.37 + 17.0 + U.p.z * 3.1) - 1.0;
  o += n / 255.0;
  return vec4f(clamp(o, vec3f(0.0), vec3f(1.0)), 1.0);
}
`;

  // ../pro/src/wgsl/boat.js
  var BOAT_RENDER = FRAME_WGSL + ATMO_CORE_WGSL + /* wgsl */
  `
@group(0) @binding(0) var<uniform> F: Frame;
@group(0) @binding(1) var transLUT: texture_2d<f32>;
@group(0) @binding(2) var skyLUT: texture_2d<f32>;
@group(0) @binding(3) var sampC: sampler;
@group(0) @binding(4) var envCube: texture_cube<f32>;
@group(0) @binding(5) var<storage, read> sunVis: array<f32, 4>;
@group(0) @binding(6) var<uniform> B: Boat;
@group(0) @binding(7) var<uniform> SH: mat4x4f;          // boat-local -> shadow clip
@group(0) @binding(8) var shadowMap: texture_depth_2d;
@group(0) @binding(9) var shadowSamp: sampler_comparison;

struct Mat {
  base: vec4f,      // linear albedo, alpha
  pbr:  vec4f,      // metallic, roughness, clear coat, procedural mode
  emis: vec4f,      // emissive rgb, has texture
};
@group(1) @binding(0) var<uniform> M: Mat;
@group(1) @binding(1) var baseTex: texture_2d<f32>;
@group(1) @binding(2) var baseSamp: sampler;
` + BOAT_WGSL + SKY_RENDER_FUNCS + /* wgsl */
  `
const ENV_MAXMIP: f32 = ${ENV_MIPS - 1}.0;

struct VSOut {
  @builtin(position) pos: vec4f,
  @location(0) rel: vec3f,
  @location(1) nrm: vec3f,
  @location(2) uv: vec2f,
  @location(3) lp: vec3f,     // boat-local position
};

@vertex fn vs(@location(0) p: vec3f, @location(1) n: vec3f, @location(2) uv: vec2f) -> VSOut {
  var o: VSOut;
  let w = B.model * vec4f(p, 1.0);
  o.rel = w.xyz - F.camPos.xyz;
  o.pos = F.viewProj * vec4f(o.rel, 1.0);
  o.nrm = (B.model * vec4f(n, 0.0)).xyz;
  o.uv = uv;
  o.lp = p;
  return o;
}

fn shadowAt(lp: vec3f, n: vec3f) -> f32 {
  let c = SH * vec4f(lp + n * 0.03, 1.0);
  let uv = vec2f(c.x * 0.5 + 0.5, 0.5 - c.y * 0.5);
  if (any(uv < vec2f(0.0)) || any(uv > vec2f(1.0)) || c.z > 1.0) { return 1.0; }
  let ts = 1.0 / f32(textureDimensions(shadowMap).x);
  var s = 0.0;
  for (var y = -1; y <= 1; y++) {
    for (var x = -1; x <= 1; x++) {
      s += textureSampleCompareLevel(shadowMap, shadowSamp, uv + vec2f(f32(x), f32(y)) * ts * 1.5, c.z - 0.0015);
    }
  }
  return s / 9.0;
}

fn ggx(NdH: f32, a: f32) -> f32 { let a2 = a * a; let d = NdH * NdH * (a2 - 1.0) + 1.0; return a2 / (PI * d * d); }
fn smithG(NdV: f32, NdL: f32, a: f32) -> f32 {
  let k = (a + 1.0) * (a + 1.0) / 8.0;
  return (NdV / (NdV * (1.0 - k) + k)) * (NdL / (NdL * (1.0 - k) + k));
}
// split-sum environment BRDF, analytic fit (Karis)
fn envBRDF(F0: vec3f, rough: f32, NdV: f32) -> vec3f {
  let c0 = vec4f(-1.0, -0.0275, -0.572, 0.022);
  let c1 = vec4f(1.0, 0.0425, 1.04, -0.04);
  let r = rough * c0 + c1;
  let a004 = min(r.x * r.x, exp2(-9.28 * NdV)) * r.x + r.y;
  let ab = vec2f(-1.04, 1.04) * a004 + r.zw;
  return F0 * ab.x + ab.y;
}
fn envLookup(dir: vec3f, lod: f32) -> vec3f { return textureSampleLevel(envCube, sampC, dir, lod).rgb; }

@fragment fn fs(in: VSOut) -> @location(0) vec4f {
  let dist = length(in.rel);
  let V = -in.rel / dist;
  var n = select(vec3f(0.0, 1.0, 0.0), normalize(in.nrm), dot(in.nrm, in.nrm) > 1e-12);
  if (dot(n, V) < 0.0) { n = -n; }               // two-sided (thin parts, open meshes)
  let L = F.sunDir.xyz;
  var albedo = M.base.rgb;
  if (M.emis.w > 0.5) { albedo *= textureSample(baseTex, baseSamp, in.uv).rgb; }
  var metal = M.pbr.x;
  var rough = M.pbr.y;
  var coat = M.pbr.z;
  let lp = in.lp;
  let mode = M.pbr.w;
  if (mode > 0.5 && mode < 1.5) {
    // hull paint: black antifouling and boot top, dark hull windows amidships
    if (lp.y < 0.32) { albedo = vec3f(0.03, 0.033, 0.036); rough = 0.35; }
    // dark rub rail just under the sheer line (the procedural yacht's sheer)
    let s = clamp((lp.x + 9.0) / 19.0, 0.0, 1.0);
    let sheerY = 1.6 + 0.9 * pow(s * s * (3.0 - 2.0 * s), 2.0);
    if (abs(lp.y - (sheerY - 0.12)) < 0.06) { albedo = vec3f(0.04, 0.05, 0.07); rough = 0.3; }
    let hw = smoothstep(0.08, 0.0, abs(lp.y - 0.78) - 0.11) * smoothstep(0.3, 0.0, abs(lp.x - 0.4) - 3.8);
    albedo = mix(albedo, vec3f(0.012, 0.016, 0.02), hw);
    rough = mix(rough, 0.05, hw);
  } else if (mode > 1.5 && mode < 2.5) {
    // superstructure: wrap-around tinted window band and raked windscreen
    let band = smoothstep(0.03, 0.0, abs(lp.y - 2.62) - 0.36);
    let screen = step(0.45, (B.invModel * vec4f(n, 0.0)).x) * step(2.05, lp.y);
    let g = max(band, screen) * step(lp.x, 4.0);
    albedo = mix(albedo, vec3f(0.012, 0.016, 0.022), g);
    rough = mix(rough, 0.03, g);
  } else if (mode > 2.5) {
    // deck: teak planks aft, white non-skid elsewhere
    if (lp.x < -5.6) {
      let plank = smoothstep(0.0, 0.06, abs(fract(lp.z * 6.5) - 0.5));
      albedo = vec3f(0.42, 0.27, 0.14) * (0.65 + 0.35 * plank);
      rough = 0.7;
    }
  }
  let NdV = max(dot(n, V), 1e-3);
  let NdL = max(dot(n, L), 0.0);
  let H = normalize(V + L);
  let NdH = max(dot(n, H), 0.0);
  let F0 = mix(vec3f(0.04), albedo, metal);
  // the sun is a disc ~0.5 deg wide: widen its highlight so a mirror can't reflect a single
  // pixel of near-infinite brightness (which bloom would smear over the whole frame)
  let a = max(rough * rough, 0.02);
  let sunE = sunIlluminance() * sunVis[0] * step(0.0, L.y);
  let sh = shadowAt(lp, (B.invModel * vec4f(n, 0.0)).xyz);
  // sky light: diffuse from the blurred cube, specular from the roughness mip
  let irr = envLookup(n, ENV_MAXMIP);
  var R = reflect(-V, n);
  let specEnv = envLookup(R, clamp(rough * ENV_MAXMIP, 0.0, ENV_MAXMIP));
  // crude ambient occlusion: surfaces facing down see the darker sea
  let ao = 0.55 + 0.45 * clamp(n.y * 0.5 + 0.6, 0.0, 1.0);
  let Fsun = F0 + (1.0 - F0) * pow(1.0 - max(dot(H, V), 0.0), 5.0);
  let kd = (1.0 - metal) * (1.0 - Fsun);
  var col = albedo * kd * (irr * ao + sunE * NdL * sh / PI);
  col += specEnv * envBRDF(F0, rough, NdV) * ao;
  col += sunE * NdL * sh * Fsun * ggx(NdH, a) * smithG(NdV, max(NdL, 1e-3), a) / (4.0 * NdV * max(NdL, 1e-3));
  if (coat > 0.0) {
    // glossy gel-coat / lacquer layer
    let Fc = 0.04 + 0.96 * pow(1.0 - NdV, 5.0);
    col = col * (1.0 - Fc * coat) + envLookup(R, 0.6) * Fc * coat;
    col += sunE * NdL * sh * 0.04 * ggx(NdH, 0.02) * smithG(NdV, max(NdL, 1e-3), 0.02) / (4.0 * NdV * max(NdL, 1e-3)) * coat;
  }
  col = min(col + M.emis.rgb, vec3f(200.0));
  // aerial perspective, as for the sea
  let T = exp(-max(F.fogExt.rgb, vec3f(F.fogExt.w)) * dist);
  let hz = normalize(vec3f(-V.x, 0.002, -V.z));
  col = col * T + skyAtmo(hz) * (1.0 - T);
  return vec4f(col, 1.0);
}
`;
  var BOAT_SHADOW = (
    /* wgsl */
    `
@group(0) @binding(0) var<uniform> SH: mat4x4f;
@vertex fn vs(@location(0) p: vec3f) -> @builtin(position) vec4f { return SH * vec4f(p, 1.0); }
`
  );
  var BOAT_PROBE = FRAME_WGSL + /* wgsl */
  `
@group(0) @binding(0) var<uniform> F: Frame;
@group(0) @binding(1) var dispTex: texture_2d_array<f32>;
@group(0) @binding(2) var samp: sampler;
@group(0) @binding(3) var<storage, read_write> outH: array<f32, 32>;
@group(0) @binding(4) var<uniform> pts: array<vec4f, 16>;   // two xz points per entry
fn dispAt(xz: vec2f) -> vec3f {
  var d = vec3f(0.0);
  for (var c = 0u; c < ${CASCADES}u; c++) {
    d += textureSampleLevel(dispTex, samp, xz * F.cascInvL[c >> 2u][c & 3u], c, 0.0).xyz;
  }
  return d;
}
@compute @workgroup_size(32)
fn main(@builtin(local_invocation_index) i: u32) {
  let e = pts[i / 2u];
  let goal = select(e.xy, e.zw, (i & 1u) == 1u);
  var x0 = goal;
  for (var k = 0; k < 4; k++) { x0 = goal - dispAt(x0).xz; }
  outH[i] = dispAt(x0).y;
}
`;

  // ../pro/src/boat.js
  var MATS = {
    hull: { base: [0.86, 0.87, 0.88, 1], metal: 0, rough: 0.22, coat: 1, mode: 1 },
    deck: { base: [0.82, 0.82, 0.8, 1], metal: 0, rough: 0.55, coat: 0, mode: 3 },
    super: { base: [0.88, 0.89, 0.9, 1], metal: 0, rough: 0.25, coat: 1, mode: 2 },
    glass: { base: [0.015, 0.02, 0.026, 1], metal: 0, rough: 0.04, coat: 1, mode: 0 },
    teak: { base: [0.42, 0.27, 0.14, 1], metal: 0, rough: 0.7, coat: 0, mode: 0 },
    chrome: { base: [0.92, 0.93, 0.95, 1], metal: 1, rough: 0.12, coat: 0, mode: 0 },
    cushion: { base: [0.86, 0.83, 0.76, 1], metal: 0, rough: 0.9, coat: 0, mode: 0 },
    dark: { base: [0.05, 0.055, 0.06, 1], metal: 0, rough: 0.4, coat: 0.5, mode: 0 }
  };
  var smooth = (a, b, x) => {
    const t = Math.min(1, Math.max(0, (x - a) / (b - a)));
    return t * t * (3 - 2 * t);
  };
  var Part = class {
    constructor(mat) {
      this.mat = mat;
      this.p = [];
      this.n = [];
      this.t = [];
      this.i = [];
    }
    vert(p, n, uv = [0, 0]) {
      this.p.push(...p);
      this.n.push(...n);
      this.t.push(...uv);
      return this.p.length / 3 - 1;
    }
    tri(a, b, c) {
      this.i.push(a, b, c);
    }
    done() {
      return { pos: new Float32Array(this.p), nrm: new Float32Array(this.n), uv: new Float32Array(this.t), idx: new Uint32Array(this.i), mat: this.mat };
    }
  };
  var sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
  var cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  var norm = (v) => {
    const l = Math.hypot(...v);
    return l > 1e-9 ? [v[0] / l, v[1] / l, v[2] / l] : [0, 1, 0];
  };
  function gridSurface(part, ni, nj, S, flip = false) {
    const base = part.p.length / 3;
    for (let i = 0; i <= ni; i++) {
      for (let j = 0; j <= nj; j++) {
        const p = S(i, j);
        const du = sub(S(Math.min(ni, i + 1), j), S(Math.max(0, i - 1), j));
        const dv = sub(S(i, Math.min(nj, j + 1)), S(i, Math.max(0, j - 1)));
        let n = norm(cross(du, dv));
        if (flip) n = n.map((v) => -v);
        if (!isFinite(n[0])) n = [0, 1, 0];
        part.vert(p, n, [i / ni, j / nj]);
      }
    }
    for (let i = 0; i < ni; i++) {
      for (let j = 0; j < nj; j++) {
        const a = base + i * (nj + 1) + j, b = a + 1, c = a + nj + 1, d = c + 1;
        if (flip) {
          part.tri(a, b, d);
          part.tri(a, d, c);
        } else {
          part.tri(a, d, b);
          part.tri(a, c, d);
        }
      }
    }
  }
  function mirrorZ(part) {
    const nv = part.p.length / 3, ni = part.i.length;
    for (let v = 0; v < nv; v++) part.vert([part.p[v * 3], part.p[v * 3 + 1], -part.p[v * 3 + 2]], [part.n[v * 3], part.n[v * 3 + 1], -part.n[v * 3 + 2]], [part.t[v * 2], part.t[v * 2 + 1]]);
    for (let k = 0; k < ni; k += 3) part.tri(part.i[k] + nv, part.i[k + 2] + nv, part.i[k + 1] + nv);
  }
  function fanCap(part, centre, outline, normal) {
    const c = part.vert(centre, normal);
    const vs = outline.map((p) => part.vert(p, normal));
    for (let k = 0; k < vs.length - 1; k++) part.tri(c, vs[k], vs[k + 1]);
  }
  function roundedBox(part, c, h, r, segs = 6) {
    const inner = h.map((v) => Math.max(v - r, 1e-4));
    const faces = [[0, 1, 2, 1], [0, 1, 2, -1], [1, 2, 0, 1], [1, 2, 0, -1], [2, 0, 1, 1], [2, 0, 1, -1]];
    for (const [a, b, n, s] of faces) {
      gridSurface(part, segs, segs, (i, j) => {
        const q = [0, 0, 0];
        q[a] = (-1 + 2 * i / segs) * h[a];
        q[b] = (-1 + 2 * j / segs) * h[b];
        q[n] = s * h[n];
        const cl = q.map((v, k) => Math.max(-inner[k], Math.min(inner[k], v)));
        const d = norm(sub(q, cl));
        return [c[0] + cl[0] + d[0] * r, c[1] + cl[1] + d[1] * r, c[2] + cl[2] + d[2] * r];
      }, s < 0);
    }
  }
  function tube(part, pts, r, segs = 8) {
    for (let k = 0; k < pts.length - 1; k++) {
      const a = pts[k], b = pts[k + 1], ax = norm(sub(b, a));
      const t0 = Math.abs(ax[1]) < 0.9 ? [0, 1, 0] : [1, 0, 0];
      const u = norm(cross(ax, t0)), v = cross(ax, u);
      gridSurface(part, 1, segs, (i, j) => {
        const p = i ? b : a, th = j / segs * Math.PI * 2;
        return [0, 1, 2].map((q) => p[q] + (u[q] * Math.cos(th) + v[q] * Math.sin(th)) * r);
      }, true);
    }
  }
  var YACHT_DIMS = { LB: 10, LS: 9, HB: 2.6, bowPow: 1.45 };
  function buildYacht() {
    const { LB, LS, HB, bowPow } = YACHT_DIMS;
    const hb = (x) => x > 0 ? HB * (1 - Math.pow(Math.min(x / LB, 1), bowPow)) : HB * (1 - 0.06 * Math.pow(Math.min(-x / LS, 1), 2));
    const sheer = (x) => 1.6 + 0.9 * Math.pow(smooth(-LS, LB, x), 2);
    const keel = (x) => -0.9 + Math.pow(Math.max(x - 1, 0), 2) * 0.03;
    const NX = 72, NU = 18;
    const stationX = (i) => {
      const s = i / NX;
      return -LS + (LB + LS) * (1 - Math.pow(1 - s, 1.25));
    };
    const hullPart = new Part(MATS.hull);
    const section = (x, u) => {
      const b = Math.max(hb(x), 2e-3), yk = keel(x), ys = sheer(x);
      const bc = b * 0.9, yc = Math.min(yk + bc * 0.32, ys - 0.25);
      if (u < 0.45) {
        const t2 = u / 0.45, z = bc * t2;
        return [x, yk + (yc - yk) * t2 - 0.05 * Math.sin(Math.PI * t2) * Math.min(1, b), z];
      }
      if (u < 0.55) {
        const t2 = (u - 0.45) / 0.1, z = bc + (b * 0.97 - bc) * Math.sin(t2 * Math.PI / 2);
        return [x, yc + 0.12 * (1 - Math.cos(t2 * Math.PI / 2)), z];
      }
      const t = (u - 0.55) / 0.45;
      const y0 = yc + 0.12;
      return [x, y0 + (ys - y0) * t, b * (0.97 + 0.03 * Math.sin(t * Math.PI / 2))];
    };
    gridSurface(hullPart, NX, NU, (i, j) => section(stationX(i), j / NU), true);
    mirrorZ(hullPart);
    const tr = [];
    for (let j = 0; j <= NU; j++) tr.push(section(-LS, j / NU));
    const trAll = [...tr.slice().reverse().map((p) => [p[0], p[1], -p[2]]), ...tr];
    fanCap(hullPart, [-LS, (keel(-LS) + sheer(-LS)) / 2, 0], trAll, [-1, 0, 0]);
    const deckPart = new Part(MATS.deck);
    gridSurface(deckPart, NX, 6, (i, j) => {
      const x = stationX(i), b = Math.max(hb(x), 2e-3), v = j / 6;
      return [x, sheer(x) + 0.07 * Math.sin(v * Math.PI / 2), b * (1 - v)];
    }, false);
    mirrorZ(deckPart);
    const dk = [];
    for (let j = 0; j <= 6; j++) {
      const b = hb(-LS), v = j / 6;
      dk.push([-LS, sheer(-LS) + 0.07 * Math.sin(v * Math.PI / 2), b * (1 - v)]);
    }
    fanCap(deckPart, [-LS, sheer(-LS) - 0.01, 0], [...dk, ...dk.slice().reverse().map((p) => [p[0], p[1], -p[2]])], [-1, 0, 0]);
    const superPart = new Part(MATS.super);
    const sx0 = -5.7, sx1 = 4.6, SN = 40, RN = 20;
    const sw = (x) => Math.min(hb(x) - 0.4, 2.15);
    const stop = (x) => 3.25 - Math.max(x - 1.6, 0) * 0.62;
    const ring = (x, u) => {
      const w = Math.max(sw(x), 0.05), y0 = sheer(x) + 0.02, y1 = Math.max(stop(x), y0 + 0.08), r = Math.min(0.32, (y1 - y0) * 0.45, w * 0.4);
      const sideLen = y1 - r - y0, arc = r * Math.PI / 2, roofLen = w - r, L = sideLen + arc + roofLen;
      let d = u * L;
      if (d < sideLen) return [x, y0 + d, w - 0.04 * (d / sideLen)];
      d -= sideLen;
      if (d < arc) {
        const a = d / r;
        return [x, y1 - r + Math.sin(a) * r, w - 0.04 - r + Math.cos(a) * r];
      }
      d -= arc;
      return [x, y1 + 0.03 * (d / roofLen), w - 0.04 - r - d];
    };
    gridSurface(superPart, SN, RN, (i, j) => ring(sx0 + (sx1 - sx0) * i / SN, j / RN), true);
    mirrorZ(superPart);
    for (const [x, nx] of [[sx0, -1], [sx1, 1]]) {
      const ol = [];
      for (let j = 0; j <= RN; j++) ol.push(ring(x, j / RN));
      fanCap(superPart, [x, (sheer(x) + stop(x)) / 2, 0], [...ol.slice().reverse().map((p) => [p[0], p[1], -p[2]]), ...ol], [nx, 0, 0]);
    }
    const flyPart = new Part(MATS.super);
    roundedBox(flyPart, [-2.2, 3.36, 0], [3, 0.1, 1.95], 0.08);
    roundedBox(flyPart, [-3, 4.62, 0], [1.75, 0.07, 1.85], 0.06);
    roundedBox(flyPart, [-3, 4.75, 0], [0.32, 0.08, 0.32], 0.08);
    const glassPart = new Part(MATS.glass);
    roundedBox(glassPart, [-3, 4.7, 0], [1.35, 0.03, 1.4], 0.03);
    roundedBox(glassPart, [2.6, 3.22, 0], [0.9, 0.04, 1.2], 0.04);
    roundedBox(glassPart, [0.55, 3.62, 0], [0.12, 0.2, 1.7], 0.05);
    const cushPart = new Part(MATS.cushion);
    roundedBox(cushPart, [-3.3, 3.62, -0.9], [0.9, 0.18, 0.7], 0.12);
    roundedBox(cushPart, [-3.3, 3.62, 0.9], [0.9, 0.18, 0.7], 0.12);
    roundedBox(cushPart, [-0.9, 3.6, 0], [0.45, 0.16, 1.2], 0.12);
    roundedBox(cushPart, [6.2, sheer(6.2) + 0.16, 0], [1.6, 0.13, 1.1], 0.12);
    roundedBox(cushPart, [-8.3, sheer(-8.3) + 0.3, 0], [0.4, 0.2, 1.9], 0.12);
    roundedBox(cushPart, [-6.4, sheer(-6.4) + 0.3, -1.5], [0.9, 0.2, 0.35], 0.12);
    roundedBox(cushPart, [-6.4, sheer(-6.4) + 0.3, 1.5], [0.9, 0.2, 0.35], 0.12);
    const chromePart = new Part(MATS.chrome);
    for (const s of [-1, 1]) {
      tube(chromePart, [[-1.4, 3.45, s * 1.75], [-1.5, 4.55, s * 1.65]], 0.06);
      tube(chromePart, [[-4.6, 3.45, s * 1.75], [-4.5, 4.55, s * 1.65]], 0.06);
      const rail = [];
      for (let k = 0; k <= 12; k++) {
        const x = 2.4 + (9.6 - 2.4) * k / 12;
        rail.push([x, sheer(x) + 0.62, s * Math.max(hb(x) - 0.12, 0.05)]);
      }
      tube(chromePart, rail, 0.022, 6);
      for (let k = 0; k <= 12; k += 3) tube(chromePart, [[rail[k][0], rail[k][1] - 0.62, rail[k][2]], rail[k]], 0.018, 6);
    }
    tube(chromePart, [[-3.6, 4.7, 0], [-3.6, 5.6, 0]], 0.03, 6);
    const teakPart = new Part(MATS.teak);
    roundedBox(teakPart, [-LS - 0.85, 0.42, 0], [0.85, 0.06, HB * 0.88], 0.03);
    roundedBox(teakPart, [-7, sheer(-7) + 0.42, 0], [0.7, 0.04, 0.55], 0.03);
    roundedBox(teakPart, [-2.2, 3.48, 0], [0.5, 0.03, 0.45], 0.03);
    return { parts: [hullPart, deckPart, superPart, flyPart, glassPart, cushPart, chromePart, teakPart].map((p) => p.done()), dims: { ...YACHT_DIMS, XB: 1 + Math.sqrt(0.9 / 0.03) } };
  }
  function fromGLTF(THREE2, scene, opts) {
    scene.updateMatrixWorld(true);
    const raw = [];
    scene.traverse((o) => {
      if (!o.isMesh || !o.geometry.attributes.position) return;
      const g = o.geometry.index ? o.geometry : o.geometry;
      const pos = g.attributes.position, nrm = g.attributes.normal, uv = g.attributes.uv;
      const m = o.matrixWorld, nm = new THREE2.Matrix3().getNormalMatrix(m);
      const P2 = new Float32Array(pos.count * 3), Nn = new Float32Array(pos.count * 3), T = new Float32Array(pos.count * 2);
      const v = new THREE2.Vector3();
      for (let i = 0; i < pos.count; i++) {
        v.fromBufferAttribute(pos, i).applyMatrix4(m);
        P2.set([v.x, v.y, v.z], i * 3);
        if (nrm) {
          v.fromBufferAttribute(nrm, i).applyMatrix3(nm).normalize();
          Nn.set([v.x, v.y, v.z], i * 3);
        }
        if (uv) T.set([uv.getX(i), uv.getY(i)], i * 2);
      }
      let idx = g.index ? Uint32Array.from(g.index.array) : Uint32Array.from({ length: pos.count }, (_, i) => i);
      if (m.determinant() < 0) for (let k = 0; k < idx.length; k += 3) [idx[k + 1], idx[k + 2]] = [idx[k + 2], idx[k + 1]];
      if (!nrm) computeNormals(P2, idx, Nn);
      const mat = [].concat(o.material)[0] || {};
      raw.push({
        pos: P2,
        nrm: Nn,
        uv: T,
        idx,
        mat: {
          base: [mat.color?.r ?? 1, mat.color?.g ?? 1, mat.color?.b ?? 1, mat.opacity ?? 1],
          metal: mat.metalness ?? 0,
          rough: mat.roughness ?? 0.6,
          coat: mat.clearcoat ?? 0,
          mode: 0,
          emissive: mat.emissive ? [mat.emissive.r, mat.emissive.g, mat.emissive.b].map((c) => c * (mat.emissiveIntensity ?? 1)) : [0, 0, 0],
          map: mat.map?.image || null
        }
      });
    });
    if (!raw.length) throw new Error("the model has no meshes");
    let bb = bounds(raw);
    if (bb.size[2] > bb.size[0]) transform(raw, (p) => [p[2], p[1], -p[0]]);
    bb = bounds(raw);
    const s = opts.length / bb.size[0];
    transform(raw, (p) => [(p[0] - bb.centre[0]) * s, (p[1] - bb.min[1]) * s, (p[2] - bb.centre[2]) * s], true);
    bb = bounds(raw);
    const L = bb.size[0], yCut = Math.min(bb.size[1] * 0.3, 4);
    let wF = 0, wB = 0;
    eachVert(raw, (p) => {
      if (p[1] > yCut) return;
      const w = Math.abs(p[2]);
      if (p[0] > bb.max[0] - 0.15 * L) wF = Math.max(wF, w);
      if (p[0] < bb.min[0] + 0.15 * L) wB = Math.max(wB, w);
    });
    if (wF > wB !== !!opts.flip) transform(raw, (p) => [-p[0], p[1], -p[2]]);
    bb = bounds(raw);
    let halfW = 0, gun = 0;
    const mid = (p) => Math.abs(p[0]) < 0.15 * L && p[1] < 0.3 * L;
    eachVert(raw, (p) => {
      if (mid(p)) halfW = Math.max(halfW, Math.abs(p[2]));
    });
    eachVert(raw, (p) => {
      if (mid(p) && Math.abs(p[2]) > 0.85 * halfW) gun = Math.max(gun, p[1]);
    });
    const depth = gun > 0 ? gun : bb.size[1] * 0.3;
    const water = opts.waterline ?? opts.draft * depth;
    transform(raw, (p) => [p[0], p[1] - water, p[2]]);
    let xb = -1e9, xs = 1e9, hbw = 0;
    eachVert(raw, (p) => {
      if (p[1] < 0.1) {
        xb = Math.max(xb, p[0]);
        xs = Math.min(xs, p[0]);
      }
      if (Math.abs(p[1]) < 0.6) hbw = Math.max(hbw, Math.abs(p[2]));
    });
    if (xb < -1e8) {
      xb = L * 0.35;
      xs = -L * 0.45;
    }
    return { parts: raw, dims: { LB: xb + 0.3, LS: -xs, HB: hbw || bb.size[2] / 2, bowPow: 1.6, XB: xb }, waterline: water };
  }
  function computeNormals(P2, idx, N2) {
    for (let k = 0; k < idx.length; k += 3) {
      const a = idx[k] * 3, b = idx[k + 1] * 3, c = idx[k + 2] * 3;
      const n = cross(sub([P2[b], P2[b + 1], P2[b + 2]], [P2[a], P2[a + 1], P2[a + 2]]), sub([P2[c], P2[c + 1], P2[c + 2]], [P2[a], P2[a + 1], P2[a + 2]]));
      for (const v of [a, b, c]) {
        N2[v] += n[0];
        N2[v + 1] += n[1];
        N2[v + 2] += n[2];
      }
    }
    for (let v = 0; v < N2.length; v += 3) {
      const n = norm([N2[v], N2[v + 1], N2[v + 2]]);
      N2.set(n, v);
    }
  }
  function eachVert(parts, f) {
    for (const pt of parts) for (let i = 0; i < pt.pos.length; i += 3) f([pt.pos[i], pt.pos[i + 1], pt.pos[i + 2]]);
  }
  function transform(parts, f, scaleOnly = false) {
    for (const pt of parts) {
      for (let i = 0; i < pt.pos.length; i += 3) {
        const q = f([pt.pos[i], pt.pos[i + 1], pt.pos[i + 2]]);
        pt.pos.set(q, i);
        if (!scaleOnly) {
          const o = f([0, 0, 0]), n = f([pt.nrm[i], pt.nrm[i + 1], pt.nrm[i + 2]]);
          pt.nrm.set(norm(sub(n, o)), i);
        }
      }
    }
  }
  function bounds(parts) {
    const min = [1e9, 1e9, 1e9], max = [-1e9, -1e9, -1e9];
    eachVert(parts, (p) => {
      for (let k = 0; k < 3; k++) {
        min[k] = Math.min(min[k], p[k]);
        max[k] = Math.max(max[k], p[k]);
      }
    });
    return { min, max, size: max.map((v, k) => v - min[k]), centre: max.map((v, k) => (v + min[k]) / 2) };
  }

  // ../pro/src/main.js
  var qs = new URLSearchParams(location.search);
  var num = (k, d) => qs.has(k) && isFinite(+qs.get(k)) ? +qs.get(k) : d;
  var D2R = Math.PI / 180;
  var GU = GPUBufferUsage;
  var TU = GPUTextureUsage;
  var BASE = {
    windSpeed: 7,
    windDir: 180,
    fetchKm: 90,
    swellHs: 1.6,
    swellDir: 160,
    swellLength: 110,
    swellSpread: 48,
    choppy: 1.25,
    turbidity: 1.6,
    foam: 1,
    sunElev: 4,
    sunAzim: 70,
    exposure: 1.35,
    haze: 3,
    clouds: 0.4,
    veil: 0.36,
    bloom: 0.1,
    water: null,
    sss: null
  };
  var SCENES = {
    sunset: { label: "Sunset", sunElev: 3.5, sunAzim: 62, clouds: 0.45, veil: 0.4, windSpeed: 8, swellHs: 1.8 },
    golden: { label: "Golden hour", sunElev: 9, sunAzim: 115, clouds: 0.2, veil: 0.12, haze: 5, windSpeed: 5, swellHs: 1, exposure: 1.2 },
    tropical: {
      label: "Tropical noon",
      sunElev: 52,
      sunAzim: 130,
      clouds: 0.3,
      veil: 0,
      haze: 1.5,
      windSpeed: 6,
      swellHs: 0.8,
      exposure: 0.85,
      water: [2e-3, 0.045, 0.065],
      sss: [0.02, 0.16, 0.15]
    },
    clear: {
      label: "Clear blue",
      sunElev: 38,
      sunAzim: 60,
      clouds: 0.12,
      veil: 0,
      haze: 2,
      windSpeed: 7,
      swellHs: 1,
      exposure: 1.05,
      water: [6e-3, 0.045, 0.085]
    },
    overcast: { label: "Overcast swell", sunElev: 24, sunAzim: 40, clouds: 0.85, veil: 0.85, haze: 5, windSpeed: 11, swellHs: 2.2, exposure: 1.6 },
    storm: { label: "Storm", sunElev: 14, sunAzim: 75, clouds: 1, veil: 1, haze: 9, windSpeed: 15, swellHs: 3, choppy: 1.4, exposure: 1.9, foam: 1.6 }
  };
  var BOATS = {
    yacht: {
      label: "Motor yacht",
      V: 12,
      wake: 1.3,
      twin: 1,
      trim: 0.04,
      heel: 0,
      buzz: 1,
      scene: "tropical",
      view: "drone",
      model: { length: 19, draft: 0.38 },
      cams: {
        chase: { pos: [-38, 9, 18], tgt: [6, 2, -2.5], fov: 40 },
        drone: { pos: [-7, 30, 0], tgt: [-7, 0, 0], up: [0, 0, -1], fov: 50 },
        side: { pos: [10, 4.5, 34], tgt: [0, 2.6, 0], fov: 38 },
        deck: { pos: [-5.3, 4.3, 0.6], tgt: [-40, -2, 0], fov: 70, local: true }
      }
    },
    galleon: {
      label: "Galleon",
      V: 3.5,
      wake: 0.4,
      twin: 0,
      trim: 0,
      heel: 0.03,
      buzz: 0,
      scene: "golden",
      view: "side",
      model: { length: 34, draft: 0.45 },
      cams: {
        chase: { pos: [-62, 14, 34], tgt: [8, 12, -4], fov: 42 },
        drone: { pos: [-6, 95, 0], tgt: [-6, 0, 0], up: [0, 0, -1], fov: 55 },
        side: { pos: [34, 5, 54], tgt: [2, 12.5, 0], fov: 40 },
        deck: { pos: [12, 2.6, -17], tgt: [-1, 13, 0], fov: 70 }
      }
    },
    fishing: {
      label: "Fishing boat",
      V: 7,
      wake: 1.2,
      twin: 1,
      trim: 0.03,
      heel: 0,
      buzz: 1.4,
      scene: "overcast",
      view: "deck",
      model: { length: 13, draft: 0.45 },
      cams: {
        chase: { pos: [-30, 7.5, 14], tgt: [5, 2.5, -2], fov: 40 },
        drone: { pos: [-6, 40, 0], tgt: [-6, 0, 0], up: [0, 0, -1], fov: 50 },
        side: { pos: [6, 3.8, 24], tgt: [0, 2.8, 0], fov: 38 },
        deck: { pos: [0.75, 4.1, 0], tgt: [-30, -1.6, 0], fov: 74, local: true }
      }
    },
    sailboat: {
      label: "Sailing yacht",
      V: 4.5,
      wake: 0.35,
      twin: 0,
      trim: 0.01,
      heel: 0.17,
      buzz: 0,
      scene: "clear",
      view: "chase",
      model: { length: 12.5, draft: 0.58 },
      cams: {
        chase: { pos: [-30, 14, -17], tgt: [3, 6.5, 0.5], fov: 42 },
        drone: { pos: [-6, 50, 0], tgt: [-6, 0, 0], up: [0, 0, -1], fov: 55 },
        side: { pos: [5, 4, 34], tgt: [0, 7.5, 0], fov: 42 },
        deck: { pos: [-4.8, 2.3, -0.4], tgt: [30, 6, 1.5], fov: 76, local: true }
      }
    }
  };
  var boatName = BOATS[qs.get("boat")] ? qs.get("boat") : "yacht";
  var BOAT = BOATS[boatName];
  var sceneName = SCENES[qs.get("scene")] ? qs.get("scene") : SCENES[BOAT.scene] ? BOAT.scene : "sunset";
  var viewName = BOAT.cams[qs.get("view")] ? qs.get("view") : BOAT.view;
  var P = { ...BASE, ...SCENES[sceneName] };
  for (const k of Object.keys(BASE)) if (qs.has(k) && isFinite(+qs.get(k))) P[k] = +qs.get(k);
  var LOOP = Math.min(300, Math.max(10, num("loop", 30)));
  var V = num("speed", BOAT.V);
  var DIST = V * LOOP;
  var FPS = num("fps", 30);
  var FRAMES = Math.round(LOOP * FPS);
  var DT = LOOP / FRAMES;
  var WARMUP = num("warmup", 12);
  var CASC_REF = [1103, 211.3, 39.7, 7.43, 1.393, 0.2617];
  var CASC_L = [];
  {
    let prevN = 0;
    for (let c = 0; c < CASCADES; c++) {
      const n = Math.max(prevN + 1, Math.round(DIST / CASC_REF[c]));
      CASC_L.push(DIST / n);
      prevN = n;
    }
  }
  var snapScale = (s) => Math.max(1, Math.round(s * DIST)) / DIST;
  var FOAM_CONSTS = `
const FOAM_S0: f32 = ${snapScale(0.23)};
const FOAM_S1: f32 = ${snapScale(0.71)};
const FOAM_S2: f32 = ${snapScale(1.9)};
const FOAM_S3: f32 = ${snapScale(0.011)};
const FOAM_S4: f32 = ${snapScale(0.035)};
`;
  function fail(msg) {
    const el = document.getElementById("err");
    if (el) {
      el.style.display = "block";
      el.textContent = msg;
    }
    window.oceanPro = { error: msg };
    throw new Error(msg);
  }
  function mat4Mul(out, a, b) {
    for (let c = 0; c < 4; c++) {
      const b0 = b[c * 4], b1 = b[c * 4 + 1], b2 = b[c * 4 + 2], b3 = b[c * 4 + 3];
      out[c * 4] = a[0] * b0 + a[4] * b1 + a[8] * b2 + a[12] * b3;
      out[c * 4 + 1] = a[1] * b0 + a[5] * b1 + a[9] * b2 + a[13] * b3;
      out[c * 4 + 2] = a[2] * b0 + a[6] * b1 + a[10] * b2 + a[14] * b3;
      out[c * 4 + 3] = a[3] * b0 + a[7] * b1 + a[11] * b2 + a[15] * b3;
    }
  }
  var sub3 = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
  var cross3 = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  var norm3 = (v) => {
    const l = Math.hypot(...v) || 1;
    return [v[0] / l, v[1] / l, v[2] / l];
  };
  var dot3 = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  async function loadBoatGeometry() {
    const b64 = window.OCEAN_MODELS && window.OCEAN_MODELS[boatName];
    if (b64 && qs.get("model") !== "0") {
      if (!window.THREE) fail("three.js (vendor/three.bundle.js) is needed to read 3D models.");
      const bin = atob(b64), buf = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i++) buf[i] = bin.charCodeAt(i);
      const loader = new THREE.GLTFLoader();
      loader.setMeshoptDecoder(THREE.MeshoptDecoder);
      const gltf = await new Promise((res, rej) => loader.parse(buf.buffer, "", res, rej));
      const opts = { length: num("length", BOAT.model.length), draft: BOAT.model.draft, flip: qs.get("flip") === "1" };
      if (qs.has("waterline")) opts.waterline = num("waterline", 1);
      return fromGLTF(THREE, gltf.scene, opts);
    }
    if (boatName !== "yacht") fail(`No 3D model for the ${BOAT.label.toLowerCase()}: add models/${boatName}.glb and run tools/pack-models.mjs. The realistic renderer only builds the yacht itself.`);
    return buildYacht();
  }
  async function main() {
    if (!navigator.gpu) fail("WebGPU is not available in this browser.");
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) fail("No WebGPU adapter found.");
    const device = await adapter.requestDevice();
    window.__oceanGPU = { adapter, device };
    device.lost.then((i) => console.error("device lost: " + i.message));
    let gpuError = null;
    device.onuncapturederror = (e) => {
      gpuError = e.error.message;
      console.error("GPU error: " + e.error.message);
    };
    const canvas = document.getElementById("c");
    const ctx = canvas.getContext("webgpu");
    const swapFormat = navigator.gpu.getPreferredCanvasFormat();
    ctx.configure({ device, format: swapFormat, alphaMode: "opaque", usage: TU.RENDER_ATTACHMENT | TU.COPY_DST });
    async function shader(code, label) {
      const m = device.createShaderModule({ code, label });
      const info = await m.getCompilationInfo();
      let bad = false;
      for (const msg of info.messages) {
        if (msg.type === "error") bad = true;
        if (msg.type !== "info") console.error(`[${label}] ${msg.type} ${msg.lineNum}:${msg.linePos} ${msg.message}
    ${code.split("\n")[msg.lineNum - 1]}`);
      }
      if (bad) fail("Shader compile error in " + label + " (see console)");
      return m;
    }
    async function computePipe(code, label) {
      const module = await shader(code, label);
      return device.createComputePipelineAsync({ label, layout: "auto", compute: { module, entryPoint: "main" } });
    }
    const C = CASCADES;
    const SIM_MIP_COUNT = Math.log2(N) + 1;
    const specBytes = N * N * C * 16;
    const mkBuf = (size, usage, label) => device.createBuffer({ size, usage, label });
    const simUBO = mkBuf(224, GU.UNIFORM | GU.COPY_DST, "sim ubo");
    const h0Buf = mkBuf(specBytes, GU.STORAGE, "h0");
    const specA = mkBuf(specBytes, GU.STORAGE, "specA");
    const specB = mkBuf(specBytes, GU.STORAGE, "specB");
    const tmpA = mkBuf(specBytes, GU.STORAGE, "tmpA");
    const tmpB = mkBuf(specBytes, GU.STORAGE, "tmpB");
    const mkSimTex = (label) => device.createTexture({
      label,
      size: [N, N, C],
      format: "rgba16float",
      mipLevelCount: SIM_MIP_COUNT,
      usage: TU.TEXTURE_BINDING | TU.STORAGE_BINDING
    });
    const dispTex = mkSimTex("disp");
    const derivTex = mkSimTex("deriv");
    const momTex = mkSimTex("moments");
    const lvl = (t, m, dim = "2d-array") => t.createView({ dimension: dim, baseMipLevel: m, mipLevelCount: 1 });
    const fullArr = (t) => t.createView({ dimension: "2d-array" });
    const [
      initPipe,
      evolvePipe,
      rowsPipe,
      colsPipe,
      simMipPipe,
      foamGenPipe,
      rgba8MipPipe,
      transPipe,
      msPipe,
      skyViewPipe,
      envPipe,
      cubeMipPipe,
      foamUpdPipe,
      arr16MipPipe,
      sunVisPipe,
      probePipe
    ] = await Promise.all([
      computePipe(SIM_INIT, "sim init"),
      computePipe(SIM_EVOLVE, "sim evolve"),
      computePipe(FFT_ROWS, "fft rows"),
      computePipe(FFT_COLS, "fft cols"),
      computePipe(SIM_MIPS, "sim mips"),
      computePipe(FOAM_TEX, "foam tex"),
      computePipe(RGBA8_MIPS, "rgba8 mips"),
      computePipe(TRANS_LUT, "transmittance"),
      computePipe(MS_LUT, "multiscatter"),
      computePipe(SKYVIEW_LUT, "skyview"),
      computePipe(ENV_CUBE, "env cube"),
      computePipe(CUBE_MIPS, "cube mips"),
      computePipe(FOAM_UPDATE, "foam update"),
      computePipe(ARRAY16_MIPS, "array16 mips"),
      computePipe(SUN_VIS, "sun visibility"),
      computePipe(BOAT_PROBE, "boat probe")
    ]);
    const bg = (pipe, entries, label, group = 0) => device.createBindGroup({
      label,
      layout: pipe.getBindGroupLayout(group),
      entries: entries.map((r, i) => ({ binding: i, resource: r }))
    });
    const bgInit = bg(initPipe, [{ buffer: simUBO }, { buffer: h0Buf }], "init");
    const bgEvolve = bg(evolvePipe, [{ buffer: simUBO }, { buffer: h0Buf }, { buffer: specA }, { buffer: specB }], "evolve");
    const bgRows = bg(rowsPipe, [{ buffer: specA }, { buffer: specB }, { buffer: tmpA }, { buffer: tmpB }], "rows");
    const bgCols = bg(colsPipe, [{ buffer: tmpA }, { buffer: tmpB }, lvl(dispTex, 0), lvl(derivTex, 0), lvl(momTex, 0)], "cols");
    const bgSimMips = [];
    for (let m = 1; m < SIM_MIP_COUNT; m++) {
      bgSimMips.push(bg(simMipPipe, [
        lvl(dispTex, m - 1),
        lvl(derivTex, m - 1),
        lvl(momTex, m - 1),
        lvl(dispTex, m),
        lvl(derivTex, m),
        lvl(momTex, m)
      ], "simmip" + m));
    }
    const FOAM_SIZE = 512, FOAM_MIPS = 10;
    const foamTex = device.createTexture({
      label: "foam detail",
      size: [FOAM_SIZE, FOAM_SIZE],
      format: "rgba8unorm",
      mipLevelCount: FOAM_MIPS,
      usage: TU.TEXTURE_BINDING | TU.STORAGE_BINDING
    });
    {
      const enc = device.createCommandEncoder();
      const cp = enc.beginComputePass();
      cp.setPipeline(foamGenPipe);
      cp.setBindGroup(0, bg(foamGenPipe, [lvl(foamTex, 0, "2d")]));
      cp.dispatchWorkgroups(FOAM_SIZE / 8, FOAM_SIZE / 8);
      cp.setPipeline(rgba8MipPipe);
      for (let m = 1; m < FOAM_MIPS; m++) {
        const s = FOAM_SIZE >> m;
        cp.setBindGroup(0, bg(rgba8MipPipe, [lvl(foamTex, m - 1, "2d"), lvl(foamTex, m, "2d")]));
        cp.dispatchWorkgroups(Math.ceil(s / 8), Math.ceil(s / 8));
      }
      cp.end();
      device.queue.submit([enc.finish()]);
    }
    const boatUBO = mkBuf(176, GU.UNIFORM | GU.COPY_DST, "boat ubo");
    const boatF = new Float32Array(44);
    const FOAM_LEVEL_SIZE = [128, 1024];
    const FOAMW_MIPS = Math.log2(FOAM_RES) + 1;
    const foamW = [0, 1].map((i) => device.createTexture({
      label: "foam world " + i,
      size: [FOAM_RES, FOAM_RES, 2],
      format: "rgba16float",
      mipLevelCount: FOAMW_MIPS,
      usage: TU.TEXTURE_BINDING | TU.STORAGE_BINDING
    }));
    const foamUBO = mkBuf(144, GU.UNIFORM | GU.COPY_DST, "foam ubo");
    const foamUArr = new ArrayBuffer(144);
    const foamUI = new Int32Array(foamUArr);
    const foamUF = new Float32Array(foamUArr);
    const foamSnap = new Int32Array(4);
    let foamFirst = true;
    let foamCur = 0;
    const samplerSim = device.createSampler({
      magFilter: "linear",
      minFilter: "linear",
      mipmapFilter: "linear",
      addressModeU: "repeat",
      addressModeV: "repeat"
    });
    const bgFoamUpd = [0, 1].map((i) => bg(foamUpdPipe, [
      { buffer: foamUBO },
      lvl(foamW[1 - i], 0),
      lvl(foamW[i], 0),
      fullArr(dispTex),
      fullArr(derivTex),
      samplerSim,
      samplerSim
    ], "foamupd" + i));
    const bgFoamMips = [0, 1].map((i) => {
      const out = [];
      for (let m = 1; m < FOAMW_MIPS; m++) out.push(bg(arr16MipPipe, [lvl(foamW[i], m - 1), lvl(foamW[i], m)]));
      return out;
    });
    const atmoUBO = mkBuf(16, GU.UNIFORM | GU.COPY_DST, "atmo ubo");
    const mkLut = (w, h, label) => device.createTexture({ label, size: [w, h], format: "rgba16float", usage: TU.TEXTURE_BINDING | TU.STORAGE_BINDING });
    const transLUT = mkLut(256, 64, "transmittance");
    const msLUT = mkLut(32, 32, "multiscatter");
    const skyLUT = mkLut(SKY_W, SKY_H, "skyview");
    const envTex = device.createTexture({
      label: "env cube",
      size: [ENV_SIZE, ENV_SIZE, 6],
      format: "rgba16float",
      mipLevelCount: ENV_MIPS,
      usage: TU.TEXTURE_BINDING | TU.STORAGE_BINDING
    });
    const envCubeView = envTex.createView({ dimension: "cube" });
    const sampClamp = device.createSampler({
      magFilter: "linear",
      minFilter: "linear",
      mipmapFilter: "linear",
      addressModeU: "clamp-to-edge",
      addressModeV: "clamp-to-edge",
      addressModeW: "clamp-to-edge"
    });
    const sampRepeat = device.createSampler({
      magFilter: "linear",
      minFilter: "linear",
      mipmapFilter: "linear",
      addressModeU: "repeat",
      addressModeV: "repeat",
      maxAnisotropy: 16
    });
    const frameUBO = mkBuf(384, GU.UNIFORM | GU.COPY_DST, "frame ubo");
    const sunVisBuf = mkBuf(16, GU.STORAGE, "sun visibility");
    const transView = transLUT.createView(), msView = msLUT.createView(), skyView = skyLUT.createView();
    const bgTrans = bg(transPipe, [{ buffer: atmoUBO }, transView], "trans");
    const bgMS = bg(msPipe, [{ buffer: atmoUBO }, transView, sampClamp, msView], "ms");
    const bgSkyView = bg(skyViewPipe, [{ buffer: frameUBO }, transView, msView, sampClamp, skyView], "skyview");
    const bgEnv = bg(envPipe, [{ buffer: frameUBO }, transView, skyView, sampClamp, lvl(envTex, 0)], "env");
    const bgSunVis = bg(sunVisPipe, [{ buffer: frameUBO }, transView, skyView, sampClamp, { buffer: sunVisBuf }], "sunvis");
    const bgCubeMips = [];
    for (let m = 1; m < ENV_MIPS; m++) bgCubeMips.push(bg(cubeMipPipe, [lvl(envTex, m - 1), lvl(envTex, m)], "cubemip" + m));
    const probePts = new Float32Array(64);
    const probeUBO = mkBuf(256, GU.UNIFORM | GU.COPY_DST, "probe pts");
    const probeOut = mkBuf(128, GU.STORAGE | GU.COPY_SRC, "probe out");
    const probeRead = mkBuf(128, GU.MAP_READ | GU.COPY_DST, "probe read");
    const bgProbe = bg(probePipe, [{ buffer: frameUBO }, fullArr(dispTex), samplerSim, { buffer: probeOut }, { buffer: probeUBO }], "probe");
    const MSAA = 4;
    const HDR = "rgba16float";
    const [oceanMod, skyMod, downMod, upMod, compMod, boatMod, shadowMod] = await Promise.all([
      shader(FOAM_CONSTS + OCEAN, "ocean"),
      shader(SKY_BG, "sky"),
      shader(BLOOM_DOWN, "bloom down"),
      shader(BLOOM_UP, "bloom up"),
      shader(COMPOSITE, "composite"),
      shader(BOAT_RENDER, "boat"),
      shader(BOAT_SHADOW, "boat shadow")
    ]);
    const oceanPipe = await device.createRenderPipelineAsync({
      label: "ocean",
      layout: "auto",
      vertex: { module: oceanMod, entryPoint: "vs", buffers: [{ arrayStride: 16, stepMode: "instance", attributes: [{ shaderLocation: 0, offset: 0, format: "float32x4" }] }] },
      fragment: { module: oceanMod, entryPoint: "fs", targets: [{ format: HDR }] },
      primitive: { topology: "triangle-list", cullMode: "none" },
      depthStencil: { format: "depth32float", depthWriteEnabled: true, depthCompare: "greater" },
      multisample: { count: MSAA }
    });
    const skyPipe = await device.createRenderPipelineAsync({
      label: "sky",
      layout: "auto",
      vertex: { module: skyMod, entryPoint: "vs" },
      fragment: { module: skyMod, entryPoint: "fs", targets: [{ format: HDR }] },
      primitive: { topology: "triangle-list" },
      depthStencil: { format: "depth32float", depthWriteEnabled: false, depthCompare: "equal" },
      multisample: { count: MSAA }
    });
    const vtxLayout = [{ arrayStride: 32, attributes: [
      { shaderLocation: 0, offset: 0, format: "float32x3" },
      { shaderLocation: 1, offset: 12, format: "float32x3" },
      { shaderLocation: 2, offset: 24, format: "float32x2" }
    ] }];
    const boatPipe = await device.createRenderPipelineAsync({
      label: "boat",
      layout: "auto",
      vertex: { module: boatMod, entryPoint: "vs", buffers: vtxLayout },
      fragment: { module: boatMod, entryPoint: "fs", targets: [{ format: HDR }] },
      primitive: { topology: "triangle-list", cullMode: "none" },
      depthStencil: { format: "depth32float", depthWriteEnabled: true, depthCompare: "greater" },
      multisample: { count: MSAA }
    });
    const SHADOW_RES = 2048;
    const shadowTex = device.createTexture({ size: [SHADOW_RES, SHADOW_RES], format: "depth32float", usage: TU.RENDER_ATTACHMENT | TU.TEXTURE_BINDING });
    const shadowPipe = await device.createRenderPipelineAsync({
      label: "boat shadow",
      layout: "auto",
      vertex: { module: shadowMod, entryPoint: "vs", buffers: [{ arrayStride: 32, attributes: [{ shaderLocation: 0, offset: 0, format: "float32x3" }] }] },
      primitive: { topology: "triangle-list", cullMode: "none" },
      depthStencil: { format: "depth32float", depthWriteEnabled: true, depthCompare: "less", depthBias: 2, depthBiasSlopeScale: 2 }
    });
    const shadowUBO = mkBuf(64, GU.UNIFORM | GU.COPY_DST, "shadow ubo");
    const shadowSamp = device.createSampler({ compare: "less", magFilter: "linear", minFilter: "linear" });
    const downPipe = await device.createRenderPipelineAsync({
      label: "bloom down",
      layout: "auto",
      vertex: { module: downMod, entryPoint: "vs" },
      fragment: { module: downMod, entryPoint: "fs", targets: [{ format: HDR }] }
    });
    const upPipe = await device.createRenderPipelineAsync({
      label: "bloom up",
      layout: "auto",
      vertex: { module: upMod, entryPoint: "vs" },
      fragment: { module: upMod, entryPoint: "fs", targets: [{ format: HDR, blend: { color: { srcFactor: "one", dstFactor: "one", operation: "add" }, alpha: { srcFactor: "one", dstFactor: "one", operation: "add" } } }] }
    });
    const compPipe = await device.createRenderPipelineAsync({
      label: "composite",
      layout: "auto",
      vertex: { module: compMod, entryPoint: "vs" },
      fragment: { module: compMod, entryPoint: "fs", targets: [{ format: swapFormat }] }
    });
    const bgOcean = [0, 1].map((i) => bg(oceanPipe, [
      { buffer: frameUBO },
      transView,
      skyView,
      sampClamp,
      fullArr(dispTex),
      fullArr(derivTex),
      fullArr(momTex),
      foamTex.createView(),
      envCubeView,
      sampRepeat,
      fullArr(foamW[i]),
      { buffer: sunVisBuf },
      { buffer: boatUBO }
    ], "ocean" + i));
    const bgSky = bg(skyPipe, [{ buffer: frameUBO }, transView, skyView, sampClamp], "sky");
    const bgBoat = bg(boatPipe, [
      { buffer: frameUBO },
      transView,
      skyView,
      sampClamp,
      envCubeView,
      { buffer: sunVisBuf },
      { buffer: boatUBO },
      { buffer: shadowUBO },
      shadowTex.createView(),
      shadowSamp
    ], "boat");
    const bgShadow = bg(shadowPipe, [{ buffer: shadowUBO }], "shadow");
    const geo = await loadBoatGeometry();
    const dims = geo.dims;
    const white = device.createTexture({ size: [1, 1], format: "rgba8unorm-srgb", usage: TU.TEXTURE_BINDING | TU.COPY_DST });
    device.queue.writeTexture({ texture: white }, new Uint8Array([255, 255, 255, 255]), {}, [1, 1]);
    const texSamp = device.createSampler({ magFilter: "linear", minFilter: "linear", mipmapFilter: "linear", addressModeU: "repeat", addressModeV: "repeat" });
    const texCache = /* @__PURE__ */ new Map();
    async function gpuTexture(img) {
      if (texCache.has(img)) return texCache.get(img);
      const bmp = img instanceof ImageBitmap ? img : await createImageBitmap(img);
      const t = device.createTexture({ size: [bmp.width, bmp.height], format: "rgba8unorm-srgb", usage: TU.TEXTURE_BINDING | TU.COPY_DST | TU.RENDER_ATTACHMENT });
      device.queue.copyExternalImageToTexture({ source: bmp, flipY: false }, { texture: t }, [bmp.width, bmp.height]);
      texCache.set(img, t);
      return t;
    }
    const draws = [];
    for (const part of geo.parts) {
      const nv = part.pos.length / 3;
      const vb = new Float32Array(nv * 8);
      for (let v = 0; v < nv; v++) {
        vb.set(part.pos.subarray(v * 3, v * 3 + 3), v * 8);
        vb.set(part.nrm.subarray(v * 3, v * 3 + 3), v * 8 + 3);
        vb.set(part.uv.subarray(v * 2, v * 2 + 2), v * 8 + 6);
      }
      const vbuf = mkBuf(vb.byteLength, GU.VERTEX | GU.COPY_DST, "boat vb");
      device.queue.writeBuffer(vbuf, 0, vb);
      const ibuf = mkBuf(Math.ceil(part.idx.byteLength / 4) * 4, GU.INDEX | GU.COPY_DST, "boat ib");
      device.queue.writeBuffer(ibuf, 0, part.idx);
      const m = part.mat;
      const mu = new Float32Array([...m.base, m.metal, m.rough, m.coat || 0, m.mode || 0, ...m.emissive || [0, 0, 0], m.map ? 1 : 0]);
      const mbuf = mkBuf(48, GU.UNIFORM | GU.COPY_DST, "boat mat");
      device.queue.writeBuffer(mbuf, 0, mu);
      const tex = m.map ? await gpuTexture(m.map) : white;
      draws.push({ vbuf, ibuf, count: part.idx.length, bg: bg(boatPipe, [{ buffer: mbuf }, tex.createView(), texSamp], "mat", 1) });
    }
    const XB = dims.XB ?? dims.LB * 0.8;
    let bmin = [1e9, 1e9, 1e9], bmax = [-1e9, -1e9, -1e9];
    for (const part of geo.parts) for (let i = 0; i < part.pos.length; i += 3) for (let k = 0; k < 3; k++) {
      bmin[k] = Math.min(bmin[k], part.pos[i + k]);
      bmax[k] = Math.max(bmax[k], part.pos[i + k]);
    }
    const bCentre = bmin.map((v, k) => (v + bmax[k]) / 2), bRadius = Math.hypot(...bmax.map((v, k) => v - bmin[k])) / 2;
    const GRID = 32, LEAF = 2, R0 = 8 * LEAF, LODS = 17, MAX_DIST = 16e4, MAX_NODES = 8192;
    const idx = new Uint16Array(GRID * GRID * 6);
    {
      let k = 0;
      for (let j = 0; j < GRID; j++) for (let i = 0; i < GRID; i++) {
        const a = i + j * (GRID + 1), b = a + 1, c = a + GRID + 1, d = c + 1;
        idx[k++] = a;
        idx[k++] = c;
        idx[k++] = d;
        idx[k++] = a;
        idx[k++] = d;
        idx[k++] = b;
      }
    }
    const idxBuf = mkBuf(idx.byteLength, GU.INDEX | GU.COPY_DST, "grid idx");
    device.queue.writeBuffer(idxBuf, 0, idx);
    const instData = new Float32Array(MAX_NODES * 4);
    const instBuf = mkBuf(instData.byteLength, GU.VERTEX | GU.COPY_DST, "instances");
    const BLOOM_LEVELS = 8;
    const linSamp = device.createSampler({ magFilter: "linear", minFilter: "linear", addressModeU: "clamp-to-edge", addressModeV: "clamp-to-edge" });
    const compUBO = mkBuf(32, GU.UNIFORM | GU.COPY_DST, "composite ubo");
    const compData = new Float32Array(8);
    const bloomUBOs = [];
    for (let i = 0; i < BLOOM_LEVELS * 2; i++) bloomUBOs.push(mkBuf(16, GU.UNIFORM | GU.COPY_DST, "bloom ubo"));
    let W = 0, H = 0;
    let colorMS, depthMS, hdrTex, bloomTex;
    const mainPass = {
      colorAttachments: [{ view: null, resolveTarget: null, clearValue: { r: 0, g: 0, b: 0, a: 1 }, loadOp: "clear", storeOp: "discard" }],
      depthStencilAttachment: { view: null, depthClearValue: 0, depthLoadOp: "clear", depthStoreOp: "discard" }
    };
    const shadowPass = { colorAttachments: [], depthStencilAttachment: { view: shadowTex.createView(), depthClearValue: 1, depthLoadOp: "clear", depthStoreOp: "store" } };
    const compPass = { colorAttachments: [{ view: null, clearValue: { r: 0, g: 0, b: 0, a: 1 }, loadOp: "clear", storeOp: "store" }] };
    let downPasses = [], upPasses = [], downBGs = [], upBGs = [], bgComp = null;
    let offTex = null;
    function resize(w, h) {
      if (w === W && h === H) return;
      W = w;
      H = h;
      canvas.width = W;
      canvas.height = H;
      for (const t of [colorMS, depthMS, hdrTex, bloomTex]) t && t.destroy();
      colorMS = device.createTexture({ size: [W, H], format: HDR, sampleCount: MSAA, usage: TU.RENDER_ATTACHMENT });
      depthMS = device.createTexture({ size: [W, H], format: "depth32float", sampleCount: MSAA, usage: TU.RENDER_ATTACHMENT });
      hdrTex = device.createTexture({ size: [W, H], format: HDR, usage: TU.RENDER_ATTACHMENT | TU.TEXTURE_BINDING });
      const bw = Math.max(1, W >> 1), bh = Math.max(1, H >> 1);
      bloomTex = device.createTexture({ size: [bw, bh], format: HDR, mipLevelCount: BLOOM_LEVELS, usage: TU.RENDER_ATTACHMENT | TU.TEXTURE_BINDING });
      mainPass.colorAttachments[0].view = colorMS.createView();
      mainPass.colorAttachments[0].resolveTarget = hdrTex.createView();
      mainPass.depthStencilAttachment.view = depthMS.createView();
      const hdrView = hdrTex.createView();
      const bv = [];
      for (let m = 0; m < BLOOM_LEVELS; m++) bv.push(bloomTex.createView({ baseMipLevel: m, mipLevelCount: 1 }));
      downPasses = [];
      upPasses = [];
      downBGs = [];
      upBGs = [];
      const tmp = new Float32Array(4);
      for (let m = 0; m < BLOOM_LEVELS; m++) {
        const srcW = m === 0 ? W : Math.max(1, bw >> m - 1), srcH = m === 0 ? H : Math.max(1, bh >> m - 1);
        tmp[0] = 1 / srcW;
        tmp[1] = 1 / srcH;
        tmp[2] = m === 0 ? 1 : 0;
        tmp[3] = 0;
        device.queue.writeBuffer(bloomUBOs[m], 0, tmp);
        downBGs.push(bg(downPipe, [m === 0 ? hdrView : bv[m - 1], linSamp, { buffer: bloomUBOs[m] }]));
        downPasses.push({ colorAttachments: [{ view: bv[m], clearValue: { r: 0, g: 0, b: 0, a: 1 }, loadOp: "clear", storeOp: "store" }] });
      }
      for (let m = BLOOM_LEVELS - 1; m > 0; m--) {
        const sw = Math.max(1, bw >> m), sh = Math.max(1, bh >> m);
        tmp[0] = 1 / sw;
        tmp[1] = 1 / sh;
        tmp[2] = 1;
        tmp[3] = 1;
        const ubo = bloomUBOs[BLOOM_LEVELS + m];
        device.queue.writeBuffer(ubo, 0, tmp);
        upBGs.push(bg(upPipe, [bv[m], linSamp, { buffer: ubo }]));
        upPasses.push({ colorAttachments: [{ view: bv[m - 1], loadOp: "load", storeOp: "store" }] });
      }
      bgComp = bg(compPipe, [hdrView, bv[0], linSamp, { buffer: compUBO }]);
      offTex && offTex.destroy();
      offTex = device.createTexture({ size: [W, H], format: swapFormat, usage: TU.RENDER_ATTACHMENT | TU.COPY_SRC });
    }
    const simF = new Float32Array(56);
    const frameF = new Float32Array(96);
    const atmoF = new Float32Array(4);
    const proj = new Float32Array(16), view = new Float32Array(16), viewProj = new Float32Array(16);
    function writeSimStatic(time, dt) {
      for (let c = 0; c < C; c++) {
        const L = CASC_L[c];
        simF[c] = L;
        simF[8 + c] = c === 0 ? Math.PI / L : 2 * Math.PI * 6 / L;
        simF[16 + c] = c < C - 1 ? 2 * Math.PI * 6 / CASC_L[c + 1] : Math.PI * N / L * 0.98;
      }
      for (let c = 1; c < C; c++) simF[8 + c] = simF[16 + c - 1];
      simF[40] = P.windSpeed;
      simF[41] = P.windDir * D2R;
      simF[42] = P.fetchKm * 1e3;
      simF[43] = 3.3;
      const ks = 2 * Math.PI / P.swellLength;
      simF[44] = P.swellHs;
      simF[45] = P.swellDir * D2R;
      simF[46] = Math.sqrt(9.81 * ks);
      simF[47] = P.swellSpread;
      simF[48] = P.choppy;
      simF[49] = 12e-4;
      simF[50] = LOOP;
      simF[51] = num("seed", 1);
      simF[52] = time;
      simF[53] = dt;
      simF[54] = 2.2;
      simF[55] = 1;
      device.queue.writeBuffer(simUBO, 0, simF);
    }
    const cam = { x: 0, y: 0, z: 0, fovY: 50 * D2R, f: [1, 0, 0], r: [0, 0, 1], u: [0, 1, 0] };
    function setCamera(pos, tgt, up, fovDeg) {
      cam.x = pos[0];
      cam.y = pos[1];
      cam.z = pos[2];
      cam.fovY = fovDeg * D2R;
      cam.f = norm3(sub3(tgt, pos));
      cam.r = norm3(cross3(cam.f, up));
      cam.u = cross3(cam.r, cam.f);
    }
    function writeFrame(time, frameIndex) {
      const aspect = W / H;
      const tanY = Math.tan(cam.fovY * 0.5), tanX = tanY * aspect;
      const near = 0.05;
      proj.fill(0);
      proj[0] = 1 / tanX;
      proj[5] = 1 / tanY;
      proj[11] = -1;
      proj[14] = near;
      const [fx, fy, fz] = cam.f, [rx, ry, rz] = cam.r, [ux, uy, uz] = cam.u;
      view.fill(0);
      view[0] = rx;
      view[4] = ry;
      view[8] = rz;
      view[1] = ux;
      view[5] = uy;
      view[9] = uz;
      view[2] = -fx;
      view[6] = -fy;
      view[10] = -fz;
      view[15] = 1;
      mat4Mul(viewProj, proj, view);
      const f = frameF;
      f.set(viewProj, 0);
      f[16] = cam.x;
      f[17] = cam.y;
      f[18] = cam.z;
      f[19] = time;
      f[20] = rx * tanX;
      f[21] = ry * tanX;
      f[22] = rz * tanX;
      f[23] = 0;
      f[24] = ux * tanY;
      f[25] = uy * tanY;
      f[26] = uz * tanY;
      f[27] = 0;
      f[28] = fx;
      f[29] = fy;
      f[30] = fz;
      f[31] = 0;
      const se = P.sunElev * D2R, sa = P.sunAzim * D2R;
      f[32] = Math.sin(sa) * Math.cos(se);
      f[33] = Math.sin(se);
      f[34] = -Math.cos(sa) * Math.cos(se);
      f[35] = 14;
      f[36] = W;
      f[37] = H;
      f[38] = 1 / W;
      f[39] = 1 / H;
      for (let c = 0; c < C; c++) {
        f[40 + c] = CASC_L[c];
        f[48 + c] = 1 / CASC_L[c];
      }
      f[56] = Math.cos(P.windDir * D2R);
      f[57] = Math.sin(P.windDir * D2R);
      f[58] = P.windSpeed;
      f[64] = R0;
      f[65] = GRID;
      f[66] = FOAM_LEVEL_SIZE[0];
      f[67] = FOAM_LEVEL_SIZE[1];
      f[68] = P.exposure;
      f[69] = P.bloom;
      f[70] = frameIndex;
      f[71] = 0;
      f[72] = P.haze;
      f[73] = Math.max(BOAT.cams[viewName].pos[1], 10) * 1e-3;
      f[74] = P.veil;
      f[75] = 0.9;
      const wc = P.water || [0.016 * P.turbidity, 0.08 * P.turbidity, 0.096 * P.turbidity];
      f[76] = wc[0];
      f[77] = wc[1];
      f[78] = wc[2];
      f[79] = 24e-4;
      const sc = P.sss || [0.035, 0.12, 0.085];
      f[80] = sc[0];
      f[81] = sc[1];
      f[82] = sc[2];
      f[83] = 2.6;
      f[84] = 0.36 + 0.12 * P.foam;
      f[85] = 2.5;
      f[86] = 1;
      f[87] = 1;
      f[88] = (5802e-6 + 44e-4 * P.haze) * 1e-3;
      f[89] = (0.013558 + 44e-4 * P.haze) * 1e-3;
      f[90] = (0.0331 + 44e-4 * P.haze) * 1e-3;
      f[91] = 11e-5;
      f[92] = 1.5;
      f[93] = P.clouds;
      f[94] = 2.2;
      f[95] = 0;
      device.queue.writeBuffer(frameUBO, 0, f);
    }
    const ranges = new Float64Array(LODS);
    for (let l = 0; l < LODS; l++) ranges[l] = R0 * Math.pow(2, l);
    const planes = new Float64Array(16);
    const HBND = 8, XBND = 6;
    let nodeCount = 0, ccx = 0, ccy = 0, ccz = 0;
    function sphereHit(x0, z0, size, r) {
      const dx = Math.max(x0 - ccx, 0, ccx - (x0 + size));
      const dz = Math.max(z0 - ccz, 0, ccz - (z0 + size));
      const dy = Math.max(-HBND - ccy, 0, ccy - HBND);
      return dx * dx + dy * dy + dz * dz <= r * r;
    }
    function inFrustum(x0, z0, size) {
      const minx = x0 - XBND - ccx, maxx = x0 + size + XBND - ccx;
      const minz = z0 - XBND - ccz, maxz = z0 + size + XBND - ccz;
      const miny = -HBND - ccy, maxy = HBND - ccy;
      for (let p = 0; p < 16; p += 4) {
        const a = planes[p], b = planes[p + 1], c = planes[p + 2], d = planes[p + 3];
        const px = a > 0 ? maxx : minx, py = b > 0 ? maxy : miny, pz = c > 0 ? maxz : minz;
        if (a * px + b * py + c * pz + d < 0) return false;
      }
      return true;
    }
    function addNode(x0, z0, size, lod) {
      if (nodeCount >= MAX_NODES) return;
      const o = nodeCount * 4;
      instData[o] = x0;
      instData[o + 1] = z0;
      instData[o + 2] = size;
      instData[o + 3] = lod;
      nodeCount++;
    }
    function selectNode(x0, z0, size, lod) {
      if (!sphereHit(x0, z0, size, ranges[lod])) return false;
      if (!sphereHit(x0, z0, size, MAX_DIST) || !inFrustum(x0, z0, size)) return true;
      if (lod === 0 || !sphereHit(x0, z0, size, ranges[lod - 1])) {
        addNode(x0, z0, size, lod);
        return true;
      }
      const h = size * 0.5;
      for (let i = 0; i < 4; i++) {
        const nx = x0 + (i & 1) * h, nz = z0 + (i >> 1) * h;
        if (!selectNode(nx, nz, h, lod - 1)) {
          if (sphereHit(nx, nz, h, MAX_DIST) && inFrustum(nx, nz, h)) addNode(nx, nz, h, lod - 1);
        }
      }
      return true;
    }
    function selectAll(shiftX) {
      nodeCount = 0;
      ccx = cam.x - shiftX;
      ccy = cam.y;
      ccz = cam.z;
      const m = viewProj;
      for (let i = 0; i < 4; i++) {
        const r = i >> 1, s = i & 1 ? -1 : 1;
        planes[i * 4] = m[3] + s * m[r];
        planes[i * 4 + 1] = m[7] + s * m[4 + r];
        planes[i * 4 + 2] = m[11] + s * m[8 + r];
        planes[i * 4 + 3] = m[15] + s * m[12 + r];
      }
      const top = LODS - 1, ts = LEAF * Math.pow(2, top);
      const x0 = Math.floor((ccx - MAX_DIST) / ts), x1 = Math.floor((ccx + MAX_DIST) / ts);
      const z0 = Math.floor((ccz - MAX_DIST) / ts), z1 = Math.floor((ccz + MAX_DIST) / ts);
      for (let zi = z0; zi <= z1; zi++) for (let xi = x0; xi <= x1; xi++) selectNode(xi * ts, zi * ts, ts, top);
      for (let k = 0; k < nodeCount; k++) instData[k * 4] += shiftX;
    }
    const HULL_PTS = [];
    for (let i = 0; i <= 5; i++) {
      const x = -dims.LS * 0.85 + (XB * 0.9 + dims.LS * 0.85) * i / 5;
      for (const z of [-dims.HB * 0.8, 0, dims.HB * 0.8]) HULL_PTS.push([x, z]);
    }
    const hullSpan = (XB * 0.9 + dims.LS * 0.85) * 0.8, hullBeam = dims.HB * 1.6;
    const cyc = (hz) => Math.max(1, Math.round(hz * LOOP)) * 2 * Math.PI / LOOP;
    function boatPose(heights, t) {
      let bow = 0, stern = 0, port = 0, stbd = 0, all = 0, nb = 0, ns = 0, np = 0, nst = 0;
      const mid = HULL_PTS[Math.floor(HULL_PTS.length / 2)][0];
      HULL_PTS.forEach(([x, z], i) => {
        const h = heights[i];
        all += h;
        if (x > mid + 0.1) {
          bow += h;
          nb++;
        } else if (x < mid - 0.1) {
          stern += h;
          ns++;
        }
        if (z < 0) {
          port += h;
          np++;
        } else if (z > 0) {
          stbd += h;
          nst++;
        }
      });
      const bz = BOAT.buzz;
      const pitch = Math.atan2(bow / nb - stern / ns, hullSpan) * 0.65 + BOAT.trim + bz * (25e-4 * Math.sin(cyc(9) * t) + 15e-4 * Math.sin(cyc(13) * t + 1.3));
      const roll = Math.atan2(port / np - stbd / nst, hullBeam) * 0.55 + BOAT.heel + 0.012 * Math.sin(cyc(0.11) * t + 0.4) + bz * 2e-3 * Math.sin(cyc(11) * t);
      const yaw = 0.02 * Math.sin(cyc(1 / LOOP) * t + 1) + 8e-3 * Math.sin(cyc(0.17) * t);
      const heave = all / HULL_PTS.length * 0.9 - 0.08 + bz * 8e-3 * Math.sin(cyc(7) * t);
      return { pitch, roll, yaw, heave };
    }
    function boatMatrix(x, pose) {
      const cy = Math.cos(pose.yaw), sy = Math.sin(pose.yaw), cp = Math.cos(pose.pitch), sp = Math.sin(pose.pitch), cr = Math.cos(pose.roll), sr = Math.sin(pose.roll);
      const Ry = [[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], Rz = [[cp, -sp, 0], [sp, cp, 0], [0, 0, 1]], Rx = [[1, 0, 0], [0, cr, -sr], [0, sr, cr]];
      const mul = (A, Bm) => A.map((r) => Bm[0].map((_, j) => r[0] * Bm[0][j] + r[1] * Bm[1][j] + r[2] * Bm[2][j]));
      const R = mul(mul(Ry, Rz), Rx);
      const m = new Float32Array(16), inv = new Float32Array(16);
      for (let c = 0; c < 3; c++) for (let r = 0; r < 3; r++) {
        m[c * 4 + r] = R[r][c];
        inv[c * 4 + r] = R[c][r];
      }
      const t = [x, pose.heave, 0];
      m[12] = t[0];
      m[13] = t[1];
      m[14] = t[2];
      m[15] = 1;
      for (let r = 0; r < 3; r++) inv[12 + r] = -(R[0][r] * t[0] + R[1][r] * t[1] + R[2][r] * t[2]);
      inv[15] = 1;
      return { m, inv, R };
    }
    let started = false;
    let simTime = 0, lastFrameIndex = 0;
    function writeFoamUniforms(dt) {
      for (let l = 0; l < 2; l++) {
        const ts = FOAM_LEVEL_SIZE[l] / FOAM_RES;
        const cx = Math.floor(cam.x / ts), cz = Math.floor(cam.z / ts);
        foamUI[l * 4] = cx;
        foamUI[l * 4 + 1] = cz;
        foamUI[l * 4 + 2] = foamFirst ? cx + 1e5 : foamSnap[l * 2];
        foamUI[l * 4 + 3] = foamFirst ? cz + 1e5 : foamSnap[l * 2 + 1];
        foamSnap[l * 2] = cx;
        foamSnap[l * 2 + 1] = cz;
        foamUF[8 + l] = ts;
      }
      foamFirst = false;
      foamUF[10] = dt;
      foamUF[11] = 0.22;
      foamUF[12] = 0.52 + 0.16 * P.foam;
      foamUF[13] = 2;
      foamUF[14] = 1;
      foamUF[15] = 0;
      for (let c = 0; c < C; c++) {
        foamUF[16 + c] = CASC_L[c];
        foamUF[24 + c] = 1 / CASC_L[c];
      }
      const drift = 0.03 * P.windSpeed;
      foamUF[32] = Math.cos(P.windDir * D2R) * drift;
      foamUF[33] = Math.sin(P.windDir * D2R) * drift;
      device.queue.writeBuffer(foamUBO, 0, foamUArr);
    }
    function cameraFor(t, boatX, pose, B) {
      const c = BOAT.cams[viewName];
      const add = (a, b) => a.map((v, i) => v + b[i]);
      if (c.local) {
        const R = B.R, o = [boatX, pose.heave, 0];
        const toW = (p) => add([R[0][0] * p[0] + R[0][1] * p[1] + R[0][2] * p[2], R[1][0] * p[0] + R[1][1] * p[1] + R[1][2] * p[2], R[2][0] * p[0] + R[2][1] * p[1] + R[2][2] * p[2]], o);
        const up = [R[0][1], R[1][1], R[2][1]];
        return { pos: toW(c.pos), tgt: toW(c.tgt), up, fov: num("fov", c.fov) };
      }
      const w1 = 2 * Math.PI / LOOP, k = Math.max(dims.LB + dims.LS, 10) / 20;
      const pos = [c.pos[0] + 0.8 * k * Math.sin(w1 * t) + boatX, c.pos[1] + 0.35 * k * Math.sin(2 * w1 * t + 0.7) + pose.heave * 0.25, c.pos[2] + 1.2 * k * Math.sin(w1 * t + 2.1)];
      pos[1] = Math.max(pos[1], P.swellHs + 2.5);
      return { pos, tgt: [c.tgt[0] + boatX, c.tgt[1] + pose.heave * 0.4, c.tgt[2]], up: c.up || [0, 1, 0], fov: num("fov", c.fov) };
    }
    async function step(index, render) {
      const t = (index * DT % LOOP + LOOP) % LOOP;
      const boatX = index * DT * V;
      simTime = t;
      writeSimStatic(t, DT);
      for (let i = 0; i < HULL_PTS.length; i++) {
        probePts[i * 2] = boatX + HULL_PTS[i][0];
        probePts[i * 2 + 1] = HULL_PTS[i][1];
      }
      device.queue.writeBuffer(probeUBO, 0, probePts);
      {
        const enc2 = device.createCommandEncoder();
        const cp2 = enc2.beginComputePass();
        if (!started) {
          cp2.setPipeline(initPipe);
          cp2.setBindGroup(0, bgInit);
          cp2.dispatchWorkgroups(N / 16, N / 16, C);
          atmoF[0] = P.haze;
          device.queue.writeBuffer(atmoUBO, 0, atmoF);
          cp2.setPipeline(transPipe);
          cp2.setBindGroup(0, bgTrans);
          cp2.dispatchWorkgroups(256 / 8, 64 / 8);
          cp2.setPipeline(msPipe);
          cp2.setBindGroup(0, bgMS);
          cp2.dispatchWorkgroups(32 / 8, 32 / 8);
        }
        cp2.setPipeline(evolvePipe);
        cp2.setBindGroup(0, bgEvolve);
        cp2.dispatchWorkgroups(N / 16, N / 16, C);
        cp2.setPipeline(rowsPipe);
        cp2.setBindGroup(0, bgRows);
        cp2.dispatchWorkgroups(N, 1, C);
        cp2.setPipeline(colsPipe);
        cp2.setBindGroup(0, bgCols);
        cp2.dispatchWorkgroups(N, 1, C);
        cp2.setPipeline(simMipPipe);
        for (let m = 1; m < SIM_MIP_COUNT; m++) {
          const s = N >> m;
          cp2.setBindGroup(0, bgSimMips[m - 1]);
          cp2.dispatchWorkgroups(Math.ceil(s / 8), Math.ceil(s / 8), C);
        }
        cp2.setPipeline(probePipe);
        cp2.setBindGroup(0, bgProbe);
        cp2.dispatchWorkgroups(1);
        cp2.end();
        enc2.copyBufferToBuffer(probeOut, 0, probeRead, 0, 128);
        device.queue.submit([enc2.finish()]);
      }
      await probeRead.mapAsync(GPUMapMode.READ);
      const heights = Array.from(new Float32Array(probeRead.getMappedRange(0, 128)));
      probeRead.unmap();
      const pose = boatPose(heights, t);
      const Bm = boatMatrix(boatX, pose);
      const cm = cameraFor(t, boatX, pose, Bm);
      setCamera(cm.pos, cm.tgt, cm.up, cm.fov);
      writeFrame(t, index);
      boatF.set(Bm.m, 0);
      boatF.set(Bm.inv, 16);
      boatF.set([dims.LB, dims.LS, dims.HB, XB], 32);
      boatF.set([V, BOAT.wake, 1, BOAT.twin], 36);
      boatF.set([dims.bowPow || 1.5, DT, 0.07, 0], 40);
      device.queue.writeBuffer(boatUBO, 0, boatF);
      writeFoamUniforms(DT);
      selectAll(boatX);
      if (nodeCount) device.queue.writeBuffer(instBuf, 0, instData, 0, nodeCount * 4);
      const se = P.sunElev * D2R, sa = P.sunAzim * D2R;
      const sunW = [Math.sin(sa) * Math.cos(se), Math.sin(se), -Math.cos(sa) * Math.cos(se)];
      const sunL = [0, 1, 2].map((k) => Bm.R[0][k] * sunW[0] + Bm.R[1][k] * sunW[1] + Bm.R[2][k] * sunW[2]);
      const fwd = norm3(sunL.map((v) => -v));
      const upH = Math.abs(fwd[1]) > 0.95 ? [1, 0, 0] : [0, 1, 0];
      const rt = norm3(cross3(fwd, upH)), up = cross3(rt, fwd);
      const r = bRadius * 1.05;
      const eye = bCentre.map((v, k) => v - fwd[k] * r * 2);
      const sh = new Float32Array(16);
      for (let k = 0; k < 3; k++) {
        sh[k * 4] = rt[k] / r;
        sh[k * 4 + 1] = up[k] / r;
        sh[k * 4 + 2] = fwd[k] / (4 * r);
      }
      sh[12] = -dot3(rt, eye) / r;
      sh[13] = -dot3(up, eye) / r;
      sh[14] = -dot3(fwd, eye) / (4 * r);
      sh[15] = 1;
      device.queue.writeBuffer(shadowUBO, 0, sh);
      const enc = device.createCommandEncoder();
      const cp = enc.beginComputePass();
      foamCur = 1 - foamCur;
      cp.setPipeline(foamUpdPipe);
      cp.setBindGroup(0, bgFoamUpd[foamCur]);
      cp.dispatchWorkgroups(FOAM_RES / 8, FOAM_RES / 8, 2);
      cp.setPipeline(arr16MipPipe);
      for (let m = 1; m < FOAMW_MIPS; m++) {
        const s = Math.max(1, FOAM_RES >> m);
        cp.setBindGroup(0, bgFoamMips[foamCur][m - 1]);
        cp.dispatchWorkgroups(Math.ceil(s / 8), Math.ceil(s / 8), 2);
      }
      if (render) {
        cp.setPipeline(skyViewPipe);
        cp.setBindGroup(0, bgSkyView);
        cp.dispatchWorkgroups(Math.ceil(SKY_W / 8), Math.ceil(SKY_H / 8));
        cp.setPipeline(sunVisPipe);
        cp.setBindGroup(0, bgSunVis);
        cp.dispatchWorkgroups(1);
        if (!envDone) {
          cp.setPipeline(envPipe);
          cp.setBindGroup(0, bgEnv);
          cp.dispatchWorkgroups(ENV_SIZE / 8, ENV_SIZE / 8, 6);
          cp.setPipeline(cubeMipPipe);
          for (let m = 1; m < ENV_MIPS; m++) {
            const s = Math.max(1, ENV_SIZE >> m);
            cp.setBindGroup(0, bgCubeMips[m - 1]);
            cp.dispatchWorkgroups(Math.ceil(s / 8), Math.ceil(s / 8), 6);
          }
          envDone = true;
        }
      }
      cp.end();
      started = true;
      if (!render) {
        device.queue.submit([enc.finish()]);
        return null;
      }
      {
        const sp = enc.beginRenderPass(shadowPass);
        sp.setPipeline(shadowPipe);
        sp.setBindGroup(0, bgShadow);
        for (const d of draws) {
          sp.setVertexBuffer(0, d.vbuf);
          sp.setIndexBuffer(d.ibuf, "uint32");
          sp.drawIndexed(d.count);
        }
        sp.end();
      }
      const rp = enc.beginRenderPass(mainPass);
      if (nodeCount) {
        rp.setPipeline(oceanPipe);
        rp.setBindGroup(0, bgOcean[foamCur]);
        rp.setVertexBuffer(0, instBuf);
        rp.setIndexBuffer(idxBuf, "uint16");
        rp.drawIndexed(GRID * GRID * 6, nodeCount);
      }
      rp.setPipeline(boatPipe);
      rp.setBindGroup(0, bgBoat);
      for (const d of draws) {
        rp.setBindGroup(1, d.bg);
        rp.setVertexBuffer(0, d.vbuf);
        rp.setIndexBuffer(d.ibuf, "uint32");
        rp.drawIndexed(d.count);
      }
      rp.setPipeline(skyPipe);
      rp.setBindGroup(0, bgSky);
      rp.draw(3);
      rp.end();
      for (let m = 0; m < BLOOM_LEVELS; m++) {
        const p = enc.beginRenderPass(downPasses[m]);
        p.setPipeline(downPipe);
        p.setBindGroup(0, downBGs[m]);
        p.draw(3);
        p.end();
      }
      for (let i = 0; i < upPasses.length; i++) {
        const p = enc.beginRenderPass(upPasses[i]);
        p.setPipeline(upPipe);
        p.setBindGroup(0, upBGs[i]);
        p.draw(3);
        p.end();
      }
      compData[0] = P.exposure;
      compData[1] = P.bloom;
      compData[2] = (index % 64 + 64) % 64;
      compData[3] = 0.1;
      compData[4] = 1.05;
      compData[5] = 1.06;
      compData[6] = -0.19;
      compData[7] = 0;
      device.queue.writeBuffer(compUBO, 0, compData);
      const swapTex = offTex;
      compPass.colorAttachments[0].view = swapTex.createView();
      const pc = enc.beginRenderPass(compPass);
      pc.setPipeline(compPipe);
      pc.setBindGroup(0, bgComp);
      pc.draw(3);
      pc.end();
      lastFrameIndex = index;
      return { enc, swapTex };
    }
    let envDone = false;
    async function capture(enc, swapTex) {
      const bpr = Math.ceil(W * 4 / 256) * 256;
      const buf = device.createBuffer({ size: bpr * H, usage: GU.MAP_READ | GU.COPY_DST });
      enc.copyTextureToBuffer({ texture: swapTex }, { buffer: buf, bytesPerRow: bpr }, [W, H]);
      device.queue.submit([enc.finish()]);
      await buf.mapAsync(GPUMapMode.READ);
      const src = new Uint8Array(buf.getMappedRange());
      const out = new Uint8Array(W * H * 4);
      const bgra = swapFormat === "bgra8unorm";
      for (let y = 0; y < H; y++) {
        const s0 = y * bpr, d0 = y * W * 4;
        if (!bgra) {
          out.set(src.subarray(s0, s0 + W * 4), d0);
          continue;
        }
        for (let x = 0; x < W * 4; x += 4) {
          out[d0 + x] = src[s0 + x + 2];
          out[d0 + x + 1] = src[s0 + x + 1];
          out[d0 + x + 2] = src[s0 + x];
          out[d0 + x + 3] = 255;
        }
      }
      buf.unmap();
      buf.destroy();
      return out;
    }
    const outW = num("w", 0) || Math.round(innerWidth * (devicePixelRatio || 1));
    const outH = num("h", 0) || Math.round(innerHeight * (devicePixelRatio || 1));
    resize(outW, outH);
    let simIndex = null;
    async function advanceTo(index, warmupSeconds) {
      if (simIndex === null || index <= simIndex) {
        simIndex = index - Math.round(warmupSeconds * FPS) - 1;
        foamFirst = true;
      }
      while (simIndex < index - 1) {
        simIndex++;
        await step(simIndex, false);
      }
      simIndex = index;
      return step(index, true);
    }
    async function renderFrame(index) {
      const r = await advanceTo(index, WARMUP);
      const px = await capture(r.enc, r.swapTex);
      if (gpuError) throw new Error(gpuError);
      return px;
    }
    function toBase64(u8) {
      let s = "";
      for (let i = 0; i < u8.length; i += 32768) s += String.fromCharCode.apply(null, u8.subarray(i, i + 32768));
      return btoa(s);
    }
    window.oceanPro = {
      info: { boat: boatName, scene: sceneName, view: viewName, loop: LOOP, fps: FPS, frames: FRAMES, width: W, height: H, speed: V, distance: DIST, cascades: CASC_L, swapFormat, model: !!(window.OCEAN_MODELS && window.OCEAN_MODELS[boatName]) },
      async frameBase64(index) {
        return toBase64(await renderFrame(index));
      },
      renderFrame
    };
    document.title = "ready";
    if (qs.get("live") === "1") {
      let i = 0;
      const loop = async () => {
        const r = await advanceTo(i++, 0);
        device.queue.submit([r.enc.finish()]);
        requestAnimationFrame(() => {
          const enc = device.createCommandEncoder();
          enc.copyTextureToTexture({ texture: offTex }, { texture: ctx.getCurrentTexture() }, [W, H]);
          device.queue.submit([enc.finish()]);
          loop();
        });
      };
      requestAnimationFrame(loop);
    }
  }
  main().catch((e) => {
    console.error("init failed: " + (e && e.stack || e));
    if (!window.oceanPro) window.oceanPro = { error: String(e && e.message || e) };
  });
})();
