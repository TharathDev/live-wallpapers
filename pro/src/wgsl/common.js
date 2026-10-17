// Shared constants and WGSL snippets.
export const N = 256;          // FFT resolution per cascade
export const LOGN = 8;
export const CASCADES = 6;
export const SKY_W = 256;      // sky-view LUT size
export const SKY_H = 144;
export const ENV_SIZE = 256;   // environment cube face size
export const ENV_MIPS = 9;

// Per-frame uniform block shared by every render / sky shader.
// Layout must match the Float32Array offsets in main.js (FR_* constants).
export const FRAME_WGSL = /* wgsl */ `
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
`;

export const ATMO_CORE_WGSL = /* wgsl */ `
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
`;

// Hashing helpers used by the procedural texture generators.
export const NOISE_WGSL = /* wgsl */ `
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
`;

// Periodic (tileable) noise for procedural textures. Requires NOISE_WGSL.
export const PERIODIC_WGSL = /* wgsl */ `
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
`;

// ---- Ocean Run: the boat, shared by the foam update, the ocean surface and the boat itself.
// Each shader declares `B: Boat` at its own binding. Layout matches BOAT_* in main.js.
export const BOAT_WGSL = /* wgsl */ `
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
    let band = exp(-pow(az / (HB * 1.1 + max(ss, 0.0) * 0.045), 2.0)) * smoothstep(-2.0, 4.0, ss);
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
`;
