// Ocean Run: drawing the boat (PBR lit by the sun and the sky cube), its sun shadow map, and
// the probe that samples the water height under the hull to rock the boat.
import { FRAME_WGSL, ATMO_CORE_WGSL, ENV_MIPS, CASCADES, BOAT_WGSL } from './common.js';
import { SKY_RENDER_FUNCS } from './sky.js';

export const BOAT_RENDER = FRAME_WGSL + ATMO_CORE_WGSL + /* wgsl */ `
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
` + BOAT_WGSL + SKY_RENDER_FUNCS + /* wgsl */ `
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

export const BOAT_SHADOW = /* wgsl */ `
@group(0) @binding(0) var<uniform> SH: mat4x4f;
@vertex fn vs(@location(0) p: vec3f) -> @builtin(position) vec4f { return SH * vec4f(p, 1.0); }
`;

// Water height at points under the hull (world xz), inverting the horizontal displacement.
export const BOAT_PROBE = FRAME_WGSL + /* wgsl */ `
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
