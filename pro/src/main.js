// Ocean Run, realistic renderer (WebGPU).
//
// Built on web-ocean by Maksymilian Dendura (MIT, see pro/LICENSE-web-ocean): FFT ocean,
// CDLOD surface, physically based sky and HDR post. Ocean Run adds a boat that runs through
// the sea, its wake, and frame-exact seamless loops:
//  * the boat really moves forward at speed V and the camera rides along. After one loop it
//    has travelled D = V * loop, so every cascade's patch size and every world-space texture
//    scale is snapped to divide D, and all wave frequencies are multiples of 2*pi / loop;
//  * frames are stepped at a fixed time step, so a frame depends only on its index (after
//    a warm-up that lets the foam reach its steady, periodic state).
import { N, CASCADES, SKY_W, SKY_H, ENV_SIZE, ENV_MIPS } from './wgsl/common.js';
import { SIM_INIT, SIM_EVOLVE, FFT_ROWS, FFT_COLS, SIM_MIPS, FOAM_TEX, RGBA8_MIPS, FOAM_UPDATE, ARRAY16_MIPS, FOAM_RES } from './wgsl/sim.js';
import { TRANS_LUT, MS_LUT, SKYVIEW_LUT, ENV_CUBE, CUBE_MIPS, SUN_VIS } from './wgsl/sky.js';
import { OCEAN, SKY_BG } from './wgsl/ocean.js';
import { BLOOM_DOWN, BLOOM_UP, COMPOSITE } from './wgsl/post.js';
import { BOAT_RENDER, BOAT_SHADOW, BOAT_PROBE } from './wgsl/boat.js';
import { buildYacht, fromGLTF } from './boat.js';

const qs = new URLSearchParams(location.search);
const num = (k, d) => (qs.has(k) && isFinite(+qs.get(k)) ? +qs.get(k) : d);
const D2R = Math.PI / 180;
const GU = GPUBufferUsage;
const TU = GPUTextureUsage;
const smooth = (a, b, x) => { const t = Math.min(1, Math.max(0, (x - a) / (b - a))); return t * t * (3 - 2 * t); };

// ----------------------------------------------------------------------------------------
// Scenes (web-ocean parameters) and boats
// ----------------------------------------------------------------------------------------
const BASE = {
  windSpeed: 7.0, windDir: 180, fetchKm: 90, swellHs: 1.6, swellDir: 160, swellLength: 110, swellSpread: 48,
  choppy: 1.25, turbidity: 1.6, foam: 1.0, sunElev: 4, sunAzim: 70, exposure: 1.35, haze: 3.0,
  clouds: 0.4, veil: 0.36, bloom: 0.1, water: null, sss: null,
};
const SCENES = {
  sunset:   { label: 'Sunset', sunElev: 3.5, sunAzim: 62, clouds: 0.45, veil: 0.4, windSpeed: 8, swellHs: 1.8 },
  golden:   { label: 'Golden hour', sunElev: 9, sunAzim: 115, clouds: 0.2, veil: 0.12, haze: 5, windSpeed: 5, swellHs: 1.0, exposure: 1.2 },
  tropical: { label: 'Tropical noon', sunElev: 52, sunAzim: 130, clouds: 0.3, veil: 0.0, haze: 1.5, windSpeed: 6, swellHs: 0.8,
              exposure: 0.85, water: [0.002, 0.045, 0.065], sss: [0.02, 0.16, 0.15] },
  clear:    { label: 'Clear blue', sunElev: 38, sunAzim: 60, clouds: 0.12, veil: 0.0, haze: 2, windSpeed: 7, swellHs: 1.0,
              exposure: 1.05, water: [0.006, 0.045, 0.085] },
  overcast: { label: 'Overcast swell', sunElev: 24, sunAzim: 40, clouds: 0.85, veil: 0.85, haze: 5, windSpeed: 11, swellHs: 2.2, exposure: 1.6 },
  storm:    { label: 'Storm', sunElev: 14, sunAzim: 75, clouds: 1.0, veil: 1.0, haze: 9, windSpeed: 15, swellHs: 3.0, choppy: 1.4, exposure: 1.9, foam: 1.6 },
};
// Hull and camera settings mirror the WebGL page (index.html). Cameras are boat-relative.
const BOATS = {
  yacht: {
    label: 'Motor yacht', V: 12, wake: 1.3, twin: 1, trim: 0.04, heel: 0, buzz: 1, scene: 'tropical', view: 'drone',
    model: { length: 19, draft: 0.38 },
    cams: {
      chase: { pos: [-38, 9, 18], tgt: [6, 2, -2.5], fov: 40 },
      drone: { pos: [-7, 30, 0], tgt: [-7, 0, 0], up: [0, 0, -1], fov: 50 },
      side: { pos: [10, 4.5, 34], tgt: [0, 2.6, 0], fov: 38 },
      deck: { pos: [-5.3, 4.3, 0.6], tgt: [-40, -2, 0], fov: 70, local: true },
    },
  },
  galleon: {
    label: 'Galleon', V: 3.5, wake: 0.4, twin: 0, trim: 0, heel: 0.03, buzz: 0, scene: 'golden', view: 'side',
    model: { length: 34, draft: 0.45 },
    cams: {
      chase: { pos: [-62, 14, 34], tgt: [8, 12, -4], fov: 42 },
      drone: { pos: [-6, 95, 0], tgt: [-6, 0, 0], up: [0, 0, -1], fov: 55 },
      side: { pos: [34, 5, 54], tgt: [2, 12.5, 0], fov: 40 },
      deck: { pos: [12, 2.6, -17], tgt: [-1, 13, 0], fov: 70 },
    },
  },
  fishing: {
    label: 'Fishing boat', V: 7, wake: 1.2, twin: 1, trim: 0.03, heel: 0, buzz: 1.4, scene: 'overcast', view: 'deck',
    model: { length: 13, draft: 0.45 },
    cams: {
      chase: { pos: [-30, 7.5, 14], tgt: [5, 2.5, -2], fov: 40 },
      drone: { pos: [-6, 40, 0], tgt: [-6, 0, 0], up: [0, 0, -1], fov: 50 },
      side: { pos: [6, 3.8, 24], tgt: [0, 2.8, 0], fov: 38 },
      deck: { pos: [0.75, 4.1, 0], tgt: [-30, -1.6, 0], fov: 74, local: true },
    },
  },
  sailboat: {
    label: 'Sailing yacht', V: 4.5, wake: 0.35, twin: 0, trim: 0.01, heel: 0.17, buzz: 0, scene: 'clear', view: 'chase',
    model: { length: 12.5, draft: 0.58 },
    cams: {
      chase: { pos: [-30, 14, -17], tgt: [3, 6.5, 0.5], fov: 42 },
      drone: { pos: [-6, 50, 0], tgt: [-6, 0, 0], up: [0, 0, -1], fov: 55 },
      side: { pos: [5, 4, 34], tgt: [0, 7.5, 0], fov: 42 },
      deck: { pos: [-4.8, 2.3, -0.4], tgt: [30, 6, 1.5], fov: 76, local: true },
    },
  },
};

const boatName = BOATS[qs.get('boat')] ? qs.get('boat') : 'yacht';
const BOAT = BOATS[boatName];
const sceneName = SCENES[qs.get('scene')] ? qs.get('scene') : (SCENES[BOAT.scene] ? BOAT.scene : 'sunset');
const viewName = BOAT.cams[qs.get('view')] ? qs.get('view') : BOAT.view;
const P = { ...BASE, ...SCENES[sceneName] };
for (const k of Object.keys(BASE)) if (qs.has(k) && isFinite(+qs.get(k))) P[k] = +qs.get(k);

const LOOP = Math.min(300, Math.max(10, num('loop', 30)));
const V = num('speed', BOAT.V);
const DIST = V * LOOP;                       // metres travelled per loop
const FPS = num('fps', 30);
const FRAMES = Math.round(LOOP * FPS);
const DT = LOOP / FRAMES;
const WARMUP = num('warmup', 12);            // seconds of simulation before the first frame (whitecap foam settles)

// Patch sizes: as close as possible to web-ocean's, but dividing the loop distance.
const CASC_REF = [1103.0, 211.3, 39.7, 7.43, 1.393, 0.2617];
const CASC_L = [];
{
  let prevN = 0;
  for (let c = 0; c < CASCADES; c++) {
    const n = Math.max(prevN + 1, Math.round(DIST / CASC_REF[c]));
    CASC_L.push(DIST / n); prevN = n;
  }
}
// World-space texture scales snapped to repeat over the loop distance.
const snapScale = s => Math.max(1, Math.round(s * DIST)) / DIST;
const FOAM_CONSTS = `
const FOAM_S0: f32 = ${snapScale(0.23)};
const FOAM_S1: f32 = ${snapScale(0.71)};
const FOAM_S2: f32 = ${snapScale(1.9)};
const FOAM_S3: f32 = ${snapScale(0.011)};
const FOAM_S4: f32 = ${snapScale(0.035)};
`;

function fail(msg) {
  const el = document.getElementById('err');
  if (el) { el.style.display = 'block'; el.textContent = msg; }
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
const sub3 = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const cross3 = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm3 = v => { const l = Math.hypot(...v) || 1; return [v[0] / l, v[1] / l, v[2] / l]; };
const dot3 = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

// ----------------------------------------------------------------------------------------
// Boat geometry: a packed glTF model if there is one, otherwise the procedural yacht
// ----------------------------------------------------------------------------------------
async function loadBoatGeometry() {
  const b64 = window.OCEAN_MODELS && window.OCEAN_MODELS[boatName];
  if (b64 && qs.get('model') !== '0') {
    if (!window.THREE) fail('three.js (vendor/three.bundle.js) is needed to read 3D models.');
    const bin = atob(b64), buf = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) buf[i] = bin.charCodeAt(i);
    const loader = new THREE.GLTFLoader();
    loader.setMeshoptDecoder(THREE.MeshoptDecoder);
    const gltf = await new Promise((res, rej) => loader.parse(buf.buffer, '', res, rej));
    const opts = { length: num('length', BOAT.model.length), draft: BOAT.model.draft, flip: qs.get('flip') === '1' };
    if (qs.has('waterline')) opts.waterline = num('waterline', 1);
    return fromGLTF(THREE, gltf.scene, opts);
  }
  if (boatName !== 'yacht') fail(`No 3D model for the ${BOAT.label.toLowerCase()}: add models/${boatName}.glb and run tools/pack-models.mjs. The realistic renderer only builds the yacht itself.`);
  return buildYacht();
}

async function main() {
  if (!navigator.gpu) fail('WebGPU is not available in this browser.');
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) fail('No WebGPU adapter found.');
  const device = await adapter.requestDevice();
  // keep the adapter alive: if it is garbage-collected, Chrome drops the WebGPU instance and the device
  window.__oceanGPU = { adapter, device };
  device.lost.then((i) => console.error('device lost: ' + i.message));
  let gpuError = null;
  device.onuncapturederror = (e) => { gpuError = e.error.message; console.error('GPU error: ' + e.error.message); };

  const canvas = document.getElementById('c');
  const ctx = canvas.getContext('webgpu');
  const swapFormat = navigator.gpu.getPreferredCanvasFormat();
  ctx.configure({ device, format: swapFormat, alphaMode: 'opaque', usage: TU.RENDER_ATTACHMENT | TU.COPY_DST });

  async function shader(code, label) {
    const m = device.createShaderModule({ code, label });
    const info = await m.getCompilationInfo();
    let bad = false;
    for (const msg of info.messages) {
      if (msg.type === 'error') bad = true;
      if (msg.type !== 'info') console.error(`[${label}] ${msg.type} ${msg.lineNum}:${msg.linePos} ${msg.message}\n    ${code.split('\n')[msg.lineNum - 1]}`);
    }
    if (bad) fail('Shader compile error in ' + label + ' (see console)');
    return m;
  }
  async function computePipe(code, label) {
    const module = await shader(code, label);
    return device.createComputePipelineAsync({ label, layout: 'auto', compute: { module, entryPoint: 'main' } });
  }

  // --------------------------------------------------------------------------------------
  // Simulation resources
  // --------------------------------------------------------------------------------------
  const C = CASCADES;
  const SIM_MIP_COUNT = Math.log2(N) + 1;
  const specBytes = N * N * C * 16;
  const mkBuf = (size, usage, label) => device.createBuffer({ size, usage, label });
  const simUBO = mkBuf(224, GU.UNIFORM | GU.COPY_DST, 'sim ubo');
  const h0Buf = mkBuf(specBytes, GU.STORAGE, 'h0');
  const specA = mkBuf(specBytes, GU.STORAGE, 'specA');
  const specB = mkBuf(specBytes, GU.STORAGE, 'specB');
  const tmpA = mkBuf(specBytes, GU.STORAGE, 'tmpA');
  const tmpB = mkBuf(specBytes, GU.STORAGE, 'tmpB');
  const mkSimTex = (label) => device.createTexture({
    label, size: [N, N, C], format: 'rgba16float', mipLevelCount: SIM_MIP_COUNT,
    usage: TU.TEXTURE_BINDING | TU.STORAGE_BINDING,
  });
  const dispTex = mkSimTex('disp');
  const derivTex = mkSimTex('deriv');
  const momTex = mkSimTex('moments');
  const lvl = (t, m, dim = '2d-array') => t.createView({ dimension: dim, baseMipLevel: m, mipLevelCount: 1 });
  const fullArr = (t) => t.createView({ dimension: '2d-array' });

  const [initPipe, evolvePipe, rowsPipe, colsPipe, simMipPipe, foamGenPipe, rgba8MipPipe,
    transPipe, msPipe, skyViewPipe, envPipe, cubeMipPipe, foamUpdPipe, arr16MipPipe, sunVisPipe, probePipe] = await Promise.all([
    computePipe(SIM_INIT, 'sim init'), computePipe(SIM_EVOLVE, 'sim evolve'),
    computePipe(FFT_ROWS, 'fft rows'), computePipe(FFT_COLS, 'fft cols'),
    computePipe(SIM_MIPS, 'sim mips'), computePipe(FOAM_TEX, 'foam tex'), computePipe(RGBA8_MIPS, 'rgba8 mips'),
    computePipe(TRANS_LUT, 'transmittance'), computePipe(MS_LUT, 'multiscatter'), computePipe(SKYVIEW_LUT, 'skyview'),
    computePipe(ENV_CUBE, 'env cube'), computePipe(CUBE_MIPS, 'cube mips'), computePipe(FOAM_UPDATE, 'foam update'),
    computePipe(ARRAY16_MIPS, 'array16 mips'), computePipe(SUN_VIS, 'sun visibility'), computePipe(BOAT_PROBE, 'boat probe'),
  ]);

  const bg = (pipe, entries, label, group = 0) => device.createBindGroup({
    label, layout: pipe.getBindGroupLayout(group),
    entries: entries.map((r, i) => ({ binding: i, resource: r })),
  });
  const bgInit = bg(initPipe, [{ buffer: simUBO }, { buffer: h0Buf }], 'init');
  const bgEvolve = bg(evolvePipe, [{ buffer: simUBO }, { buffer: h0Buf }, { buffer: specA }, { buffer: specB }], 'evolve');
  const bgRows = bg(rowsPipe, [{ buffer: specA }, { buffer: specB }, { buffer: tmpA }, { buffer: tmpB }], 'rows');
  const bgCols = bg(colsPipe, [{ buffer: tmpA }, { buffer: tmpB }, lvl(dispTex, 0), lvl(derivTex, 0), lvl(momTex, 0)], 'cols');
  const bgSimMips = [];
  for (let m = 1; m < SIM_MIP_COUNT; m++) {
    bgSimMips.push(bg(simMipPipe, [lvl(dispTex, m - 1), lvl(derivTex, m - 1), lvl(momTex, m - 1),
      lvl(dispTex, m), lvl(derivTex, m), lvl(momTex, m)], 'simmip' + m));
  }

  // foam micro-structure texture (generated once)
  const FOAM_SIZE = 512, FOAM_MIPS = 10;
  const foamTex = device.createTexture({
    label: 'foam detail', size: [FOAM_SIZE, FOAM_SIZE], format: 'rgba8unorm', mipLevelCount: FOAM_MIPS,
    usage: TU.TEXTURE_BINDING | TU.STORAGE_BINDING,
  });
  {
    const enc = device.createCommandEncoder();
    const cp = enc.beginComputePass();
    cp.setPipeline(foamGenPipe);
    cp.setBindGroup(0, bg(foamGenPipe, [lvl(foamTex, 0, '2d')]));
    cp.dispatchWorkgroups(FOAM_SIZE / 8, FOAM_SIZE / 8);
    cp.setPipeline(rgba8MipPipe);
    for (let m = 1; m < FOAM_MIPS; m++) {
      const s = FOAM_SIZE >> m;
      cp.setBindGroup(0, bg(rgba8MipPipe, [lvl(foamTex, m - 1, '2d'), lvl(foamTex, m, '2d')]));
      cp.dispatchWorkgroups(Math.ceil(s / 8), Math.ceil(s / 8));
    }
    cp.end();
    device.queue.submit([enc.finish()]);
  }

  // boat uniforms (see BOAT_WGSL): model, invModel, dims, wake, plan
  const boatUBO = mkBuf(176, GU.UNIFORM | GU.COPY_DST, 'boat ubo');
  const boatF = new Float32Array(44);

  // world-space foam clipmaps (ping-pong)
  const FOAM_LEVEL_SIZE = [128.0, 1024.0];
  const FOAMW_MIPS = Math.log2(FOAM_RES) + 1;
  const foamW = [0, 1].map((i) => device.createTexture({
    label: 'foam world ' + i, size: [FOAM_RES, FOAM_RES, 2], format: 'rgba16float', mipLevelCount: FOAMW_MIPS,
    usage: TU.TEXTURE_BINDING | TU.STORAGE_BINDING,
  }));
  const foamUBO = mkBuf(144, GU.UNIFORM | GU.COPY_DST, 'foam ubo');
  const foamUArr = new ArrayBuffer(144);
  const foamUI = new Int32Array(foamUArr);
  const foamUF = new Float32Array(foamUArr);
  const foamSnap = new Int32Array(4);
  let foamFirst = true;
  let foamCur = 0;
  const samplerSim = device.createSampler({
    magFilter: 'linear', minFilter: 'linear', mipmapFilter: 'linear', addressModeU: 'repeat', addressModeV: 'repeat',
  });
  const bgFoamUpd = [0, 1].map((i) => bg(foamUpdPipe, [{ buffer: foamUBO }, lvl(foamW[1 - i], 0), lvl(foamW[i], 0),
    fullArr(dispTex), fullArr(derivTex), samplerSim, samplerSim], 'foamupd' + i));
  const bgFoamMips = [0, 1].map((i) => {
    const out = [];
    for (let m = 1; m < FOAMW_MIPS; m++) out.push(bg(arr16MipPipe, [lvl(foamW[i], m - 1), lvl(foamW[i], m)]));
    return out;
  });

  // --------------------------------------------------------------------------------------
  // Atmosphere resources
  // --------------------------------------------------------------------------------------
  const atmoUBO = mkBuf(16, GU.UNIFORM | GU.COPY_DST, 'atmo ubo');
  const mkLut = (w, h, label) => device.createTexture({ label, size: [w, h], format: 'rgba16float', usage: TU.TEXTURE_BINDING | TU.STORAGE_BINDING });
  const transLUT = mkLut(256, 64, 'transmittance');
  const msLUT = mkLut(32, 32, 'multiscatter');
  const skyLUT = mkLut(SKY_W, SKY_H, 'skyview');
  const envTex = device.createTexture({
    label: 'env cube', size: [ENV_SIZE, ENV_SIZE, 6], format: 'rgba16float', mipLevelCount: ENV_MIPS,
    usage: TU.TEXTURE_BINDING | TU.STORAGE_BINDING,
  });
  const envCubeView = envTex.createView({ dimension: 'cube' });
  const sampClamp = device.createSampler({
    magFilter: 'linear', minFilter: 'linear', mipmapFilter: 'linear',
    addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge', addressModeW: 'clamp-to-edge',
  });
  const sampRepeat = device.createSampler({
    magFilter: 'linear', minFilter: 'linear', mipmapFilter: 'linear', addressModeU: 'repeat', addressModeV: 'repeat', maxAnisotropy: 16,
  });
  const frameUBO = mkBuf(384, GU.UNIFORM | GU.COPY_DST, 'frame ubo');
  const sunVisBuf = mkBuf(16, GU.STORAGE, 'sun visibility');
  const transView = transLUT.createView(), msView = msLUT.createView(), skyView = skyLUT.createView();
  const bgTrans = bg(transPipe, [{ buffer: atmoUBO }, transView], 'trans');
  const bgMS = bg(msPipe, [{ buffer: atmoUBO }, transView, sampClamp, msView], 'ms');
  const bgSkyView = bg(skyViewPipe, [{ buffer: frameUBO }, transView, msView, sampClamp, skyView], 'skyview');
  const bgEnv = bg(envPipe, [{ buffer: frameUBO }, transView, skyView, sampClamp, lvl(envTex, 0)], 'env');
  const bgSunVis = bg(sunVisPipe, [{ buffer: frameUBO }, transView, skyView, sampClamp, { buffer: sunVisBuf }], 'sunvis');
  const bgCubeMips = [];
  for (let m = 1; m < ENV_MIPS; m++) bgCubeMips.push(bg(cubeMipPipe, [lvl(envTex, m - 1), lvl(envTex, m)], 'cubemip' + m));

  // probe: water heights under the hull
  const probePts = new Float32Array(64);
  const probeUBO = mkBuf(256, GU.UNIFORM | GU.COPY_DST, 'probe pts');
  const probeOut = mkBuf(128, GU.STORAGE | GU.COPY_SRC, 'probe out');
  const probeRead = mkBuf(128, GU.MAP_READ | GU.COPY_DST, 'probe read');
  const bgProbe = bg(probePipe, [{ buffer: frameUBO }, fullArr(dispTex), samplerSim, { buffer: probeOut }, { buffer: probeUBO }], 'probe');

  // --------------------------------------------------------------------------------------
  // Render pipelines
  // --------------------------------------------------------------------------------------
  const MSAA = 4;
  const HDR = 'rgba16float';
  const [oceanMod, skyMod, downMod, upMod, compMod, boatMod, shadowMod] = await Promise.all([
    shader(FOAM_CONSTS + OCEAN, 'ocean'), shader(SKY_BG, 'sky'), shader(BLOOM_DOWN, 'bloom down'),
    shader(BLOOM_UP, 'bloom up'), shader(COMPOSITE, 'composite'), shader(BOAT_RENDER, 'boat'), shader(BOAT_SHADOW, 'boat shadow'),
  ]);
  const oceanPipe = await device.createRenderPipelineAsync({
    label: 'ocean', layout: 'auto',
    vertex: { module: oceanMod, entryPoint: 'vs', buffers: [{ arrayStride: 16, stepMode: 'instance', attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x4' }] }] },
    fragment: { module: oceanMod, entryPoint: 'fs', targets: [{ format: HDR }] },
    primitive: { topology: 'triangle-list', cullMode: 'none' },
    depthStencil: { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'greater' },
    multisample: { count: MSAA },
  });
  const skyPipe = await device.createRenderPipelineAsync({
    label: 'sky', layout: 'auto',
    vertex: { module: skyMod, entryPoint: 'vs' },
    fragment: { module: skyMod, entryPoint: 'fs', targets: [{ format: HDR }] },
    primitive: { topology: 'triangle-list' },
    depthStencil: { format: 'depth32float', depthWriteEnabled: false, depthCompare: 'equal' },
    multisample: { count: MSAA },
  });
  const vtxLayout = [{ arrayStride: 32, attributes: [
    { shaderLocation: 0, offset: 0, format: 'float32x3' }, { shaderLocation: 1, offset: 12, format: 'float32x3' },
    { shaderLocation: 2, offset: 24, format: 'float32x2' }] }];
  const boatPipe = await device.createRenderPipelineAsync({
    label: 'boat', layout: 'auto',
    vertex: { module: boatMod, entryPoint: 'vs', buffers: vtxLayout },
    fragment: { module: boatMod, entryPoint: 'fs', targets: [{ format: HDR }] },
    primitive: { topology: 'triangle-list', cullMode: 'none' },
    depthStencil: { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'greater' },
    multisample: { count: MSAA },
  });
  const SHADOW_RES = 2048;
  const shadowTex = device.createTexture({ size: [SHADOW_RES, SHADOW_RES], format: 'depth32float', usage: TU.RENDER_ATTACHMENT | TU.TEXTURE_BINDING });
  const shadowPipe = await device.createRenderPipelineAsync({
    label: 'boat shadow', layout: 'auto',
    vertex: { module: shadowMod, entryPoint: 'vs', buffers: [{ arrayStride: 32, attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3' }] }] },
    primitive: { topology: 'triangle-list', cullMode: 'none' },
    depthStencil: { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'less', depthBias: 2, depthBiasSlopeScale: 2 },
  });
  const shadowUBO = mkBuf(64, GU.UNIFORM | GU.COPY_DST, 'shadow ubo');
  const shadowSamp = device.createSampler({ compare: 'less', magFilter: 'linear', minFilter: 'linear' });
  const downPipe = await device.createRenderPipelineAsync({
    label: 'bloom down', layout: 'auto', vertex: { module: downMod, entryPoint: 'vs' },
    fragment: { module: downMod, entryPoint: 'fs', targets: [{ format: HDR }] },
  });
  const upPipe = await device.createRenderPipelineAsync({
    label: 'bloom up', layout: 'auto', vertex: { module: upMod, entryPoint: 'vs' },
    fragment: { module: upMod, entryPoint: 'fs', targets: [{ format: HDR, blend: { color: { srcFactor: 'one', dstFactor: 'one', operation: 'add' }, alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' } } }] },
  });
  const compPipe = await device.createRenderPipelineAsync({
    label: 'composite', layout: 'auto', vertex: { module: compMod, entryPoint: 'vs' },
    fragment: { module: compMod, entryPoint: 'fs', targets: [{ format: swapFormat }] },
  });

  const bgOcean = [0, 1].map((i) => bg(oceanPipe, [{ buffer: frameUBO }, transView, skyView, sampClamp,
    fullArr(dispTex), fullArr(derivTex), fullArr(momTex), foamTex.createView(), envCubeView, sampRepeat,
    fullArr(foamW[i]), { buffer: sunVisBuf }, { buffer: boatUBO }], 'ocean' + i));
  const bgSky = bg(skyPipe, [{ buffer: frameUBO }, transView, skyView, sampClamp], 'sky');
  const bgBoat = bg(boatPipe, [{ buffer: frameUBO }, transView, skyView, sampClamp, envCubeView, { buffer: sunVisBuf },
    { buffer: boatUBO }, { buffer: shadowUBO }, shadowTex.createView(), shadowSamp], 'boat');
  const bgShadow = bg(shadowPipe, [{ buffer: shadowUBO }], 'shadow');

  // --------------------------------------------------------------------------------------
  // Boat geometry on the GPU
  // --------------------------------------------------------------------------------------
  const geo = await loadBoatGeometry();
  const dims = geo.dims;
  const white = device.createTexture({ size: [1, 1], format: 'rgba8unorm-srgb', usage: TU.TEXTURE_BINDING | TU.COPY_DST });
  device.queue.writeTexture({ texture: white }, new Uint8Array([255, 255, 255, 255]), {}, [1, 1]);
  const texSamp = device.createSampler({ magFilter: 'linear', minFilter: 'linear', mipmapFilter: 'linear', addressModeU: 'repeat', addressModeV: 'repeat' });
  const texCache = new Map();
  async function gpuTexture(img) {
    if (texCache.has(img)) return texCache.get(img);
    const bmp = img instanceof ImageBitmap ? img : await createImageBitmap(img);
    const t = device.createTexture({ size: [bmp.width, bmp.height], format: 'rgba8unorm-srgb', usage: TU.TEXTURE_BINDING | TU.COPY_DST | TU.RENDER_ATTACHMENT });
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
    const vbuf = mkBuf(vb.byteLength, GU.VERTEX | GU.COPY_DST, 'boat vb');
    device.queue.writeBuffer(vbuf, 0, vb);
    const ibuf = mkBuf(Math.ceil(part.idx.byteLength / 4) * 4, GU.INDEX | GU.COPY_DST, 'boat ib');
    device.queue.writeBuffer(ibuf, 0, part.idx);
    const m = part.mat;
    const mu = new Float32Array([...m.base, m.metal, m.rough, m.coat || 0, m.mode || 0, ...(m.emissive || [0, 0, 0]), m.map ? 1 : 0]);
    const mbuf = mkBuf(48, GU.UNIFORM | GU.COPY_DST, 'boat mat');
    device.queue.writeBuffer(mbuf, 0, mu);
    const tex = m.map ? await gpuTexture(m.map) : white;
    draws.push({ vbuf, ibuf, count: part.idx.length, bg: bg(boatPipe, [{ buffer: mbuf }, tex.createView(), texSamp], 'mat', 1) });
  }
  const XB = dims.XB ?? dims.LB * 0.8;
  // bounding sphere of the boat (for the shadow camera)
  let bmin = [1e9, 1e9, 1e9], bmax = [-1e9, -1e9, -1e9];
  for (const part of geo.parts) for (let i = 0; i < part.pos.length; i += 3) for (let k = 0; k < 3; k++) { bmin[k] = Math.min(bmin[k], part.pos[i + k]); bmax[k] = Math.max(bmax[k], part.pos[i + k]); }
  const bCentre = bmin.map((v, k) => (v + bmax[k]) / 2), bRadius = Math.hypot(...bmax.map((v, k) => v - bmin[k])) / 2;

  // --------------------------------------------------------------------------------------
  // CDLOD grid mesh
  // --------------------------------------------------------------------------------------
  const GRID = 32, LEAF = 2.0, R0 = 8 * LEAF, LODS = 17, MAX_DIST = 160000, MAX_NODES = 8192;
  const idx = new Uint16Array(GRID * GRID * 6);
  {
    let k = 0;
    for (let j = 0; j < GRID; j++) for (let i = 0; i < GRID; i++) {
      const a = i + j * (GRID + 1), b = a + 1, c = a + GRID + 1, d = c + 1;
      idx[k++] = a; idx[k++] = c; idx[k++] = d; idx[k++] = a; idx[k++] = d; idx[k++] = b;
    }
  }
  const idxBuf = mkBuf(idx.byteLength, GU.INDEX | GU.COPY_DST, 'grid idx');
  device.queue.writeBuffer(idxBuf, 0, idx);
  const instData = new Float32Array(MAX_NODES * 4);
  const instBuf = mkBuf(instData.byteLength, GU.VERTEX | GU.COPY_DST, 'instances');

  // --------------------------------------------------------------------------------------
  // Screen targets
  // --------------------------------------------------------------------------------------
  const BLOOM_LEVELS = 8;
  const linSamp = device.createSampler({ magFilter: 'linear', minFilter: 'linear', addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge' });
  const compUBO = mkBuf(32, GU.UNIFORM | GU.COPY_DST, 'composite ubo');
  const compData = new Float32Array(8);
  const bloomUBOs = [];
  for (let i = 0; i < BLOOM_LEVELS * 2; i++) bloomUBOs.push(mkBuf(16, GU.UNIFORM | GU.COPY_DST, 'bloom ubo'));
  let W = 0, H = 0;
  let colorMS, depthMS, hdrTex, bloomTex;
  const mainPass = {
    colorAttachments: [{ view: null, resolveTarget: null, clearValue: { r: 0, g: 0, b: 0, a: 1 }, loadOp: 'clear', storeOp: 'discard' }],
    depthStencilAttachment: { view: null, depthClearValue: 0, depthLoadOp: 'clear', depthStoreOp: 'discard' },
  };
  const shadowPass = { colorAttachments: [], depthStencilAttachment: { view: shadowTex.createView(), depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'store' } };
  const compPass = { colorAttachments: [{ view: null, clearValue: { r: 0, g: 0, b: 0, a: 1 }, loadOp: 'clear', storeOp: 'store' }] };
  let downPasses = [], upPasses = [], downBGs = [], upBGs = [], bgComp = null;
  let offTex = null;

  function resize(w, h) {
    if (w === W && h === H) return;
    W = w; H = h;
    canvas.width = W; canvas.height = H;
    for (const t of [colorMS, depthMS, hdrTex, bloomTex]) t && t.destroy();
    colorMS = device.createTexture({ size: [W, H], format: HDR, sampleCount: MSAA, usage: TU.RENDER_ATTACHMENT });
    depthMS = device.createTexture({ size: [W, H], format: 'depth32float', sampleCount: MSAA, usage: TU.RENDER_ATTACHMENT });
    hdrTex = device.createTexture({ size: [W, H], format: HDR, usage: TU.RENDER_ATTACHMENT | TU.TEXTURE_BINDING });
    const bw = Math.max(1, W >> 1), bh = Math.max(1, H >> 1);
    bloomTex = device.createTexture({ size: [bw, bh], format: HDR, mipLevelCount: BLOOM_LEVELS, usage: TU.RENDER_ATTACHMENT | TU.TEXTURE_BINDING });
    mainPass.colorAttachments[0].view = colorMS.createView();
    mainPass.colorAttachments[0].resolveTarget = hdrTex.createView();
    mainPass.depthStencilAttachment.view = depthMS.createView();
    const hdrView = hdrTex.createView();
    const bv = [];
    for (let m = 0; m < BLOOM_LEVELS; m++) bv.push(bloomTex.createView({ baseMipLevel: m, mipLevelCount: 1 }));
    downPasses = []; upPasses = []; downBGs = []; upBGs = [];
    const tmp = new Float32Array(4);
    for (let m = 0; m < BLOOM_LEVELS; m++) {
      const srcW = m === 0 ? W : Math.max(1, bw >> (m - 1)), srcH = m === 0 ? H : Math.max(1, bh >> (m - 1));
      tmp[0] = 1 / srcW; tmp[1] = 1 / srcH; tmp[2] = m === 0 ? 1 : 0; tmp[3] = 0;
      device.queue.writeBuffer(bloomUBOs[m], 0, tmp);
      downBGs.push(bg(downPipe, [m === 0 ? hdrView : bv[m - 1], linSamp, { buffer: bloomUBOs[m] }]));
      downPasses.push({ colorAttachments: [{ view: bv[m], clearValue: { r: 0, g: 0, b: 0, a: 1 }, loadOp: 'clear', storeOp: 'store' }] });
    }
    for (let m = BLOOM_LEVELS - 1; m > 0; m--) {
      const sw = Math.max(1, bw >> m), sh = Math.max(1, bh >> m);
      tmp[0] = 1 / sw; tmp[1] = 1 / sh; tmp[2] = 1.0; tmp[3] = 1.0;
      const ubo = bloomUBOs[BLOOM_LEVELS + m];
      device.queue.writeBuffer(ubo, 0, tmp);
      upBGs.push(bg(upPipe, [bv[m], linSamp, { buffer: ubo }]));
      upPasses.push({ colorAttachments: [{ view: bv[m - 1], loadOp: 'load', storeOp: 'store' }] });
    }
    bgComp = bg(compPipe, [hdrView, bv[0], linSamp, { buffer: compUBO }]);
    offTex && offTex.destroy();
    offTex = device.createTexture({ size: [W, H], format: swapFormat, usage: TU.RENDER_ATTACHMENT | TU.COPY_SRC });
  }

  // --------------------------------------------------------------------------------------
  // Uniform data
  // --------------------------------------------------------------------------------------
  const simF = new Float32Array(56);
  const frameF = new Float32Array(96);
  const atmoF = new Float32Array(4);
  const proj = new Float32Array(16), view = new Float32Array(16), viewProj = new Float32Array(16);

  function writeSimStatic(time, dt) {
    for (let c = 0; c < C; c++) {
      const L = CASC_L[c];
      simF[c] = L;
      simF[8 + c] = c === 0 ? Math.PI / L : (2 * Math.PI * 6) / L;
      simF[16 + c] = c < C - 1 ? (2 * Math.PI * 6) / CASC_L[c + 1] : (Math.PI * N / L) * 0.98;
    }
    for (let c = 1; c < C; c++) simF[8 + c] = simF[16 + c - 1];
    simF[40] = P.windSpeed; simF[41] = P.windDir * D2R; simF[42] = P.fetchKm * 1000; simF[43] = 3.3;
    const ks = 2 * Math.PI / P.swellLength;
    simF[44] = P.swellHs; simF[45] = P.swellDir * D2R; simF[46] = Math.sqrt(9.81 * ks); simF[47] = P.swellSpread;
    simF[48] = P.choppy; simF[49] = 0.0012; simF[50] = LOOP; simF[51] = num('seed', 1);
    simF[52] = time; simF[53] = dt; simF[54] = 2.2; simF[55] = 1.0;
    device.queue.writeBuffer(simUBO, 0, simF);
  }

  // camera: position/target/up -> camera-relative reverse-Z infinite projection (as web-ocean)
  const cam = { x: 0, y: 0, z: 0, fovY: 50 * D2R, f: [1, 0, 0], r: [0, 0, 1], u: [0, 1, 0] };
  function setCamera(pos, tgt, up, fovDeg) {
    cam.x = pos[0]; cam.y = pos[1]; cam.z = pos[2]; cam.fovY = fovDeg * D2R;
    cam.f = norm3(sub3(tgt, pos));
    cam.r = norm3(cross3(cam.f, up));
    cam.u = cross3(cam.r, cam.f);
  }
  function writeFrame(time, frameIndex) {
    const aspect = W / H;
    const tanY = Math.tan(cam.fovY * 0.5), tanX = tanY * aspect;
    const near = 0.05;
    proj.fill(0);
    proj[0] = 1 / tanX; proj[5] = 1 / tanY; proj[11] = -1; proj[14] = near;
    const [fx, fy, fz] = cam.f, [rx, ry, rz] = cam.r, [ux, uy, uz] = cam.u;
    view.fill(0);
    view[0] = rx; view[4] = ry; view[8] = rz;
    view[1] = ux; view[5] = uy; view[9] = uz;
    view[2] = -fx; view[6] = -fy; view[10] = -fz;
    view[15] = 1;
    mat4Mul(viewProj, proj, view);
    const f = frameF;
    f.set(viewProj, 0);
    f[16] = cam.x; f[17] = cam.y; f[18] = cam.z; f[19] = time;
    f[20] = rx * tanX; f[21] = ry * tanX; f[22] = rz * tanX; f[23] = 0;
    f[24] = ux * tanY; f[25] = uy * tanY; f[26] = uz * tanY; f[27] = 0;
    f[28] = fx; f[29] = fy; f[30] = fz; f[31] = 0;
    const se = P.sunElev * D2R, sa = P.sunAzim * D2R;
    f[32] = Math.sin(sa) * Math.cos(se); f[33] = Math.sin(se); f[34] = -Math.cos(sa) * Math.cos(se); f[35] = 14.0;
    f[36] = W; f[37] = H; f[38] = 1 / W; f[39] = 1 / H;
    for (let c = 0; c < C; c++) { f[40 + c] = CASC_L[c]; f[48 + c] = 1 / CASC_L[c]; }
    f[56] = Math.cos(P.windDir * D2R); f[57] = Math.sin(P.windDir * D2R); f[58] = P.windSpeed;
    f[64] = R0; f[65] = GRID; f[66] = FOAM_LEVEL_SIZE[0]; f[67] = FOAM_LEVEL_SIZE[1];
    f[68] = P.exposure; f[69] = P.bloom; f[70] = frameIndex; f[71] = 0;
    // camera altitude for the sky LUTs: the camera's base height, not its drift, so every frame sees the same sky
    f[72] = P.haze; f[73] = Math.max(BOAT.cams[viewName].pos[1], 10) * 0.001; f[74] = P.veil; f[75] = 0.9;
    const wc = P.water || [0.016 * P.turbidity, 0.080 * P.turbidity, 0.096 * P.turbidity];
    f[76] = wc[0]; f[77] = wc[1]; f[78] = wc[2]; f[79] = 0.0024;
    const sc = P.sss || [0.035, 0.120, 0.085];
    f[80] = sc[0]; f[81] = sc[1]; f[82] = sc[2]; f[83] = 2.6;
    f[84] = 0.36 + 0.12 * P.foam; f[85] = 2.5; f[86] = 1.0; f[87] = 1.0;
    f[88] = (5.802e-3 + 4.40e-3 * P.haze) * 1e-3;
    f[89] = (13.558e-3 + 4.40e-3 * P.haze) * 1e-3;
    f[90] = (33.1e-3 + 4.40e-3 * P.haze) * 1e-3;
    f[91] = 1.1e-4;
    f[92] = 1.5; f[93] = P.clouds; f[94] = 2.2; f[95] = 0;     // clouds frozen: the loop must close
    device.queue.writeBuffer(frameUBO, 0, f);
  }

  // --------------------------------------------------------------------------------------
  // CDLOD node selection. Nodes are chosen around a camera position that stays fixed
  // relative to the boat, then shifted along with it, so the mesh layout never changes.
  // --------------------------------------------------------------------------------------
  const ranges = new Float64Array(LODS);
  for (let l = 0; l < LODS; l++) ranges[l] = R0 * Math.pow(2, l);
  const planes = new Float64Array(16);
  const HBND = 8.0, XBND = 6.0;
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
    instData[o] = x0; instData[o + 1] = z0; instData[o + 2] = size; instData[o + 3] = lod;
    nodeCount++;
  }
  function selectNode(x0, z0, size, lod) {
    if (!sphereHit(x0, z0, size, ranges[lod])) return false;
    if (!sphereHit(x0, z0, size, MAX_DIST) || !inFrustum(x0, z0, size)) return true;
    if (lod === 0 || !sphereHit(x0, z0, size, ranges[lod - 1])) { addNode(x0, z0, size, lod); return true; }
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
    ccx = cam.x - shiftX; ccy = cam.y; ccz = cam.z;
    const m = viewProj;
    for (let i = 0; i < 4; i++) {
      const r = i >> 1, s = (i & 1) ? -1 : 1;
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

  // --------------------------------------------------------------------------------------
  // Boat motion from the water under the hull (periodic, because the waves are)
  // --------------------------------------------------------------------------------------
  const HULL_PTS = [];
  for (let i = 0; i <= 5; i++) {
    const x = -dims.LS * 0.85 + (XB * 0.9 + dims.LS * 0.85) * i / 5;
    for (const z of [-dims.HB * 0.8, 0, dims.HB * 0.8]) HULL_PTS.push([x, z]);
  }
  const hullSpan = (XB * 0.9 + dims.LS * 0.85) * 0.8, hullBeam = dims.HB * 1.6;
  const cyc = hz => Math.max(1, Math.round(hz * LOOP)) * 2 * Math.PI / LOOP;
  function boatPose(heights, t) {
    let bow = 0, stern = 0, port = 0, stbd = 0, all = 0, nb = 0, ns = 0, np = 0, nst = 0;
    const mid = HULL_PTS[Math.floor(HULL_PTS.length / 2)][0];
    HULL_PTS.forEach(([x, z], i) => {
      const h = heights[i];
      all += h;
      if (x > mid + 0.1) { bow += h; nb++; } else if (x < mid - 0.1) { stern += h; ns++; }
      if (z < 0) { port += h; np++; } else if (z > 0) { stbd += h; nst++; }
    });
    const bz = BOAT.buzz;
    const pitch = Math.atan2(bow / nb - stern / ns, hullSpan) * 0.65 + BOAT.trim
      + bz * (0.0025 * Math.sin(cyc(9.0) * t) + 0.0015 * Math.sin(cyc(13.0) * t + 1.3));
    const roll = Math.atan2(port / np - stbd / nst, hullBeam) * 0.55 + BOAT.heel
      + 0.012 * Math.sin(cyc(0.11) * t + 0.4) + bz * 0.002 * Math.sin(cyc(11.0) * t);
    const yaw = 0.02 * Math.sin(cyc(1 / LOOP) * t + 1.0) + 0.008 * Math.sin(cyc(0.17) * t);
    const heave = all / HULL_PTS.length * 0.9 - 0.08 + bz * 0.008 * Math.sin(cyc(7.0) * t);
    return { pitch, roll, yaw, heave };
  }
  // column-major boat-local -> world
  function boatMatrix(x, pose) {
    const cy = Math.cos(pose.yaw), sy = Math.sin(pose.yaw), cp = Math.cos(pose.pitch), sp = Math.sin(pose.pitch), cr = Math.cos(pose.roll), sr = Math.sin(pose.roll);
    const Ry = [[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], Rz = [[cp, -sp, 0], [sp, cp, 0], [0, 0, 1]], Rx = [[1, 0, 0], [0, cr, -sr], [0, sr, cr]];
    const mul = (A, Bm) => A.map(r => Bm[0].map((_, j) => r[0] * Bm[0][j] + r[1] * Bm[1][j] + r[2] * Bm[2][j]));
    const R = mul(mul(Ry, Rz), Rx);
    const m = new Float32Array(16), inv = new Float32Array(16);
    for (let c = 0; c < 3; c++) for (let r = 0; r < 3; r++) { m[c * 4 + r] = R[r][c]; inv[c * 4 + r] = R[c][r]; }
    const t = [x, pose.heave, 0];
    m[12] = t[0]; m[13] = t[1]; m[14] = t[2]; m[15] = 1;
    for (let r = 0; r < 3; r++) inv[12 + r] = -(R[0][r] * t[0] + R[1][r] * t[1] + R[2][r] * t[2]);
    inv[15] = 1;
    return { m, inv, R };
  }

  // --------------------------------------------------------------------------------------
  // One simulation step (+ optionally a rendered frame)
  // --------------------------------------------------------------------------------------
  let started = false;
  let simTime = 0, lastFrameIndex = 0;

  function writeFoamUniforms(dt) {
    for (let l = 0; l < 2; l++) {
      const ts = FOAM_LEVEL_SIZE[l] / FOAM_RES;
      const cx = Math.floor(cam.x / ts), cz = Math.floor(cam.z / ts);
      foamUI[l * 4] = cx; foamUI[l * 4 + 1] = cz;
      foamUI[l * 4 + 2] = foamFirst ? cx + 100000 : foamSnap[l * 2];
      foamUI[l * 4 + 3] = foamFirst ? cz + 100000 : foamSnap[l * 2 + 1];
      foamSnap[l * 2] = cx; foamSnap[l * 2 + 1] = cz;
      foamUF[8 + l] = ts;
    }
    foamFirst = false;
    foamUF[10] = dt; foamUF[11] = 0.22;
    foamUF[12] = 0.52 + 0.16 * P.foam; foamUF[13] = 2.0; foamUF[14] = 1.0; foamUF[15] = 0;
    for (let c = 0; c < C; c++) { foamUF[16 + c] = CASC_L[c]; foamUF[24 + c] = 1 / CASC_L[c]; }
    const drift = 0.03 * P.windSpeed;
    foamUF[32] = Math.cos(P.windDir * D2R) * drift; foamUF[33] = Math.sin(P.windDir * D2R) * drift;
    device.queue.writeBuffer(foamUBO, 0, foamUArr);
  }

  function cameraFor(t, boatX, pose, B) {
    const c = BOAT.cams[viewName];
    const add = (a, b) => a.map((v, i) => v + b[i]);
    if (c.local) {
      const R = B.R, o = [boatX, pose.heave, 0];
      const toW = p => add([R[0][0] * p[0] + R[0][1] * p[1] + R[0][2] * p[2], R[1][0] * p[0] + R[1][1] * p[1] + R[1][2] * p[2], R[2][0] * p[0] + R[2][1] * p[1] + R[2][2] * p[2]], o);
      const up = [R[0][1], R[1][1], R[2][1]];
      return { pos: toW(c.pos), tgt: toW(c.tgt), up, fov: num('fov', c.fov) };
    }
    const w1 = 2 * Math.PI / LOOP, k = Math.max(dims.LB + dims.LS, 10) / 20;
    const pos = [c.pos[0] + 0.8 * k * Math.sin(w1 * t) + boatX, c.pos[1] + 0.35 * k * Math.sin(2 * w1 * t + 0.7) + pose.heave * 0.25, c.pos[2] + 1.2 * k * Math.sin(w1 * t + 2.1)];
    pos[1] = Math.max(pos[1], P.swellHs + 2.5);
    return { pos, tgt: [c.tgt[0] + boatX, c.tgt[1] + pose.heave * 0.4, c.tgt[2]], up: c.up || [0, 1, 0], fov: num('fov', c.fov) };
  }

  // frame index -> loop time and the boat's position. `index` may be negative (warm-up).
  async function step(index, render) {
    const t = ((index * DT) % LOOP + LOOP) % LOOP;
    const boatX = index * DT * V;
    simTime = t;
    // 1. waves at time t, and the water under the hull
    writeSimStatic(t, DT);
    for (let i = 0; i < HULL_PTS.length; i++) { probePts[i * 2] = boatX + HULL_PTS[i][0]; probePts[i * 2 + 1] = HULL_PTS[i][1]; }
    device.queue.writeBuffer(probeUBO, 0, probePts);
    {
      const enc = device.createCommandEncoder();
      const cp = enc.beginComputePass();
      if (!started) {
        cp.setPipeline(initPipe); cp.setBindGroup(0, bgInit); cp.dispatchWorkgroups(N / 16, N / 16, C);
        atmoF[0] = P.haze; device.queue.writeBuffer(atmoUBO, 0, atmoF);
        cp.setPipeline(transPipe); cp.setBindGroup(0, bgTrans); cp.dispatchWorkgroups(256 / 8, 64 / 8);
        cp.setPipeline(msPipe); cp.setBindGroup(0, bgMS); cp.dispatchWorkgroups(32 / 8, 32 / 8);
      }
      cp.setPipeline(evolvePipe); cp.setBindGroup(0, bgEvolve); cp.dispatchWorkgroups(N / 16, N / 16, C);
      cp.setPipeline(rowsPipe); cp.setBindGroup(0, bgRows); cp.dispatchWorkgroups(N, 1, C);
      cp.setPipeline(colsPipe); cp.setBindGroup(0, bgCols); cp.dispatchWorkgroups(N, 1, C);
      cp.setPipeline(simMipPipe);
      for (let m = 1; m < SIM_MIP_COUNT; m++) {
        const s = N >> m;
        cp.setBindGroup(0, bgSimMips[m - 1]); cp.dispatchWorkgroups(Math.ceil(s / 8), Math.ceil(s / 8), C);
      }
      cp.setPipeline(probePipe); cp.setBindGroup(0, bgProbe); cp.dispatchWorkgroups(1);
      cp.end();
      enc.copyBufferToBuffer(probeOut, 0, probeRead, 0, 128);
      device.queue.submit([enc.finish()]);
    }
    await probeRead.mapAsync(GPUMapMode.READ);
    const heights = Array.from(new Float32Array(probeRead.getMappedRange(0, 128)));
    probeRead.unmap();
    const pose = boatPose(heights, t);
    const Bm = boatMatrix(boatX, pose);

    // 2. camera, boat uniforms, foam (with the wake), sky
    const cm = cameraFor(t, boatX, pose, Bm);
    setCamera(cm.pos, cm.tgt, cm.up, cm.fov);
    writeFrame(t, index);
    boatF.set(Bm.m, 0); boatF.set(Bm.inv, 16);
    boatF.set([dims.LB, dims.LS, dims.HB, XB], 32);
    boatF.set([V, BOAT.wake, 1, BOAT.twin], 36);
    boatF.set([dims.bowPow || 1.5, DT, 0.07, 0], 40);
    device.queue.writeBuffer(boatUBO, 0, boatF);
    writeFoamUniforms(DT);
    selectAll(boatX);
    if (nodeCount) device.queue.writeBuffer(instBuf, 0, instData, 0, nodeCount * 4);

    // sun shadow camera around the boat, in boat-local space
    const se = P.sunElev * D2R, sa = P.sunAzim * D2R;
    const sunW = [Math.sin(sa) * Math.cos(se), Math.sin(se), -Math.cos(sa) * Math.cos(se)];
    const sunL = [0, 1, 2].map(k => Bm.R[0][k] * sunW[0] + Bm.R[1][k] * sunW[1] + Bm.R[2][k] * sunW[2]);
    const fwd = norm3(sunL.map(v => -v));
    const upH = Math.abs(fwd[1]) > 0.95 ? [1, 0, 0] : [0, 1, 0];
    const rt = norm3(cross3(fwd, upH)), up = cross3(rt, fwd);
    const r = bRadius * 1.05;
    const eye = bCentre.map((v, k) => v - fwd[k] * r * 2);
    const sh = new Float32Array(16);
    // orthographic: x,y in [-r, r] -> [-1, 1]; depth along fwd from eye: [0, 4r] -> [0, 1]
    for (let k = 0; k < 3; k++) { sh[k * 4] = rt[k] / r; sh[k * 4 + 1] = up[k] / r; sh[k * 4 + 2] = fwd[k] / (4 * r); }
    sh[12] = -dot3(rt, eye) / r; sh[13] = -dot3(up, eye) / r; sh[14] = -dot3(fwd, eye) / (4 * r); sh[15] = 1;
    device.queue.writeBuffer(shadowUBO, 0, sh);

    const enc = device.createCommandEncoder();
    const cp = enc.beginComputePass();
    foamCur = 1 - foamCur;
    cp.setPipeline(foamUpdPipe); cp.setBindGroup(0, bgFoamUpd[foamCur]); cp.dispatchWorkgroups(FOAM_RES / 8, FOAM_RES / 8, 2);
    cp.setPipeline(arr16MipPipe);
    for (let m = 1; m < FOAMW_MIPS; m++) {
      const s = Math.max(1, FOAM_RES >> m);
      cp.setBindGroup(0, bgFoamMips[foamCur][m - 1]); cp.dispatchWorkgroups(Math.ceil(s / 8), Math.ceil(s / 8), 2);
    }
    if (render) {
      cp.setPipeline(skyViewPipe); cp.setBindGroup(0, bgSkyView); cp.dispatchWorkgroups(Math.ceil(SKY_W / 8), Math.ceil(SKY_H / 8));
      cp.setPipeline(sunVisPipe); cp.setBindGroup(0, bgSunVis); cp.dispatchWorkgroups(1);
      if (!envDone) {
        // the sky never changes during a loop: build the environment cube once (all faces)
        cp.setPipeline(envPipe); cp.setBindGroup(0, bgEnv); cp.dispatchWorkgroups(ENV_SIZE / 8, ENV_SIZE / 8, 6);
        cp.setPipeline(cubeMipPipe);
        for (let m = 1; m < ENV_MIPS; m++) {
          const s = Math.max(1, ENV_SIZE >> m);
          cp.setBindGroup(0, bgCubeMips[m - 1]); cp.dispatchWorkgroups(Math.ceil(s / 8), Math.ceil(s / 8), 6);
        }
        envDone = true;
      }
    }
    cp.end();
    started = true;
    if (!render) { device.queue.submit([enc.finish()]); return null; }

    // 3. draw: boat shadow, sea + boat + sky, bloom, composite
    {
      const sp = enc.beginRenderPass(shadowPass);
      sp.setPipeline(shadowPipe); sp.setBindGroup(0, bgShadow);
      for (const d of draws) { sp.setVertexBuffer(0, d.vbuf); sp.setIndexBuffer(d.ibuf, 'uint32'); sp.drawIndexed(d.count); }
      sp.end();
    }
    const rp = enc.beginRenderPass(mainPass);
    if (nodeCount) {
      rp.setPipeline(oceanPipe); rp.setBindGroup(0, bgOcean[foamCur]);
      rp.setVertexBuffer(0, instBuf); rp.setIndexBuffer(idxBuf, 'uint16');
      rp.drawIndexed(GRID * GRID * 6, nodeCount);
    }
    rp.setPipeline(boatPipe); rp.setBindGroup(0, bgBoat);
    for (const d of draws) { rp.setBindGroup(1, d.bg); rp.setVertexBuffer(0, d.vbuf); rp.setIndexBuffer(d.ibuf, 'uint32'); rp.drawIndexed(d.count); }
    rp.setPipeline(skyPipe); rp.setBindGroup(0, bgSky); rp.draw(3);
    rp.end();
    for (let m = 0; m < BLOOM_LEVELS; m++) {
      const p = enc.beginRenderPass(downPasses[m]); p.setPipeline(downPipe); p.setBindGroup(0, downBGs[m]); p.draw(3); p.end();
    }
    for (let i = 0; i < upPasses.length; i++) {
      const p = enc.beginRenderPass(upPasses[i]); p.setPipeline(upPipe); p.setBindGroup(0, upBGs[i]); p.draw(3); p.end();
    }
    compData[0] = P.exposure; compData[1] = P.bloom; compData[2] = ((index % 64) + 64) % 64; compData[3] = 0.10;
    compData[4] = 1.05; compData[5] = 1.06; compData[6] = -0.19; compData[7] = 0;
    device.queue.writeBuffer(compUBO, 0, compData);
    // composite into our own texture: rendering into the canvas outside an animation frame
    // loses the device in headless Chrome. Live preview copies it to the canvas.
    const swapTex = offTex;
    compPass.colorAttachments[0].view = swapTex.createView();
    const pc = enc.beginRenderPass(compPass);
    pc.setPipeline(compPipe); pc.setBindGroup(0, bgComp); pc.draw(3); pc.end();
    lastFrameIndex = index;
    return { enc, swapTex };
  }
  let envDone = false;

  // read the finished frame back as raw RGBA
  async function capture(enc, swapTex) {
    const bpr = Math.ceil(W * 4 / 256) * 256;
    const buf = device.createBuffer({ size: bpr * H, usage: GU.MAP_READ | GU.COPY_DST });
    enc.copyTextureToBuffer({ texture: swapTex }, { buffer: buf, bytesPerRow: bpr }, [W, H]);
    device.queue.submit([enc.finish()]);
    await buf.mapAsync(GPUMapMode.READ);
    const src = new Uint8Array(buf.getMappedRange());
    const out = new Uint8Array(W * H * 4);
    const bgra = swapFormat === 'bgra8unorm';
    for (let y = 0; y < H; y++) {
      const s0 = y * bpr, d0 = y * W * 4;
      if (!bgra) { out.set(src.subarray(s0, s0 + W * 4), d0); continue; }
      for (let x = 0; x < W * 4; x += 4) { out[d0 + x] = src[s0 + x + 2]; out[d0 + x + 1] = src[s0 + x + 1]; out[d0 + x + 2] = src[s0 + x]; out[d0 + x + 3] = 255; }
    }
    buf.unmap(); buf.destroy();
    return out;
  }

  // --------------------------------------------------------------------------------------
  // Public API (used by tools/render-video.mjs) and live preview
  // --------------------------------------------------------------------------------------
  const outW = num('w', 0) || Math.round(innerWidth * (devicePixelRatio || 1));
  const outH = num('h', 0) || Math.round(innerHeight * (devicePixelRatio || 1));
  resize(outW, outH);
  let simIndex = null;            // index of the last simulated step
  async function advanceTo(index, warmupSeconds) {
    if (simIndex === null || index <= simIndex) {
      simIndex = index - Math.round(warmupSeconds * FPS) - 1;
      foamFirst = true;
    }
    while (simIndex < index - 1) { simIndex++; await step(simIndex, false); }
    simIndex = index;
    return step(index, true);
  }
  async function renderFrame(index) {
    const r = await advanceTo(index, WARMUP);
    const px = await capture(r.enc, r.swapTex);
    if (gpuError) throw new Error(gpuError);
    return px;
  }
  // raw RGBA -> base64 (chunked, for transfer to Node)
  function toBase64(u8) {
    let s = '';
    for (let i = 0; i < u8.length; i += 0x8000) s += String.fromCharCode.apply(null, u8.subarray(i, i + 0x8000));
    return btoa(s);
  }
  window.oceanPro = {
    info: { boat: boatName, scene: sceneName, view: viewName, loop: LOOP, fps: FPS, frames: FRAMES, width: W, height: H, speed: V, distance: DIST, cascades: CASC_L, swapFormat, model: !!(window.OCEAN_MODELS && window.OCEAN_MODELS[boatName]) },
    async frameBase64(index) { return toBase64(await renderFrame(index)); },
    renderFrame,
  };
  document.title = 'ready';

  if (qs.get('live') === '1') {
    // real-time preview: steps at the video frame rate, as fast as the GPU allows
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

main().catch((e) => { console.error('init failed: ' + (e && e.stack || e)); if (!window.oceanPro) window.oceanPro = { error: String(e && e.message || e) }; });
