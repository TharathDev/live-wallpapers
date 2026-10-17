// Renders the wallpaper offline, frame by frame, into an H.264 .mp4 that loops seamlessly.
// Used by .github/workflows/render-video.yml, which splits the frames across many machines;
// it also runs locally (needs Node, `npm install` in tools/, and ffmpeg on the PATH).
//
//   node tools/render-video.mjs --renderer realistic --query "boat=yacht&scene=tropical" \
//        --width 2560 --height 1600 --fps 30 [--start 0 --end 900] --out video.mp4
//
// Renderers:
//   realistic  pro/index.html: WebGPU FFT ocean, physically based sky, 3D boat (default)
//   basic      index.html: the lighter WebGL wallpaper
//
// Frame i shows time i * loop / frames, where frames = loop * fps. Frame number `frames` would
// be identical to frame 0, so it's left out and the video repeats without a jump.
import { chromium } from 'playwright';
import { spawn } from 'node:child_process';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const args = Object.fromEntries(process.argv.slice(2).reduce((a, v, i, all) => (v.startsWith('--') ? [...a, [v.slice(2), all[i + 1]]] : a), []));
const width = +(args.width || 1920), height = +(args.height || 1080), fps = +(args.fps || 30);
const query = args.query || '';
const renderer = args.renderer || 'realistic';
const out = resolve(args.out || 'video.mp4');
const root = join(dirname(fileURLToPath(import.meta.url)), '..');

const launchArgs = renderer === 'realistic'
  ? ['--enable-unsafe-webgpu', '--enable-features=Vulkan', '--use-vulkan=swiftshader', '--use-webgpu-adapter=swiftshader', '--ignore-gpu-blocklist']
  : ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'];
const browser = await chromium.launch({ args: launchArgs });
const page = await browser.newPage({ viewport: { width: 640, height: 360 } });
page.on('pageerror', e => { console.error('page error:', e); process.exitCode = 1; });
page.on('console', m => { if (m.type() === 'error') console.error('console:', m.text().slice(0, 500)); });

let frames, loop, grab;
if (renderer === 'realistic') {
  await page.goto(`${pathToFileURL(join(root, 'pro', 'index.html')).href}?${query}&w=${width}&h=${height}&fps=${fps}`);
  await page.waitForFunction(() => window.oceanPro, null, { timeout: 300000 });
  const info = await page.evaluate(() => window.oceanPro.error || window.oceanPro.info);
  if (typeof info === 'string') throw new Error(info);
  console.log(JSON.stringify(info));
  frames = info.frames; loop = info.loop;
  grab = async i => {
    // started in the page and polled: the frame can take minutes in software
    await page.evaluate(i => {
      window.__res = null; window.__err = null;
      window.oceanPro.frameBase64(i).then(r => { window.__res = r; }, e => { window.__err = String(e && e.stack || e); });
    }, i);
    await page.waitForFunction(() => window.__res || window.__err, null, { timeout: 0, polling: 250 });
    const err = await page.evaluate(() => window.__err);
    if (err) throw new Error(err);
    return Buffer.from(await page.evaluate(() => window.__res), 'base64');
  };
} else {
  await page.goto(`${pathToFileURL(join(root, 'index.html')).href}?still=1&${query}`);
  await page.waitForFunction(() => document.title === 'ready', null, { timeout: 300000 });
  loop = await page.evaluate(() => window.__ocean.LOOP);
  frames = Math.round(loop * fps);
  grab = async i => {
    const png = await page.evaluate(([i, n, w, h]) => {
      const o = window.__ocean;
      o.renderAt(i * o.LOOP / n, w, h);
      return document.getElementById('c').toDataURL('image/png');
    }, [i, frames, width, height]);
    return Buffer.from(png.slice(png.indexOf(',') + 1), 'base64');
  };
}
const start = +(args.start || 0), end = Math.min(frames, +(args.end || frames));
console.log(`${renderer} ${query || '(defaults)'}: frames ${start}-${end - 1} of ${frames} (${loop} s at ${fps} fps, ${width}x${height})`);

const input = renderer === 'realistic'
  ? ['-f', 'rawvideo', '-pix_fmt', 'rgba', '-s', `${width}x${height}`, '-framerate', String(fps), '-i', '-']
  : ['-f', 'image2pipe', '-framerate', String(fps), '-i', '-'];
const ff = spawn('ffmpeg', [
  '-y', '-loglevel', 'error', ...input,
  '-vf', 'scale=out_color_matrix=bt709:out_range=tv', '-c:v', 'libx264', '-preset', 'slow', '-crf', '17',
  '-bf', '0', '-g', String(fps * 2), '-pix_fmt', 'yuv420p',
  '-color_primaries', 'bt709', '-color_trc', 'bt709', '-colorspace', 'bt709',
  '-movflags', '+faststart', '-tag:v', 'avc1', out,
], { stdio: ['pipe', 'inherit', 'inherit'] });
const ffDone = new Promise((res, rej) => ff.on('close', code => (code ? rej(new Error(`ffmpeg exited with ${code}`)) : res())));

const t0 = Date.now();
for (let i = start; i < end; i++) {
  const buf = await grab(i);
  if (!ff.stdin.write(buf)) await new Promise(r => ff.stdin.once('drain', r));
  const done = i - start + 1;
  if (done % 5 === 0 || i === end - 1 || done === 1) {
    const per = (Date.now() - t0) / 1000 / done;
    console.log(`frame ${done}/${end - start}  ${per.toFixed(1)} s/frame  ~${Math.round(per * (end - i - 1) / 60)} min left`);
  }
}
ff.stdin.end();
await ffDone;
await browser.close();
console.log(`wrote ${out}`);
