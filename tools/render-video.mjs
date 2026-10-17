// Renders the wallpaper offline, frame by frame, into an H.264 .mp4 that loops seamlessly.
// Used by .github/workflows/render-video.yml, which splits the frames across many machines;
// it also runs locally (needs Node, `npm install` in tools/, and ffmpeg on the PATH).
//
//   node tools/render-video.mjs --query "boat=yacht&scene=sunset" --width 1920 --height 1080 \
//        --fps 30 [--start 0 --end 900] --out video.mp4
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
const out = resolve(args.out || 'video.mp4');
const page_ = pathToFileURL(join(dirname(fileURLToPath(import.meta.url)), '..', 'index.html')).href;

const browser = await chromium.launch({ args: ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'] });
const page = await browser.newPage({ viewport: { width: 640, height: 360 } });
page.on('pageerror', e => { console.error('page error:', e); process.exitCode = 1; });
await page.goto(`${page_}?still=1&${query}`);
await page.waitForFunction(() => document.title === 'ready', null, { timeout: 300000 });
const loop = await page.evaluate(() => window.__ocean.LOOP);
const frames = Math.round(loop * fps);
const start = +(args.start || 0), end = Math.min(frames, +(args.end || frames));
console.log(`${query || '(defaults)'}: frames ${start}-${end - 1} of ${frames} (${loop} s at ${fps} fps, ${width}x${height})`);

const ff = spawn('ffmpeg', [
  '-y', '-loglevel', 'error', '-f', 'image2pipe', '-framerate', String(fps), '-i', '-',
  '-vf', 'scale=out_color_matrix=bt709:out_range=tv', '-c:v', 'libx264', '-preset', 'slow', '-crf', '17',
  '-bf', '0', '-g', String(fps * 2), '-pix_fmt', 'yuv420p',
  '-color_primaries', 'bt709', '-color_trc', 'bt709', '-colorspace', 'bt709',
  '-movflags', '+faststart', '-tag:v', 'avc1', out,
], { stdio: ['pipe', 'inherit', 'inherit'] });
const ffDone = new Promise((res, rej) => ff.on('close', code => (code ? rej(new Error(`ffmpeg exited with ${code}`)) : res())));

const t0 = Date.now();
for (let i = start; i < end; i++) {
  const png = await page.evaluate(([i, n, w, h]) => {
    const o = window.__ocean;
    o.renderAt(i * o.LOOP / n, w, h);
    return document.getElementById('c').toDataURL('image/png');
  }, [i, frames, width, height]);
  if (!ff.stdin.write(Buffer.from(png.slice(png.indexOf(',') + 1), 'base64'))) await new Promise(r => ff.stdin.once('drain', r));
  const done = i - start + 1;
  if (done % 10 === 0 || i === end - 1) {
    const per = (Date.now() - t0) / 1000 / done;
    console.log(`frame ${i + 1 - start}/${end - start}  ${per.toFixed(1)} s/frame  ~${Math.round(per * (end - i - 1) / 60)} min left`);
  }
}
ff.stdin.end();
await ffDone;
await browser.close();
console.log(`wrote ${out}`);
