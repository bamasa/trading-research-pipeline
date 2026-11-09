// Renders the post's animation to artifacts/posts/btc_ath_2024.mp4 (1080 x 1080, 30 fps)
// and writes the self-contained page next to it, artifacts/posts/btc_ath_2024.html.
//
//   node posts/btc_ath_2024/render.mjs
//
// Needs Chrome (set CHROME to its binary if it is not at the macOS default) and
// ffmpeg on the PATH. Every frame is drawn by animation.html's own renderAt(ms),
// so the video is the page frame for frame, with no screen recording.
import { spawn, spawnSync } from "node:child_process";
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const out = resolve(here, "../../artifacts/posts");
const chromeBin = process.env.CHROME || "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const fps = 30;

const post = {
  data: JSON.parse(readFileSync(join(here, "data.json"), "utf8")),
  numbers: JSON.parse(readFileSync(join(here, "numbers.json"), "utf8")),
  monitor: JSON.parse(readFileSync(join(here, "monitor.json"), "utf8")),
};
const page = readFileSync(join(here, "animation.html"), "utf8")
  .replace("<script>/*DATA*/</script>", `<script>window.POST=${JSON.stringify(post)};</script>`);
mkdirSync(out, { recursive: true });
writeFileSync(join(out, "btc_ath_2024.html"), page);

const work = mkdtempSync(join(tmpdir(), "post-"));
const frames = join(work, "frames");
mkdirSync(frames);
const local = join(work, "page.html");
writeFileSync(local, '<meta charset="utf-8">' + page);

const chrome = spawn(chromeBin, ["--headless=new", "--remote-debugging-port=9333", `--user-data-dir=${join(work, "profile")}`,
  "--hide-scrollbars", "--window-size=1080,1080", "--force-device-scale-factor=1", "about:blank"], { stdio: "ignore" });
const sleep = ms => new Promise(r => setTimeout(r, ms));
let targets = [];
for (let i = 0; i < 50 && !targets.length; i++) {
  try { targets = (await (await fetch("http://127.0.0.1:9333/json")).json()).filter(t => t.type === "page"); } catch { await sleep(200); }
}
const ws = new WebSocket(targets[0].webSocketDebuggerUrl);
await new Promise(r => ws.addEventListener("open", r));
let id = 0;
const pending = new Map();
ws.addEventListener("message", e => { const m = JSON.parse(e.data); pending.get(m.id)?.(m); pending.delete(m.id); });
const send = (method, params = {}) => new Promise(r => { pending.set(++id, r); ws.send(JSON.stringify({ id, method, params })); });
const ev = async expression => (await send("Runtime.evaluate", { expression, returnByValue: true, awaitPromise: true })).result?.result?.value;

await send("Emulation.setDeviceMetricsOverride", { width: 1080, height: 1080, deviceScaleFactor: 1, mobile: false });
await send("Page.navigate", { url: pathToFileURL(local).href + "?render=1" });
for (let i = 0; i < 100 && !(await ev("window.READY === true")); i++) await sleep(200);
const { PLAY, HOLD } = await ev("window.TIMING");
const n = Math.round((PLAY + HOLD) / 1000 * fps);
for (let f = 0; f < n; f++) {
  const png = await ev(`renderAt(${f * 1000 / fps}); document.getElementById('c').toDataURL('image/png')`);
  writeFileSync(join(frames, `f_${String(f).padStart(5, "0")}.png`), Buffer.from(png.split(",")[1], "base64"));
}
ws.close();
chrome.kill();

const video = join(out, "btc_ath_2024.mp4");
const ff = spawnSync("ffmpeg", ["-y", "-loglevel", "error", "-framerate", String(fps), "-i", join(frames, "f_%05d.png"),
  "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "slow", "-movflags", "+faststart", video], { stdio: "inherit" });
rmSync(work, { recursive: true, force: true });
if (ff.status !== 0) process.exit(ff.status ?? 1);
console.log(`${n} frames -> ${video}`);
