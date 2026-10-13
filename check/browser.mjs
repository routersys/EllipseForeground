import { existsSync, mkdirSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { resolve } from "node:path";
import puppeteer from "puppeteer-core";
import { createSiteServer } from "./serve.mjs";

const OUTPUT = resolve(fileURLToPath(new URL("./out/", import.meta.url)));
const CANDIDATES = [
  process.env.CHROME_PATH,
  "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
  "C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe",
  "/usr/bin/google-chrome",
  "/usr/bin/chromium",
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
].filter(Boolean);
const executablePath = CANDIDATES.find((path) => existsSync(path));
if (!executablePath) {
  console.error("Chrome が見つかりません。環境変数 CHROME_PATH に実行ファイルを指定してください。");
  process.exit(2);
}

mkdirSync(OUTPUT, { recursive: true });
const server = createSiteServer();
await new Promise((done) => server.listen(0, "127.0.0.1", done));
const base = `http://127.0.0.1:${server.address().port}/EllipseForeground/`;
const browser = await puppeteer.launch({ executablePath, headless: true });

const results = [];
const check = (condition, message) => {
  results.push({ ok: Boolean(condition), message });
  console.log(`${condition ? "ok" : "NG"} ${message}`);
};

async function open(url, viewport = { width: 1200, height: 900 }) {
  const page = await browser.newPage();
  await page.setViewport(viewport);
  const errors = [];
  page.on("pageerror", (error) => errors.push(`pageerror ${error.message}`));
  page.on("console", (message) => {
    if (message.type() === "error") errors.push(`console ${message.text()}`);
  });
  page.on("requestfailed", (request) => {
    if (request.failure()?.errorText !== "net::ERR_ABORTED") errors.push(`requestfailed ${request.url()}`);
  });
  const response = await page.goto(url, { waitUntil: "load" });
  return { page, errors, response };
}

async function wait(page, predicate, argument, timeout = 40000) {
  await page.waitForFunction(predicate, { timeout }, argument);
}

const rendered = () => document.querySelectorAll("#viewer canvas").length > 0 && document.querySelectorAll("#viewer .textLayer span").length > 0;

for (const [language, path, searchTerm] of [
  ["en", "paper/", "階層木"],
  ["ja", "ja/paper/", "階層木"],
]) {
  const { page, errors } = await open(base + path);
  await wait(page, rendered);
  const initial = await page.evaluate(() => ({
    status: document.getElementById("status").textContent,
    pages: Number(document.getElementById("pages").textContent),
    elements: document.querySelectorAll(".page").length,
    canvases: document.querySelectorAll("#viewer canvas").length,
    disabled: document.querySelectorAll("#toolbar [disabled]").length,
    width: document.querySelector(".page").getBoundingClientRect().width,
    overflow: document.documentElement.scrollWidth - window.innerWidth,
  }));
  check(initial.pages > 1 && initial.pages === initial.elements, `${language}: ページ数 ${initial.pages} とページの要素数が一致`);
  check(initial.canvases >= 1 && initial.canvases <= 6, `${language}: 見えている付近だけを描画 (canvas ${initial.canvases} 枚)`);
  check(initial.disabled === 0, `${language}: 操作が有効になる`);
  check(initial.overflow <= 0, `${language}: 横にはみ出さない`);

  const ink = await page.evaluate(() => {
    const canvas = document.querySelector("#viewer canvas");
    const data = canvas.getContext("2d").getImageData(0, 0, canvas.width, canvas.height).data;
    let dark = 0;
    for (let i = 0; i < data.length; i += 4) if (data[i] < 128 && data[i + 3] > 0) dark += 1;
    return dark;
  });
  check(ink > 2000, `${language}: 1 ページ目の canvas に文字が描かれる (暗い画素 ${ink})`);

  const text = await page.evaluate(() => [...document.querySelectorAll("#viewer .textLayer")].map((layer) => layer.textContent).join(""));
  check(text.includes("楕円"), `${language}: テキスト層から本文を読める`);
  check(!/[\u2E80-\u2FDF]/.test(text), `${language}: テキスト層に康熙部首が残らない`);
  await page.screenshot({ path: resolve(OUTPUT, `viewer-${language}-top.png`) });

  await page.click("#next");
  await wait(page, () => document.getElementById("page").value === "2" && location.hash === "#page=2");
  await wait(page, () => { const top = document.getElementById("page-2").getBoundingClientRect().top; return top > -5 && top < 120; });
  check(true, `${language}: 次のページへ移り、URL の #page=2 が変わる`);

  const before = await page.evaluate(() => document.querySelector(".page").getBoundingClientRect().width);
  await page.click("#zoomIn");
  await wait(page, (previous) => document.querySelector(".page").getBoundingClientRect().width > previous + 1, before);
  const zoomed = await page.evaluate(() => ({ width: document.querySelector(".page").getBoundingClientRect().width, value: document.getElementById("zoom").value }));
  check(zoomed.width > before && zoomed.value !== "fit", `${language}: 拡大で幅が ${Math.round(before)} から ${Math.round(zoomed.width)} になる (倍率 ${zoomed.value})`);
  await page.select("#zoom", "fit");
  await wait(page, (previous) => Math.abs(document.querySelector(".page").getBoundingClientRect().width - previous) < 2, before);
  check(true, `${language}: 幅に合わせるへ戻せる`);

  await page.type("#query", searchTerm);
  await page.keyboard.press("Enter");
  await wait(page, () => document.querySelectorAll("#hits li").length > 0);
  const hit = await page.evaluate(() => ({
    count: document.querySelectorAll("#hits li").length,
    summary: document.getElementById("summary").textContent,
    first: document.querySelector("#hits button").dataset.page,
    snippet: document.querySelector("#hits li").textContent,
  }));
  check(hit.count > 0 && hit.snippet.includes(searchTerm), `${language}: 検索で ${hit.count} 件、先頭は ${hit.first} ページ (${hit.summary})`);
  await page.click("#hits button");
  await wait(page, (number) => document.getElementById("page").value === number, hit.first);
  await wait(page, (number) => document.querySelector(`#page-${number} canvas`) !== null, hit.first);
  check(true, `${language}: 検索結果から ${hit.first} ページへ移動して描画される`);
  await page.screenshot({ path: resolve(OUTPUT, `viewer-${language}-search.png`) });

  await page.type("#query", "存在しない語zzqq");
  await page.keyboard.press("Enter");
  await wait(page, () => document.querySelectorAll("#hits li").length === 0 && document.getElementById("summary").textContent !== "");
  check(true, `${language}: 一致なしのとき一覧が空になり、メッセージが出る`);

  check(errors.length === 0, `${language}: コンソールの誤りと読み込みの失敗がない ${errors.join(" | ")}`);
  await page.close();
}

{
  const { page, errors } = await open(`${base}paper/#page=12`);
  await wait(page, () => document.getElementById("page").value === "12" && document.querySelector("#page-12 canvas") !== null);
  check(true, "en: #page=12 で開くと 12 ページ目が描画される");
  check(errors.length === 0, `en: #page=12 でも誤りがない ${errors.join(" | ")}`);
  await page.close();
}

{
  const { page, errors } = await open(`${base}paper/`, { width: 390, height: 800 });
  await wait(page, rendered);
  const mobile = await page.evaluate(() => ({
    overflow: document.documentElement.scrollWidth - window.innerWidth,
    pageWidth: document.querySelector(".page").getBoundingClientRect().width,
  }));
  check(mobile.overflow <= 0 && mobile.pageWidth <= 390, `幅 390 でも横にはみ出さない (ページの幅 ${Math.round(mobile.pageWidth)})`);
  await page.screenshot({ path: resolve(OUTPUT, "viewer-mobile.png") });
  check(errors.length === 0, `幅 390 でも誤りがない ${errors.join(" | ")}`);
  await page.close();
}

for (const [label, path, status] of [
  ["en の入口", "", 200],
  ["ja の入口", "ja/", 200],
  ["存在しないページ", "nope", 404],
]) {
  const { page, errors, response } = await open(base + path);
  check(response.status() === status, `${label}: HTTP ${response.status()}`);
  if (status === 200) {
    const info = await page.evaluate(() => ({ overflow: document.documentElement.scrollWidth - window.innerWidth, h1: document.querySelector("h1").textContent }));
    check(info.overflow <= 0 && info.h1 === "EllipseForeground", `${label}: 見出しがあり、横にはみ出さない`);
    await page.screenshot({ path: resolve(OUTPUT, `home-${path === "" ? "en" : "ja"}.png`) });
  }
  check(errors.length === 0 || status === 404, `${label}: 誤りがない ${errors.join(" | ")}`);
  await page.close();
}

await browser.close();
server.close();
const failed = results.filter((result) => !result.ok);
console.log(failed.length === 0 ? `ok: ${results.length} 項目` : `NG: ${failed.length} / ${results.length} 項目`);
process.exit(failed.length === 0 ? 0 : 1);
