import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { existsSync, readFileSync, readdirSync, statSync } from "node:fs";
import { dirname, join, relative, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";

const SITE = "https://lib.routersys.com/EllipseForeground/";
const ROOT = resolve(fileURLToPath(new URL("../docs/", import.meta.url)));
const REPOSITORY = resolve(ROOT, "..");
const PDF = "paper/ellipse-foreground-extraction.pdf";

const problems = [];
const fail = (message) => problems.push(message);
const read = (path) => readFileSync(path, "utf-8");

function walk(directory) {
  return readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
    const path = join(directory, entry.name);
    return entry.isDirectory() ? walk(path) : [path];
  });
}

const files = walk(ROOT);
const pages = files.filter((file) => file.endsWith(".html") && !file.includes(`${sep}vendor${sep}`));
const urlOf = (file) => {
  const path = relative(ROOT, file).split(sep).join("/").replace(/index\.html$/, "");
  return SITE + path;
};

function attribute(tag, name) {
  const match = new RegExp(`\\s${name}=("([^"]*)"|'([^']*)')`, "i").exec(tag);
  return match ? (match[2] ?? match[3]) : null;
}

function tags(html, name) {
  return [...html.matchAll(new RegExp(`<${name}\\b[^>]*>`, "gi"))].map((match) => match[0]);
}

function localTarget(file, reference) {
  if (/^(https?:|mailto:|data:|#)/i.test(reference) || reference === "") {
    return null;
  }

  const path = reference.split("#")[0].split("?")[0];
  const target = resolve(dirname(file), decodeURIComponent(path));
  return existsSync(target) && statSync(target).isDirectory() ? join(target, "index.html") : target;
}

const canonicals = new Map();
for (const file of pages) {
  const html = read(file);
  const name = relative(ROOT, file).split(sep).join("/");
  const notFound = name === "404.html";

  if (!/^<!doctype html>/i.test(html)) fail(`${name}: doctype がありません`);
  if (!/<html lang="(en|ja)">/.test(html)) fail(`${name}: html の lang がありません`);
  if (!/<meta charset="utf-8">/i.test(html)) fail(`${name}: charset がありません`);
  if (!/<meta name="viewport"/.test(html)) fail(`${name}: viewport がありません`);
  if (!/<meta name="color-scheme" content="light dark">/.test(html)) fail(`${name}: color-scheme がありません`);
  if (!/<title>[^<]+<\/title>/.test(html)) fail(`${name}: title がありません`);

  if (notFound) {
    if (!/<meta name="robots" content="noindex">/.test(html)) fail(`${name}: noindex がありません`);
  } else {
    const links = tags(html, "link");
    const canonical = links.map((tag) => (attribute(tag, "rel") === "canonical" ? attribute(tag, "href") : null)).find(Boolean);
    if (canonical !== urlOf(file)) fail(`${name}: canonical が ${canonical} です。${urlOf(file)} にしてください`);
    canonicals.set(name, canonical);

    const alternates = Object.fromEntries(
      links.filter((tag) => attribute(tag, "rel") === "alternate").map((tag) => [attribute(tag, "hreflang"), attribute(tag, "href")]),
    );
    for (const language of ["en", "ja", "x-default"]) {
      if (!alternates[language]) fail(`${name}: hreflang ${language} がありません`);
    }
    if (alternates[html.match(/<html lang="(en|ja)">/)?.[1]] !== canonical) fail(`${name}: 自分の言語の hreflang が canonical と違います`);

    const description = /<meta name="description" content="([^"]+)"/.exec(html)?.[1];
    if (!description) fail(`${name}: description がありません`);
    for (const property of ["og:title", "og:description", "og:url", "og:site_name", "og:type", "og:locale"]) {
      if (!new RegExp(`<meta property="${property}" content="[^"]+"`).test(html)) fail(`${name}: ${property} がありません`);
    }
    if (new RegExp(`<meta property="og:url" content="([^"]+)"`).exec(html)?.[1] !== canonical) fail(`${name}: og:url が canonical と違います`);
    if (new RegExp(`<meta property="og:description" content="([^"]+)"`).exec(html)?.[1] !== description) fail(`${name}: og:description が description と違います`);

    const ld = /<script type="application\/ld\+json">([\s\S]*?)<\/script>/.exec(html)?.[1];
    if (!ld) {
      fail(`${name}: JSON-LD がありません`);
    } else {
      try {
        const data = JSON.parse(ld);
        if (data.url !== canonical) fail(`${name}: JSON-LD の url が canonical と違います`);
      } catch (error) {
        fail(`${name}: JSON-LD を読めません ${error.message}`);
      }
    }
  }

  const references = [...html.matchAll(/\s(?:href|src|data-src)="([^"]*)"/g)].map((match) => match[1]);
  for (const reference of references) {
    const target = localTarget(file, reference);
    if (target !== null && !existsSync(target)) fail(`${name}: ${reference} の行き先がありません`);
    if (target !== null && !target.startsWith(ROOT)) fail(`${name}: ${reference} がサイトの外を指しています`);
  }
  for (const reference of references) {
    if (/^https:\/\/lib\.routersys\.com\/EllipseForeground\//.test(reference)) {
      const path = reference.slice(SITE.length).split("#")[0];
      const target = join(ROOT, path === "" || path.endsWith("/") ? `${path}index.html` : path);
      if (!existsSync(target)) fail(`${name}: ${reference} に対応するファイルがありません`);
    }
  }

  if (/<html lang="ja">/.test(html)) {
    const text = html
      .replace(/<(script|style|pre|code)\b[\s\S]*?<\/\1>/gi, "")
      .replace(/<div id="toolbar"[\s\S]*?<\/div>/i, "")
      .replace(/<[^>]+>/g, "");
    const spaced = text.match(/[぀-ヿ一-鿿][ ][A-Za-z0-9]|[A-Za-z0-9][ ][぀-ヿ一-鿿]/g);
    if (spaced) fail(`${name}: 和文と欧文の間に半角空白があります ${[...new Set(spaced)].join(" / ")}`);
  }
}

const viewerPages = pages.filter((file) => file.endsWith(`paper${sep}index.html`));
assert.equal(viewerPages.length, 2, "PDF ビューアのページは 2 つのはずです");
const script = read(join(ROOT, "paper", "viewer.js"));
const ids = [...script.matchAll(/getElementById\("([^"]+)"\)/g)].map((match) => match[1]);
for (const file of viewerPages) {
  const html = read(file);
  const name = relative(ROOT, file).split(sep).join("/");
  for (const id of new Set(ids)) {
    if (!new RegExp(`\\sid="${id}"`).test(html)) fail(`${name}: viewer.js が使う id="${id}" がありません`);
  }
  const messages = JSON.parse(/<script type="application\/json" id="messages">([\s\S]*?)<\/script>/.exec(html)[1]);
  for (const key of [...script.matchAll(/messages\.(\w+)/g)].map((match) => match[1])) {
    if (!(key in messages)) fail(`${name}: messages に ${key} がありません`);
  }
}

const sitemap = read(join(ROOT, "sitemap.xml"));
const locations = [...sitemap.matchAll(/<loc>([^<]+)<\/loc>/g)].map((match) => match[1]);
const expected = [...canonicals.values()].sort();
assert.deepEqual([...locations].sort(), expected, "sitemap.xml の loc がページの canonical と一致しません");
for (const block of sitemap.split("<url>").slice(1)) {
  const loc = /<loc>([^<]+)<\/loc>/.exec(block)[1];
  const file = [...canonicals.entries()].find(([, value]) => value === loc)[0];
  const html = read(join(ROOT, file));
  for (const match of block.matchAll(/hreflang="([^"]+)" href="([^"]+)"/g)) {
    const own = new RegExp(`<link rel="alternate" hreflang="${match[1]}" href="${match[2]}">`).test(html);
    if (!own) fail(`sitemap.xml: ${loc} の ${match[1]} がページの hreflang と違います`);
  }
}

const pdfPath = join(ROOT, PDF);
const pdf = readFileSync(pdfPath);
if (pdf.subarray(0, 5).toString("latin1") !== "%PDF-") fail("PDF の先頭が %PDF- ではありません");
const pdfHash = createHash("sha256").update(pdf).digest("hex");
let published = null;
for (const reference of ["main", "origin/main"]) {
  try {
    published = execFileSync("git", ["-C", REPOSITORY, "show", `${reference}:docs/ellipse-foreground-extraction.pdf`], { maxBuffer: 1 << 28, stdio: ["ignore", "pipe", "ignore"] });
    break;
  } catch {
    published = null;
  }
}
if (published === null) {
  console.log("注意: main ブランチの PDF が見つからないため、PDF の一致は確かめませんでした");
} else if (createHash("sha256").update(published).digest("hex") !== pdfHash) {
  fail("docs/paper の PDF が main ブランチの docs の PDF と違います。更新してください");
}

const vendor = join(ROOT, "paper", "vendor", "pdfjs");
for (const line of read(join(vendor, "SOURCE.txt")).split("\n")) {
  const match = /^([0-9a-f]{64})\s+(\S+)$/.exec(line);
  if (match && createHash("sha256").update(readFileSync(join(vendor, match[2]))).digest("hex") !== match[1]) {
    fail(`vendor/pdfjs/${match[2]} が SOURCE.txt のハッシュと違います`);
  }
}

if (!existsSync(join(ROOT, ".nojekyll"))) fail(".nojekyll がありません");
if (!existsSync(join(ROOT, "icon.png"))) fail("icon.png がありません");

if (problems.length > 0) {
  console.error(problems.map((problem) => `NG ${problem}`).join("\n"));
  process.exit(1);
}

console.log(`ok: ${pages.length} ページ、sitemap ${locations.length} 件、PDF ${pdf.length} バイト (sha256 ${pdfHash.slice(0, 12)})`);
