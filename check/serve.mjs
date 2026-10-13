import { createReadStream, existsSync, readFileSync, statSync } from "node:fs";
import { createServer } from "node:http";
import { extname, join, normalize, resolve, sep } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const ROOT = resolve(fileURLToPath(new URL("../docs/", import.meta.url)));
const PREFIX = "/EllipseForeground";
const TYPES = {
  ".html": "text/html; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".xml": "application/xml; charset=utf-8",
  ".txt": "text/plain; charset=utf-8",
  ".png": "image/png",
  ".pdf": "application/pdf",
};

export function resolvePath(pathname) {
  if (pathname !== PREFIX && !pathname.startsWith(`${PREFIX}/`)) {
    return null;
  }

  const relative = decodeURIComponent(pathname.slice(PREFIX.length)).replace(/^\/+/, "");
  const target = resolve(ROOT, normalize(relative));
  if (target !== ROOT && !target.startsWith(ROOT + sep)) {
    return null;
  }

  if (existsSync(target) && statSync(target).isDirectory()) {
    return join(target, "index.html");
  }

  return target;
}

function send(response, status, type, body) {
  response.writeHead(status, { "content-type": type, "cache-control": "no-store" });
  response.end(body);
}

export function createSiteServer() {
  return createServer((request, response) => {
    if (request.method !== "GET" && request.method !== "HEAD") {
      send(response, 405, "text/plain; charset=utf-8", "Method Not Allowed");
      return;
    }

    const url = new URL(request.url, "http://localhost");
    if (url.pathname === "/") {
      response.writeHead(302, { location: `${PREFIX}/` });
      response.end();
      return;
    }

    if (url.pathname === PREFIX) {
      response.writeHead(301, { location: `${PREFIX}/` });
      response.end();
      return;
    }

    const file = resolvePath(url.pathname);
    if (file === null || !existsSync(file) || !statSync(file).isFile()) {
      send(response, 404, "text/html; charset=utf-8", request.method === "HEAD" ? "" : readFileSync(join(ROOT, "404.html")));
      return;
    }

    const size = statSync(file).size;
    const type = TYPES[extname(file)] ?? "application/octet-stream";
    const range = /^bytes=(\d*)-(\d*)$/.exec(request.headers.range ?? "");
    if (range && (range[1] !== "" || range[2] !== "")) {
      const start = range[1] === "" ? Math.max(0, size - Number(range[2])) : Number(range[1]);
      const end = range[1] === "" || range[2] === "" ? size - 1 : Math.min(Number(range[2]), size - 1);
      if (start > end || start >= size) {
        response.writeHead(416, { "content-range": `bytes */${size}` });
        response.end();
        return;
      }

      response.writeHead(206, {
        "content-type": type,
        "content-range": `bytes ${start}-${end}/${size}`,
        "content-length": end - start + 1,
        "accept-ranges": "bytes",
        "cache-control": "no-store",
      });
      if (request.method === "HEAD") {
        response.end();
      } else {
        createReadStream(file, { start, end }).pipe(response);
      }
      return;
    }

    response.writeHead(200, { "content-type": type, "content-length": size, "accept-ranges": "bytes", "cache-control": "no-store" });
    if (request.method === "HEAD") {
      response.end();
    } else {
      createReadStream(file).pipe(response);
    }
  });
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const port = Number(process.argv[2] ?? 18123);
  createSiteServer().listen(port, "127.0.0.1", () => {
    console.log(`http://127.0.0.1:${port}${PREFIX}/`);
  });
}
