import * as pdfjs from "./vendor/pdfjs/pdf.min.mjs";

const PDF_UNITS_TO_CSS = 96 / 72;
const ZOOM_STEPS = [0.5, 0.75, 1, 1.25, 1.5, 2, 3];
const HIT_LIMIT = 200;
const CONTEXT = 24;
const RADICALS = /[\u2E80-\u2FDF]/;
const RADICALS_GLOBAL = /[\u2E80-\u2FDF]/g;

const messages = JSON.parse(document.getElementById("messages").textContent);
const reader = document.getElementById("reader");
const status = document.getElementById("status");
const viewer = document.getElementById("viewer");
const pageInput = document.getElementById("page");
const pageCount = document.getElementById("pages");
const zoomSelect = document.getElementById("zoom");
const search = document.getElementById("find");
const query = document.getElementById("query");
const hits = document.getElementById("hits");
const summary = document.getElementById("summary");

const format = (template, values) => template.replace(/\{(\w+)\}/g, (_, key) => String(values[key]));
const megabytes = (bytes) => (bytes / 1048576).toFixed(1);
const plain = (text) => (RADICALS.test(text) ? text.replace(RADICALS_GLOBAL, (character) => character.normalize("NFKC")) : text);

let pdf;
let sizes = [];
let states = [];
let zoom = "fit";
let scale = 1;
let current = 1;
let observer;

function availableWidth() {
  return Math.max(200, viewer.clientWidth - 4);
}

function computeScale() {
  if (zoom === "fit") {
    return availableWidth() / Math.max(...sizes.map((size) => size.width));
  }
  return zoom * PDF_UNITS_TO_CSS;
}

function release(state) {
  state.generation += 1;
  state.task?.cancel();
  state.task = null;
  state.rendered = false;
  state.element.replaceChildren();
}

async function render(index) {
  const state = states[index];
  if (state.rendered && state.renderedScale === scale) {
    return;
  }

  state.generation += 1;
  const generation = state.generation;
  state.task?.cancel();
  const page = await pdf.getPage(index + 1);
  if (generation !== state.generation || !state.visible) {
    return;
  }

  const viewport = page.getViewport({ scale });
  const output = window.devicePixelRatio || 1;
  const canvas = document.createElement("canvas");
  canvas.width = Math.floor(viewport.width * output);
  canvas.height = Math.floor(viewport.height * output);
  const task = page.render({
    canvas,
    viewport,
    transform: output === 1 ? undefined : [output, 0, 0, output, 0, 0],
  });
  state.task = task;
  try {
    await task.promise;
  } catch (error) {
    if (error instanceof pdfjs.RenderingCancelledException) {
      return;
    }
    throw error;
  }

  if (generation !== state.generation) {
    return;
  }

  const textLayer = document.createElement("div");
  textLayer.className = "textLayer";
  state.element.replaceChildren(canvas, textLayer);
  state.rendered = true;
  state.renderedScale = scale;
  await new pdfjs.TextLayer({ textContentSource: page.streamTextContent(), container: textLayer, viewport }).render();
  for (const span of textLayer.querySelectorAll("span")) {
    span.textContent = plain(span.textContent);
  }
}

function layout() {
  scale = computeScale();
  states.forEach((state, index) => {
    const { element } = state;
    element.style.width = `${sizes[index].width * scale}px`;
    element.style.height = `${sizes[index].height * scale}px`;
    element.style.setProperty("--total-scale-factor", scale);
    if (state.visible) {
      render(index).catch(report);
    }
  });
}

function updateZoomSelect() {
  const value = zoom === "fit" ? "fit" : String(zoom);
  if (![...zoomSelect.options].some((option) => option.value === value)) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = `${Math.round(zoom * 100)}%`;
    zoomSelect.append(option);
  }

  zoomSelect.value = value;
}

function setZoom(value) {
  const anchor = current;
  zoom = value;
  updateZoomSelect();
  layout();
  goTo(anchor, false);
}

function effectiveZoom() {
  return scale / PDF_UNITS_TO_CSS;
}

function stepZoom(direction) {
  const now = effectiveZoom();
  const candidates = direction > 0 ? ZOOM_STEPS.filter((step) => step > now + 1e-6) : ZOOM_STEPS.filter((step) => step < now - 1e-6).reverse();
  if (candidates.length > 0) {
    setZoom(candidates[0]);
  }
}

function goTo(number, smooth = true) {
  const bounded = Math.min(Math.max(1, Math.round(number)), states.length);
  states[bounded - 1].element.scrollIntoView({ block: "start", behavior: smooth ? "smooth" : "auto" });
  setCurrent(bounded);
}

function setCurrent(number) {
  current = number;
  pageInput.value = String(number);
  const hash = `#page=${number}`;
  if (location.hash !== hash) {
    history.replaceState(null, "", hash);
  }
}

function trackCurrent() {
  const line = window.innerHeight / 3;
  let best = current;
  for (let index = 0; index < states.length; index += 1) {
    const rectangle = states[index].element.getBoundingClientRect();
    if (rectangle.top <= line && rectangle.bottom > line) {
      best = index + 1;
      break;
    }
  }

  if (best !== current) {
    setCurrent(best);
  }
}

function report(error) {
  console.error(error);
  status.textContent = messages.failed;
}

function normalized(text) {
  const characters = [];
  const origins = [];
  let origin = 0;
  for (const character of text) {
    const folded = character.normalize("NFKC").toLowerCase();
    if (!/^\s*$/.test(folded)) {
      for (const part of folded) {
        characters.push(part);
        origins.push(origin);
      }
    }

    origin += character.length;
  }

  return { compact: characters.join(""), origins };
}

let pageTexts = null;

async function loadTexts() {
  if (pageTexts) {
    return pageTexts;
  }

  const texts = [];
  for (let number = 1; number <= pdf.numPages; number += 1) {
    status.textContent = format(messages.searching, { page: number, pages: pdf.numPages });
    const page = await pdf.getPage(number);
    const content = await page.getTextContent();
    const text = content.items.map((item) => item.str + (item.hasEOL ? " " : "")).join("");
    texts.push({ text, ...normalized(text) });
  }

  pageTexts = texts;
  status.textContent = format(messages.ready, { pages: pdf.numPages });
  return texts;
}

async function find(event) {
  event.preventDefault();
  const needle = normalized(query.value).compact;
  hits.replaceChildren();
  summary.textContent = "";
  if (needle === "") {
    return;
  }

  const texts = await loadTexts();
  let count = 0;
  for (let index = 0; index < texts.length; index += 1) {
    const { text, compact, origins } = texts[index];
    let from = 0;
    for (let at = compact.indexOf(needle, from); at !== -1; at = compact.indexOf(needle, from)) {
      count += 1;
      if (count <= HIT_LIMIT) {
        const begin = Math.max(0, origins[at] - CONTEXT);
        const end = Math.min(text.length, origins[at + needle.length - 1] + CONTEXT + 1);
        const item = document.createElement("li");
        const button = document.createElement("button");
        button.type = "button";
        button.dataset.page = String(index + 1);
        button.textContent = format(messages.page, { page: index + 1 });
        item.append(button, ` ${plain(text.slice(begin, end)).replace(/\s+/g, " ")}`);
        hits.append(item);
      }

      from = at + needle.length;
    }
  }

  summary.textContent = count === 0
    ? messages.noResults
    : format(messages.results, { count, more: count > HIT_LIMIT ? format(messages.limited, { limit: HIT_LIMIT }) : "" });
}

async function main() {
  pdfjs.GlobalWorkerOptions.workerSrc = new URL("./vendor/pdfjs/pdf.worker.min.mjs", import.meta.url).href;
  const task = pdfjs.getDocument({ url: new URL(reader.dataset.src, location.href).href });
  task.onProgress = ({ loaded, total }) => {
    status.textContent = total
      ? format(messages.loadingOf, { loaded: megabytes(loaded), total: megabytes(total) })
      : format(messages.loading, { loaded: megabytes(loaded) });
  };
  pdf = await task.promise;

  for (let number = 1; number <= pdf.numPages; number += 1) {
    const view = (await pdf.getPage(number)).getViewport({ scale: 1 });
    sizes.push({ width: view.width, height: view.height });
  }

  observer = new IntersectionObserver(
    (entries) => {
      for (const entry of entries) {
        const index = Number(entry.target.dataset.index);
        const state = states[index];
        state.visible = entry.isIntersecting;
        if (entry.isIntersecting) {
          render(index).catch(report);
        } else {
          release(state);
        }
      }
    },
    { rootMargin: "100% 0px" },
  );

  states = sizes.map((_, index) => {
    const element = document.createElement("div");
    element.className = "page";
    element.id = `page-${index + 1}`;
    element.dataset.index = String(index);
    element.setAttribute("aria-label", format(messages.page, { page: index + 1 }));
    viewer.append(element);
    return { element, visible: false, rendered: false, renderedScale: 0, generation: 0, task: null };
  });

  pageInput.max = String(pdf.numPages);
  pageCount.textContent = String(pdf.numPages);
  status.textContent = format(messages.ready, { pages: pdf.numPages });
  layout();
  states.forEach((state) => observer.observe(state.element));

  const requested = Number(new URLSearchParams(location.hash.slice(1)).get("page"));
  if (requested >= 1) {
    goTo(requested, false);
  } else {
    setCurrent(1);
  }

  let scrolling = false;
  window.addEventListener("scroll", () => {
    if (!scrolling) {
      scrolling = true;
      requestAnimationFrame(() => {
        scrolling = false;
        trackCurrent();
      });
    }
  }, { passive: true });

  let resizing = 0;
  window.addEventListener("resize", () => {
    clearTimeout(resizing);
    resizing = setTimeout(() => {
      if (zoom === "fit") {
        setZoom("fit");
      }
    }, 150);
  });

  window.addEventListener("hashchange", () => {
    const number = Number(new URLSearchParams(location.hash.slice(1)).get("page"));
    if (number >= 1 && number !== current) {
      goTo(number);
    }
  });

  document.getElementById("previous").addEventListener("click", () => goTo(current - 1));
  document.getElementById("next").addEventListener("click", () => goTo(current + 1));
  document.getElementById("zoomOut").addEventListener("click", () => stepZoom(-1));
  document.getElementById("zoomIn").addEventListener("click", () => stepZoom(1));
  zoomSelect.addEventListener("change", () => setZoom(zoomSelect.value === "fit" ? "fit" : Number(zoomSelect.value)));
  pageInput.addEventListener("change", () => goTo(Number(pageInput.value)));
  search.addEventListener("submit", (event) => find(event).catch(report));
  hits.addEventListener("click", (event) => {
    const button = event.target.closest("button[data-page]");
    if (button) {
      goTo(Number(button.dataset.page));
    }
  });
  for (const control of document.querySelectorAll("#toolbar [disabled]")) {
    control.disabled = false;
  }
}

main().catch(report);
