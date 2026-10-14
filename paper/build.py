import json
import re
import shutil
import subprocess
from html import unescape
from pathlib import Path

from paper.document import SOURCE, render
from paper.figures import build_figures
from paper.results import Results
from paper.study import Study, Timing

CHROME = Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe")
VIRTUAL_TIME_MS = 6000
PDF_TIMEOUT = 300
CONTENT_WIDTH = 635
PROBE = (
    f"<style>html{{width:{CONTENT_WIDTH}px}}</style>"
    "<script>addEventListener('load',()=>{"
    "const wide=[...document.querySelectorAll('table,figure,img,svg,.eq,.alg')]"
    f".filter(e=>e.getBoundingClientRect().right>{CONTENT_WIDTH}.5)"
    ".map(e=>(e.closest('.table-block')?.querySelector('.tcap')?.textContent||e.tagName).slice(0,40));"
    "document.title='OVERFLOW:'+JSON.stringify(wide)})</script>"
)


def print_to_pdf(page: Path, destination: Path, chrome: Path) -> None:
    command = [
        str(chrome),
        "--headless=new",
        "--disable-gpu",
        "--no-pdf-header-footer",
        f"--print-to-pdf={destination.resolve()}",
        f"--virtual-time-budget={VIRTUAL_TIME_MS}",
        page.resolve().as_uri(),
    ]
    subprocess.run(command, check=True, capture_output=True, timeout=PDF_TIMEOUT)


def probe_html(page_text: str) -> str:
    if "</body>" not in page_text:
        raise ValueError("本文の終わりが見つからない")
    return page_text.replace("</body>", PROBE + "</body>")


def parse_overflow(dom: str) -> list[str]:
    match = re.search(r"<title>OVERFLOW:(.*?)</title>", dom, re.DOTALL)
    if match is None:
        raise RuntimeError("幅の検査の結果が読み取れない")
    return json.loads(unescape(match.group(1)))


def overflowing(page: Path, chrome: Path) -> list[str]:
    probe = page.with_name("probe.html")
    probe.write_text(probe_html(page.read_text(encoding="utf-8")), encoding="utf-8", newline="\n")
    try:
        command = [
            str(chrome),
            "--headless=new",
            "--disable-gpu",
            f"--window-size={CONTENT_WIDTH},1200",
            f"--virtual-time-budget={VIRTUAL_TIME_MS}",
            "--dump-dom",
            probe.resolve().as_uri(),
        ]
        result = subprocess.run(command, check=True, capture_output=True, timeout=PDF_TIMEOUT)
    finally:
        probe.unlink(missing_ok=True)
    return parse_overflow(result.stdout.decode("utf-8", errors="replace"))


def build(
    results_directory: Path,
    data_root: Path,
    timing_path: Path,
    output: Path,
    chrome: Path = CHROME,
    figures: bool = True,
    pdf: bool = True,
) -> Path:
    study = Study(Results(results_directory), Timing.read(timing_path))
    output.mkdir(parents=True, exist_ok=True)
    if figures:
        build_figures(study, data_root, output / "figs")
    html, _ = render(study, data_root)
    page = output / "paper.html"
    page.write_text(html, encoding="utf-8", newline="\n")
    shutil.copy(SOURCE / "style.css", output / "style.css")
    if pdf:
        wide = overflowing(page, chrome)
        if wide:
            raise RuntimeError("幅を超える要素があり、PDFが縮小される: " + "、".join(wide))
        print_to_pdf(page, output / "paper.pdf", chrome)
    return page
