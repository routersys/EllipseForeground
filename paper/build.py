import shutil
import subprocess
from pathlib import Path

from paper.document import SOURCE, render
from paper.figures import build_figures
from paper.results import Results
from paper.study import Study, Timing

CHROME = Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe")
VIRTUAL_TIME_MS = 6000
PDF_TIMEOUT = 300


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
    subprocess.run(command, check=True, capture_output=True, text=True, timeout=PDF_TIMEOUT)


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
        print_to_pdf(page, output / "paper.pdf", chrome)
    return page
