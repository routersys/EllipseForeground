from pathlib import Path

IMAGE_SUFFIXES = (".png", ".webp", ".jpg", ".jpeg")


def listed_paths(source: Path) -> list[Path]:
    if source.is_dir():
        return sorted(path for path in source.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    return [Path(line.strip()) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
