import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

DATA = Path(__file__).resolve().parent.parent / "data"
ALLOWED_LICENSES = {"CC0-1.0", "CC-BY-4.0"}
OWN_FILES = {"sprite1.npz", "sprite2.npz"}


@pytest.mark.parametrize("folder", ["sprites", "backgrounds"])
def test_manifest_entries_match_the_files(folder):
    manifest = json.loads((DATA / folder / "manifest.json").read_text(encoding="utf-8"))
    assert manifest
    for entry in manifest:
        path = DATA / folder / entry["file"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"], entry["file"]


@pytest.mark.parametrize("folder", ["sprites", "backgrounds"])
def test_manifest_entries_state_their_origin_and_a_redistributable_license(folder):
    manifest = json.loads((DATA / folder / "manifest.json").read_text(encoding="utf-8"))
    for entry in manifest:
        assert entry["license"] in ALLOWED_LICENSES
        assert entry["author"]
        assert entry["title"]
        assert entry["source"].startswith("https://")


@pytest.mark.parametrize("folder", ["sprites", "backgrounds"])
def test_every_image_in_the_folder_is_listed(folder):
    manifest = json.loads((DATA / folder / "manifest.json").read_text(encoding="utf-8"))
    listed = {entry["file"] for entry in manifest}
    present = {path.name for path in (DATA / folder).iterdir() if path.name != "manifest.json"}
    assert listed == present


def test_manifest_identifiers_are_unique():
    for folder in ("sprites", "backgrounds"):
        manifest = json.loads((DATA / folder / "manifest.json").read_text(encoding="utf-8"))
        assert len({entry["id"] for entry in manifest}) == len(manifest)


def test_own_folder_holds_only_the_character_data():
    assert {path.name for path in (DATA / "own").glob("*.npz")} == OWN_FILES
    assert not any(path.suffix in {".png", ".jpg", ".webp"} for path in (DATA / "own").rglob("*"))


@pytest.mark.parametrize("name", sorted(OWN_FILES))
def test_own_character_data_has_no_background(name):
    with np.load(DATA / "own" / name) as stored:
        alpha, foreground = stored["alpha"], stored["foreground"]
    assert alpha.dtype == np.float32 and foreground.dtype == np.float32
    assert alpha.shape == foreground.shape[:2]
    assert alpha.min() >= 0.0 and alpha.max() <= 1.0
    assert not foreground[alpha == 0].any()
