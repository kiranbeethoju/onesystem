"""Locate OneSystem weights: a local directory, or the GitHub release."""

from __future__ import annotations

import os
import shutil
import sys
import urllib.request
from pathlib import Path
from typing import List, Optional

from onesystem import RELEASE_REPO, RELEASE_TAG

REQUIRED_FILES = ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"]
OPTIONAL_FILES = ["special_tokens_map.json", "vocab.txt", "onesystem.json", "eval.json"]

LOCAL_MODEL_DIR = Path("models/onesystem")


def cache_root() -> Path:
    override = os.environ.get("ONESYSTEM_HOME")
    if override:
        return Path(override)
    return Path.home() / ".cache" / "onesystem"


def has_weights(directory: Path) -> bool:
    return all((directory / name).is_file() for name in REQUIRED_FILES)


def release_url(filename: str, repo: str = RELEASE_REPO, tag: str = RELEASE_TAG) -> str:
    return f"https://github.com/{repo}/releases/download/{tag}/{filename}"


def _download(url: str, destination: Path) -> bool:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "onesystem"})
    try:
        with urllib.request.urlopen(request) as response, partial.open("wb") as handle:
            total = int(response.headers.get("Content-Length") or 0)
            done = 0
            while True:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                handle.write(chunk)
                done += len(chunk)
                if total:
                    sys.stderr.write(f"\r{destination.name}: {done / total:6.1%}")
            if total:
                sys.stderr.write("\n")
    except urllib.error.HTTPError as error:
        if partial.exists():
            partial.unlink()
        if error.code == 404:
            return False
        raise
    shutil.move(str(partial), str(destination))
    return True


def download_release(repo: str = RELEASE_REPO, tag: str = RELEASE_TAG, destination: Optional[Path] = None) -> Path:
    """Fetch the published OneSystem files into the cache and return the directory."""
    directory = destination or (cache_root() / repo.replace("/", "--") / tag)
    if has_weights(directory):
        return directory
    for name in REQUIRED_FILES:
        target = directory / name
        if target.is_file():
            continue
        if not _download(release_url(name, repo, tag), target):
            raise FileNotFoundError(f"{name} is not in release {tag} of {repo}")
    for name in OPTIONAL_FILES:
        target = directory / name
        if not target.is_file():
            _download(release_url(name, repo, tag), target)
    return directory


def resolve(source: Optional[str] = None) -> Path:
    """Turn a source string into a directory holding OneSystem weights.

    ``None`` means: the local ``models/onesystem`` directory when it has weights,
    otherwise the GitHub release. A path is used as-is. ``github:owner/repo@tag``
    names a specific release.
    """
    if source is None:
        if has_weights(LOCAL_MODEL_DIR):
            return LOCAL_MODEL_DIR
        return download_release()
    if source.startswith("github:"):
        spec = source[len("github:"):]
        repo, _, tag = spec.partition("@")
        return download_release(repo, tag or RELEASE_TAG)
    directory = Path(source)
    if has_weights(directory):
        return directory
    raise FileNotFoundError(
        f"{source} has no OneSystem weights; expected {', '.join(REQUIRED_FILES)}"
    )


def published_files(directory: Path) -> List[Path]:
    return [directory / name for name in REQUIRED_FILES + OPTIONAL_FILES if (directory / name).is_file()]
