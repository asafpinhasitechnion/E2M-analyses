"""Download selected scPerturb h5ad files from Zenodo."""

from __future__ import annotations

import hashlib
import json
import socket
import time
from pathlib import Path
from typing import Callable, Iterable, Optional, TypeVar
from urllib.error import URLError
from urllib.request import Request, urlopen

import config


ZENODO_API_URL = f"https://zenodo.org/api/records/{config.ZENODO_RECORD_ID}"
ZENODO_FILE_URL = f"https://zenodo.org/records/{config.ZENODO_RECORD_ID}/files/{{name}}"

DEFAULT_RETRIES = 6
DEFAULT_BACKOFF_BASE = 4.0  # seconds; doubled per retry
NET_TIMEOUT = 60.0
USER_AGENT = "single_cell_analysis-downloader/1.0"


T = TypeVar("T")


def _retry(fn: Callable[[], T], what: str, retries: int = DEFAULT_RETRIES) -> T:
    """Retry ``fn`` on transient network errors with exponential backoff."""
    last_exc: Optional[BaseException] = None
    for attempt in range(retries + 1):
        try:
            return fn()
        except (URLError, socket.gaierror, socket.timeout, ConnectionError, TimeoutError) as exc:
            last_exc = exc
            if attempt == retries:
                break
            wait = DEFAULT_BACKOFF_BASE * (2 ** attempt)
            print(f"  [warn] {what} failed ({exc}); retry {attempt + 1}/{retries} in {wait:.0f}s")
            time.sleep(wait)
    raise RuntimeError(f"{what} failed after {retries + 1} attempts: {last_exc}") from last_exc


def _open(url: str, headers: Optional[dict] = None):
    req = Request(url, headers={"User-Agent": USER_AGENT, **(headers or {})})
    return urlopen(req, timeout=NET_TIMEOUT)


def fetch_record_metadata() -> dict:
    def _do() -> dict:
        with _open(ZENODO_API_URL) as response:
            return json.loads(response.read().decode("utf-8"))

    return _retry(_do, what="fetch Zenodo record metadata")


def file_md5(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _stream_to_part(url: str, tmp: Path, expected_size: int) -> None:
    """One streaming attempt. Resumes from ``tmp`` if it already exists."""
    start = tmp.stat().st_size if tmp.exists() else 0
    headers = {"Range": f"bytes={start}-"} if start > 0 else None
    if start > 0 and expected_size and start >= expected_size:
        return  # already complete

    mode = "ab" if start > 0 else "wb"
    seen = start
    with _open(url, headers=headers) as response, open(tmp, mode) as handle:
        cl = int(response.headers.get("Content-Length") or 0)
        total = expected_size or (start + cl if cl else 0)
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            handle.write(chunk)
            seen += len(chunk)
            if total:
                pct = 100.0 * seen / total
                print(f"\r  {tmp.stem}: {seen / 1e9:.2f}/{total / 1e9:.2f} GB ({pct:5.1f}%)", end="")
            else:
                print(f"\r  {tmp.stem}: {seen / 1e9:.2f} GB", end="")
    print()

    if expected_size and seen < expected_size:
        raise IOError(
            f"stream ended early: got {seen} bytes, expected {expected_size} "
            f"(short by {expected_size - seen} bytes)"
        )


def download_file(url: str, dst: Path, expected_size: int = 0, retries: int = DEFAULT_RETRIES) -> None:
    tmp = dst.with_name(dst.name + ".part")
    dst.parent.mkdir(parents=True, exist_ok=True)
    _retry(lambda: _stream_to_part(url, tmp, expected_size),
           what=f"download {dst.name}", retries=retries)

    if expected_size:
        got = tmp.stat().st_size
        if got < expected_size:
            raise RuntimeError(
                f"{dst.name}: download incomplete ({got} bytes, expected {expected_size}); "
                f"keeping .part for resume."
            )
    tmp.replace(dst)


def resolve_requested_files(record: dict, dataset_keys: Iterable[str]) -> list[dict]:
    wanted_names = {config.DATASET_FILES[key] for key in dataset_keys}
    files = [item for item in record.get("files", []) if item.get("key") in wanted_names]
    found = {item.get("key") for item in files}
    missing = sorted(wanted_names - found)
    if missing:
        raise RuntimeError(f"Zenodo record does not contain expected files: {missing}")
    return files


def _files_from_fallback(dataset_keys: Iterable[str]) -> list[dict]:
    """Build minimal file entries (without size/md5) using a guessed URL pattern."""
    return [
        {
            "key": config.DATASET_FILES[key],
            "size": 0,
            "checksum": "",
            "links": {"self": ZENODO_FILE_URL.format(name=config.DATASET_FILES[key])},
        }
        for key in dataset_keys
    ]
