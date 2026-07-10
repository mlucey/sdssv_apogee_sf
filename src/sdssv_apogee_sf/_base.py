"""
Data directory management and download utilities.
"""
import hashlib
import os
from pathlib import Path

import requests
from tqdm import tqdm

_ENV_VAR = "SDSSV_APOGEE_SF_DATADIR"
_DEFAULT_DATADIR = Path.home() / ".sdssv_apogee_sf"


def get_datadir() -> Path:
    """Return the data directory, respecting SDSSV_APOGEE_SF_DATADIR env var."""
    return Path(os.environ.get(_ENV_VAR, _DEFAULT_DATADIR))


def _download(url: str, dest: Path, md5sum: str | None = None) -> None:
    """Download *url* to *dest*, with a progress bar."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    try:
        r = requests.get(url, stream=True, timeout=60)
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(tmp, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True,
            desc=f"Downloading {dest.name}"
        ) as bar:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
        if md5sum is not None:
            actual = hashlib.md5(tmp.read_bytes()).hexdigest()
            if actual != md5sum:
                tmp.unlink()
                raise ValueError(
                    f"MD5 mismatch for {dest.name}: expected {md5sum}, got {actual}"
                )
        tmp.rename(dest)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise


class DownloadMixin:
    """
    Mixin that auto-downloads named data files from URLs to the data directory.

    Subclasses declare a ``datafiles`` class attribute mapping filename → URL.
    Call ``_get_data(filename)`` to get a ``Path`` to the local copy, downloading
    if necessary.

    Example::

        class MySF(DownloadMixin):
            datafiles = {
                "denominator.npz": "https://zenodo.org/record/.../denominator.npz"
            }

            def __init__(self):
                path = self._get_data("denominator.npz")
                self._denom = np.load(path)
    """

    datafiles: dict[str, str] = {}

    def _get_data(self, filename: str) -> Path:
        """Return path to *filename*, downloading from ``datafiles[filename]`` if missing."""
        if filename not in self.datafiles:
            raise KeyError(f"{filename!r} is not listed in datafiles for {type(self).__name__}")
        dest = get_datadir() / filename
        if not dest.exists():
            url = self.datafiles[filename]
            print(f"Downloading {filename} to {dest} …")
            _download(url, dest)
        return dest
