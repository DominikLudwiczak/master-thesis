import abc
import io
import json
import re
import shutil
import tarfile
import tempfile
import zipfile
import pathlib
import urllib.parse
import urllib.request

import git


# ── helpers ───────────────────────────────────────────────────────────────────

def _read_readme(dest: pathlib.Path) -> str:
    for name in ["README.md", "readme.md", "README.rst", "README.txt", "README"]:
        p = dest / name
        if p.exists():
            return p.read_text(errors="replace")
    return ""


def _slug(url: str) -> str:
    """Derive a short directory name from any URL."""
    path = urllib.parse.urlparse(url).path.rstrip("/")
    name = path.split("/")[-1]
    for ext in (".tar.gz", ".tgz", ".tar.bz2", ".tar.xz", ".zip"):
        if name.lower().endswith(ext):
            name = name[: -len(ext)]
            break
    name = name.replace(".git", "")
    return name or "repo"


def _extract_or_save(data: bytes, file_name: str, dest: pathlib.Path) -> None:
    name_lower = file_name.lower()
    if name_lower.endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            zf.extractall(dest)
    elif any(name_lower.endswith(ext) for ext in (".tar.gz", ".tgz", ".tar.bz2", ".tar.xz")):
        with tarfile.open(fileobj=io.BytesIO(data)) as tf:
            tf.extractall(dest)
    else:
        (dest / file_name).write_bytes(data)


# ── strategies ────────────────────────────────────────────────────────────────

class CloneStrategy(abc.ABC):
    @abc.abstractmethod
    def can_handle(self, url: str) -> bool: ...

    @abc.abstractmethod
    def fetch(self, url: str, dest: pathlib.Path) -> None: ...


class GitHubZipStrategy(CloneStrategy):
    def can_handle(self, url: str) -> bool:
        return "github.com" in urllib.parse.urlparse(url).netloc.lower()

    def fetch(self, url: str, dest: pathlib.Path) -> None:
        parts = urllib.parse.urlparse(url).path.strip("/").split("/")
        owner, repo = parts[0], parts[1].replace(".git", "")
        zip_url = f"https://github.com/{owner}/{repo}/archive/HEAD.zip"
        print(f"    Downloading ZIP {zip_url} …")
        dest.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
            tmp_path = pathlib.Path(tmp.name)
            with urllib.request.urlopen(zip_url) as resp:
                shutil.copyfileobj(resp, tmp)
        try:
            with zipfile.ZipFile(tmp_path) as zf:
                # GitHub ZIPs wrap content in a top-level "{repo}-{sha}/" dir — strip it
                for member in zf.infolist():
                    rel = member.filename.split("/", 1)
                    if len(rel) < 2 or not rel[1]:
                        continue
                    member.filename = rel[1]
                    zf.extract(member, dest)
        finally:
            tmp_path.unlink(missing_ok=True)


class ZenodoStrategy(CloneStrategy):
    def can_handle(self, url: str) -> bool:
        return "zenodo.org" in urllib.parse.urlparse(url).netloc.lower()

    def fetch(self, url: str, dest: pathlib.Path) -> None:
        match = re.search(r"zenodo\.org/records?/(\d+)", url)
        if not match:
            raise ValueError(f"Cannot extract Zenodo record ID from: {url}")
        record_id = match.group(1)
        api_url = f"https://zenodo.org/api/records/{record_id}"

        print(f"    Fetching Zenodo metadata: {api_url}")
        with urllib.request.urlopen(api_url) as resp:
            meta = json.loads(resp.read())

        files = meta.get("files") or meta.get("metadata", {}).get("_files", [])
        if not files:
            raise RuntimeError(f"No files found in Zenodo record {record_id}")

        dest.mkdir(parents=True, exist_ok=True)
        for entry in files:
            file_url = entry.get("links", {}).get("self") or entry.get("download_url") or entry["key"]
            file_name = entry.get("key") or file_url.split("/")[-1]
            print(f"    Downloading {file_name} …")
            with urllib.request.urlopen(file_url) as resp:
                data = resp.read()
            _extract_or_save(data, file_name, dest)


class ArchiveStrategy(CloneStrategy):
    _EXTENSIONS = (".zip", ".tar.gz", ".tgz", ".tar.bz2", ".tar.xz")

    def can_handle(self, url: str) -> bool:
        path = urllib.parse.urlparse(url).path.lower()
        return any(path.endswith(ext) for ext in self._EXTENSIONS)

    def fetch(self, url: str, dest: pathlib.Path) -> None:
        file_name = urllib.parse.urlparse(url).path.split("/")[-1]
        print(f"    Downloading {file_name} …")
        with urllib.request.urlopen(url) as resp:
            data = resp.read()
        dest.mkdir(parents=True, exist_ok=True)
        _extract_or_save(data, file_name, dest)


class GitStrategy(CloneStrategy):
    _GIT_HOSTS = ("gitlab.com", "bitbucket.org", "codeberg.org", "sourceforge.net")

    def can_handle(self, url: str) -> bool:
        host = urllib.parse.urlparse(url).netloc.lower()
        return any(h in host for h in self._GIT_HOSTS) or url.endswith(".git")

    def fetch(self, url: str, dest: pathlib.Path) -> None:
        git.Repo.clone_from(url, dest, depth=1)


# Ordered by specificity — first match wins; GitStrategy is the catch-all fallback
_STRATEGIES = [
    GitHubZipStrategy(),
    ZenodoStrategy(),
    ArchiveStrategy(),
    GitStrategy(),
]


# ── public API ────────────────────────────────────────────────────────────────

def clone_repo(url: str, workspace_path: str) -> tuple[pathlib.Path, str]:
    slug = _slug(url)
    dest = pathlib.Path(workspace_path) / slug
    if dest.exists():
        shutil.rmtree(dest)

    strategy = next((s for s in _STRATEGIES if s.can_handle(url)), GitStrategy())
    print(f"[1/3] {strategy.__class__.__name__}: {url} → {dest}")
    strategy.fetch(url, dest)

    return dest, _read_readme(dest)
