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
import requests as http_requests


# ── helpers ───────────────────────────────────────────────────────────────────

def _read_readme(dest: pathlib.Path) -> str:
    for name in ["README.md", "readme.md", "README.rst", "README.txt", "README"]:
        p = dest / name
        if p.exists():
            return p.read_text(errors="replace")
    return ""


def _slug(url: str) -> str:
    """Derive a short directory name from any URL."""
    parsed = urllib.parse.urlparse(url)
    host = parsed.netloc.lower()
    path = parsed.path.rstrip("/")

    # DOI URLs: use the DOI suffix (e.g. 10.5281/zenodo.7037946 → zenodo.7037946)
    if host == "doi.org":
        doi_path = path.lstrip("/")
        name = doi_path.replace("/", ".").replace("m9.", "")
        return name or "repo"

    # Figshare: use article ID
    if "figshare.com" in host:
        match = re.search(r"/(\d+)", path)
        if match:
            return f"figshare_{match.group(1)}"

    # GitHub /tree/ref: use repo name, not the ref
    if "github.com" in host:
        parts = path.strip("/").split("/")
        if len(parts) >= 2:
            return parts[1].replace(".git", "")

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
        # Use specific ref if URL points to /tree/<ref>, otherwise HEAD
        ref = "HEAD"
        if len(parts) >= 4 and parts[2] == "tree":
            ref = "/".join(parts[3:])
        zip_url = f"https://github.com/{owner}/{repo}/archive/{ref}.zip"
        print(f"    Downloading ZIP {zip_url} …")
        dest.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
            tmp_path = pathlib.Path(tmp.name)
            resp = http_requests.get(zip_url, stream=True, timeout=300)
            resp.raise_for_status()
            downloaded = 0
            last_logged = 0
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                tmp.write(chunk)
                downloaded += len(chunk)
                mb = downloaded // (1024 * 1024)
                if mb >= last_logged + 50:
                    print(f"    {mb} MB downloaded …")
                    last_logged = mb
            print(f"    Download complete: {downloaded / (1024*1024):.0f} MB")
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


class FigshareStrategy(CloneStrategy):
    def can_handle(self, url: str) -> bool:
        host = urllib.parse.urlparse(url).netloc.lower()
        return "figshare.com" in host

    def fetch(self, url: str, dest: pathlib.Path) -> None:
        # Extract article ID from URL like /articles/software/NAME/ID/VERSION
        match = re.search(r"figshare\.com/articles/\w+/\w+/(\d+)(?:/(\d+))?", url)
        if not match:
            raise ValueError(f"Cannot extract Figshare article ID from: {url}")
        article_id = match.group(1)
        version = match.group(2) or ""
        api_url = f"https://api.figshare.com/v2/articles/{article_id}"
        if version:
            api_url += f"/versions/{version}"
        print(f"    Fetching Figshare metadata: {api_url}")
        with urllib.request.urlopen(api_url) as resp:
            meta = json.loads(resp.read())
        files = meta.get("files", [])
        if not files:
            raise RuntimeError(f"No files found in Figshare article {article_id}")
        dest.mkdir(parents=True, exist_ok=True)
        for entry in files:
            file_url = entry.get("download_url", "")
            file_name = entry.get("name", file_url.split("/")[-1])
            print(f"    Downloading {file_name} …")
            with urllib.request.urlopen(file_url) as resp:
                data = resp.read()
            _extract_or_save(data, file_name, dest)


class DOIStrategy(CloneStrategy):
    """Resolve doi.org URLs to their final destination and delegate."""

    def can_handle(self, url: str) -> bool:
        host = urllib.parse.urlparse(url).netloc.lower()
        return host == "doi.org"

    def fetch(self, url: str, dest: pathlib.Path) -> None:
        print(f"    Resolving DOI: {url}")
        resp = http_requests.head(url, allow_redirects=True, timeout=30)
        resolved = resp.url
        print(f"    Resolved to: {resolved}")
        # Find an appropriate strategy for the resolved URL
        strategy = next(
            (s for s in _STRATEGIES if not isinstance(s, DOIStrategy) and s.can_handle(resolved)),
            GitStrategy(),
        )
        print(f"    Delegating to {strategy.__class__.__name__}")
        strategy.fetch(resolved, dest)


class GitStrategy(CloneStrategy):
    _GIT_HOSTS = ("gitlab.com", "bitbucket.org", "codeberg.org", "sourceforge.net")

    def can_handle(self, url: str) -> bool:
        host = urllib.parse.urlparse(url).netloc.lower()
        return any(h in host for h in self._GIT_HOSTS) or url.endswith(".git")

    def fetch(self, url: str, dest: pathlib.Path) -> None:
        git.Repo.clone_from(url, dest, depth=1)


# Ordered by specificity — first match wins; GitStrategy is the catch-all fallback
_STRATEGIES = [
    DOIStrategy(),
    GitHubZipStrategy(),
    ZenodoStrategy(),
    FigshareStrategy(),
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
