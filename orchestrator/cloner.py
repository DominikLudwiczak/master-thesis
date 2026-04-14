import shutil
import pathlib
import subprocess

import git


def clone_repo(github_url: str, workspace_path: str) -> tuple[pathlib.Path, str]:
    repo_name = github_url.rstrip("/").split("/")[-1].replace(".git", "")
    dest = pathlib.Path(workspace_path) / repo_name
    if dest.exists():
        shutil.rmtree(dest)
    print(f"[1/3] Cloning {github_url} → {dest}")

    try:
        git.Repo.clone_from(github_url, dest, depth=1)
    except git.exc.GitCommandError as e:
        if "No space left" in str(e) or "cannot create" in str(e):
            print("    full checkout failed (disk space) — retrying with sparse checkout")
            shutil.rmtree(dest, ignore_errors=True)
            _sparse_clone(github_url, dest)
        else:
            raise

    readme_candidates = ["README.md", "readme.md", "README.rst", "README.txt", "README"]
    readme = ""
    for name in readme_candidates:
        p = dest / name
        if p.exists():
            readme = p.read_text(errors="replace")
            break
    return dest, readme


def _sparse_clone(github_url: str, dest: pathlib.Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init"], cwd=dest, check=True)
    subprocess.run(["git", "remote", "add", "origin", github_url], cwd=dest, check=True)
    subprocess.run(["git", "sparse-checkout", "init", "--cone"], cwd=dest, check=True)
    subprocess.run(["git", "fetch", "--depth=1", "origin", "HEAD"], cwd=dest, check=True)
    result = subprocess.run(
        ["git", "checkout", "FETCH_HEAD"],
        cwd=dest, capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"    [warn] sparse checkout partial: {result.stderr[:300]}")
        print("    continuing with whatever was checked out")