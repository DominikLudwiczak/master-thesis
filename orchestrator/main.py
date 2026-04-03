"""
Orchestrator: clone repo → hand to OpenHands agent → analyze with Ollama.

Usage:
    python main.py <github_url>
    docker compose run orchestrator python main.py https://github.com/org/repo
"""

import sys
import os
import json
import time
import shutil
import pathlib
import requests
import ollama
import git
from pydantic import BaseModel


# ── Config (from env / defaults) ──────────────────────────────────────────────

OPENHANDS_URL  = os.getenv("OPENHANDS_URL", "http://localhost:3000")
OLLAMA_URL     = os.getenv("OLLAMA_URL",    "http://localhost:11434")
ANALYSIS_MODEL = os.getenv("ANALYSIS_MODEL", "llama3.2:3b")
WORKSPACE_PATH = os.getenv("WORKSPACE_PATH", "/opt/workspace")
RESULTS_DIR    = pathlib.Path(os.getenv("RESULTS_DIR", "/app/results"))
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Pydantic result schema ─────────────────────────────────────────────────────

class ReproductionResult(BaseModel):
    repo_url:       str
    agent_output:   str          # raw OpenHands final response
    verdict:        str          # "reproduced" | "failed" | "partial"
    error_type:     str | None   # "env" | "code" | "data" | "timeout" | None
    metrics_found:  dict         # any numeric results the agent reported
    analysis:       str          # Ollama's natural-language verdict


# ── Step 1: clone ──────────────────────────────────────────────────────────────

def clone_repo(github_url: str) -> tuple[pathlib.Path, str]:
    repo_name = github_url.rstrip("/").split("/")[-1].replace(".git", "")
    dest = pathlib.Path(WORKSPACE_PATH) / repo_name
    if dest.exists():
        shutil.rmtree(dest)
    print(f"[1/3] Cloning {github_url} → {dest}")

    # Always shallow clone; fall back to sparse checkout if disk fills up
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
    """Shallow clone skipping large data/dataset/model dirs to save disk space."""
    import subprocess
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init"], cwd=dest, check=True)
    subprocess.run(["git", "remote", "add", "origin", github_url], cwd=dest, check=True)
    subprocess.run(["git", "sparse-checkout", "init", "--cone"], cwd=dest, check=True)
    # Fetch only — let checkout be partial if it has to be
    subprocess.run(["git", "fetch", "--depth=1", "origin", "HEAD"], cwd=dest, check=True)
    result = subprocess.run(
        ["git", "checkout", "FETCH_HEAD"],
        cwd=dest, capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"    [warn] sparse checkout partial: {result.stderr[:300]}")
        print("    continuing with whatever was checked out")


# ── Step 2: run OpenHands agent ────────────────────────────────────────────────

AGENT_TASK_TEMPLATE = """\
You are reproducing a scientific experiment from a GitHub repository.

The repository has been cloned to /workspace.

Your goal:
1. Read the README carefully and identify the reproduction steps.
2. Install all dependencies.
3. Run the experiment exactly as described.
4. Capture ALL output (stdout, stderr, metrics, figures saved, etc.).
5. If a step fails, diagnose why and try to fix it (once).
6. At the end, output a JSON block like this:

```json
{{
  "success": true or false,
  "metrics": {{}},
  "error": null or "description of what went wrong",
  "steps_completed": []
}}
```

README:
---
{readme}
---
"""

def ensure_openhands_settings():
    """
    OpenHands 1.x requires settings to be POSTed at least once before
    POST /api/conversations will work — even with LLM env vars set.
    Safe to call on every run.
    """
    agent_model = os.getenv("AGENT_MODEL", "qwen2.5-coder:14b")
    ollama_base  = os.getenv("OLLAMA_URL", "http://ollama:11434")
    payload = {
        "llm_model":    f"ollama/{agent_model}",
        "llm_base_url": ollama_base,
        "llm_api_key":  "ollama",
    }
    r = requests.post(f"{OPENHANDS_URL}/api/settings", json=payload, timeout=10)
    if r.status_code not in (200, 201):
        print(f"    [warn] settings POST returned {r.status_code}: {r.text[:300]}")
    else:
        print("    settings saved OK")


def run_openhands_agent(repo_path: pathlib.Path, readme: str) -> str:
    """
    Send task to OpenHands via its REST API and poll for completion.
    Returns the agent's final text output.
    """
    readme_trimmed = smart_truncate_readme(readme, max_chars=6000)
    task = AGENT_TASK_TEMPLATE.format(readme=readme_trimmed)

    print("[2/3] Sending task to OpenHands agent...")
    ensure_openhands_settings()

    # Create conversation
    resp = requests.post(
        f"{OPENHANDS_URL}/api/conversations",
        json={"initial_user_msg": task},
        timeout=30,
    )
    if not resp.ok:
        raise RuntimeError(
            f"OpenHands POST /api/conversations failed {resp.status_code}: {resp.text[:300]}"
        )
    conversation_id = resp.json()["conversation_id"]
    print(f"    conversation id: {conversation_id}")

    # Poll until done (max 20 min)
    deadline = time.time() + 1200
    while time.time() < deadline:
        time.sleep(10)
        status_resp = requests.get(
            f"{OPENHANDS_URL}/api/conversations/{conversation_id}",
            timeout=60,
        )
        status_resp.raise_for_status()
        data = status_resp.json()
        state = data.get("status", "running")
        print(f"    agent status: {state}")
        if state in ("finished", "error", "stopped"):
            messages = data.get("messages", [])
            for msg in reversed(messages):
                if msg.get("role") == "assistant":
                    return msg.get("content", "")
            return str(data)

    return "TIMEOUT: agent did not finish within 20 minutes"


def smart_truncate_readme(readme: str, max_chars: int) -> str:
    """Keep sections that mention install/usage/experiment/run."""
    if len(readme) <= max_chars:
        return readme
    priority_keywords = ["install", "usage", "run", "experiment", "reproduc", "requirement", "setup"]
    lines = readme.splitlines()
    scored, current_section, score = [], [], 0
    for line in lines:
        low = line.lower()
        if low.startswith("#"):
            if current_section:
                scored.append((score, "\n".join(current_section)))
            current_section, score = [line], sum(k in low for k in priority_keywords)
        else:
            current_section.append(line)
    if current_section:
        scored.append((score, "\n".join(current_section)))
    scored.sort(key=lambda x: -x[0])
    result = ""
    for _, section in scored:
        if len(result) + len(section) > max_chars:
            break
        result += section + "\n\n"
    return result or readme[:max_chars]


# ── Step 3: analyze with Ollama ────────────────────────────────────────────────

ANALYSIS_PROMPT = """\
You are analyzing the output of an automated scientific experiment reproduction attempt.

Agent output:
---
{agent_output}
---

Please respond with a JSON object only (no markdown, no preamble):
{{
  "verdict": "reproduced" | "partial" | "failed",
  "error_type": null | "env" | "code" | "data" | "timeout",
  "metrics_found": {{}},
  "explanation": "2-3 sentence summary of what happened"
}}

verdict meanings:
- reproduced: experiment ran to completion with plausible results
- partial: some steps succeeded but final result was not obtained
- failed: could not run the experiment at all

error_type meanings (if verdict != reproduced):
- env: missing dependency, wrong Python version, CUDA issue
- code: bug in the repo code itself
- data: missing dataset or download failed
- timeout: ran out of time
"""

def analyze_with_ollama(agent_output: str) -> dict:
    print("[3/3] Analyzing result with Ollama...")
    client = ollama.Client(host=OLLAMA_URL)
    response = client.chat(
        model=ANALYSIS_MODEL,
        messages=[{"role": "user", "content": ANALYSIS_PROMPT.format(agent_output=agent_output[:4000])}],
    )
    raw = response["message"]["content"].strip()
    # Strip accidental markdown fences
    raw = raw.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"verdict": "unknown", "error_type": None, "metrics_found": {}, "explanation": raw}


# ── Main ───────────────────────────────────────────────────────────────────────

def reproduce(github_url: str) -> ReproductionResult:
    repo_path, readme = clone_repo(github_url)
    agent_output      = run_openhands_agent(repo_path, readme)
    analysis          = analyze_with_ollama(agent_output)

    result = ReproductionResult(
        repo_url      = github_url,
        agent_output  = agent_output,
        verdict       = analysis.get("verdict", "unknown"),
        error_type    = analysis.get("error_type"),
        metrics_found = analysis.get("metrics_found", {}),
        analysis      = analysis.get("explanation", ""),
    )

    # Save JSON result
    slug = github_url.rstrip("/").split("/")[-1]
    out_path = RESULTS_DIR / f"{slug}_{int(time.time())}.json"
    out_path.write_text(result.model_dump_json(indent=2))
    print(f"\nResult saved → {out_path}")
    print(f"Verdict: {result.verdict}")
    print(f"Analysis: {result.analysis}")
    return result


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        github_url = sys.argv[1]
    elif os.environ.get("GITHUB_REPO"):
        github_url = os.environ.get("GITHUB_REPO")
    else:
        print("Usage: python main.py <github_url>")
        sys.exit(1)
    reproduce(github_url)