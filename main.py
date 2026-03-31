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

OPENHANDS_URL  = os.getenv("OPENHANDS_URL", "http://localhost:3000")
OLLAMA_URL     = os.getenv("OLLAMA_URL",    "http://localhost:11434")
ANALYSIS_MODEL = os.getenv("ANALYSIS_MODEL", "llama3.2:3b")
WORKSPACE_PATH = os.getenv("WORKSPACE_PATH", "/opt/workspace")
RESULTS_DIR    = pathlib.Path(os.getenv("RESULTS_DIR", "/app/results"))
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

class ReproductionResult(BaseModel):
    repo_url:       str
    agent_output:   str
    verdict:        str          # "reproduced" | "failed" | "partial"
    error_type:     str | None   # "env" | "code" | "data" | "timeout" | None
    metrics_found:  dict
    analysis:       str

def clone_repo(github_url: str) -> tuple[pathlib.Path, str]:
    repo_name = github_url.rstrip("/").split("/")[-1].replace(".git", "")
    dest = pathlib.Path(WORKSPACE_PATH) / repo_name
    if dest.exists():
        shutil.rmtree(dest)
    print(f"[1/3] Cloning {github_url} → {dest}")
    git.Repo.clone_from(github_url, dest)

    readme_candidates = ["README.md", "readme.md", "README.rst", "README.txt", "README"]
    readme = ""
    for name in readme_candidates:
        p = dest / name
        if p.exists():
            readme = p.read_text(errors="replace")
            break
    return dest, readme


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

def run_openhands_agent(repo_path: pathlib.Path, readme: str) -> str:
    readme_trimmed = smart_truncate_readme(readme, max_chars=6000)
    task = AGENT_TASK_TEMPLATE.format(readme=readme_trimmed)

    print("[2/3] Sending task to OpenHands agent...")

    resp = requests.post(
        f"{OPENHANDS_URL}/api/conversations",
        json={
            "task": task,
            "workspace_mount_path": str(repo_path),
        },
        timeout=30,
    )
    resp.raise_for_status()
    conversation_id = resp.json()["conversation_id"]

    deadline = time.time() + 1200
    while time.time() < deadline:
        time.sleep(10)
        status_resp = requests.get(
            f"{OPENHANDS_URL}/api/conversations/{conversation_id}",
            timeout=10,
        )
        status_resp.raise_for_status()
        data = status_resp.json()
        state = data.get("status", "running")
        print(f"    agent status: {state}")
        if state in ("finished", "error", "stopped"):
            # Return last assistant message
            messages = data.get("messages", [])
            for msg in reversed(messages):
                if msg.get("role") == "assistant":
                    return msg.get("content", "")
            return str(data)

    return "TIMEOUT: agent did not finish within 20 minutes"


def smart_truncate_readme(readme: str, max_chars: int) -> str:
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
    raw = raw.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"verdict": "unknown", "error_type": None, "metrics_found": {}, "explanation": raw}



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

    slug = github_url.rstrip("/").split("/")[-1]
    out_path = RESULTS_DIR / f"{slug}_{int(time.time())}.json"
    out_path.write_text(result.model_dump_json(indent=2))
    print(f"\nResult saved → {out_path}")
    print(f"Verdict: {result.verdict}")
    print(f"Analysis: {result.analysis}")
    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <github_url>")
        sys.exit(1)
    reproduce(sys.argv[1])
