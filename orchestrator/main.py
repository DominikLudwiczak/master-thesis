"""
Orchestrator: clone repo → hand to OpenHands agent → analyze with Ollama.

Usage:
    python main.py                          # runs REPOS list defined below
    python main.py <url1> [url2] [url3]    # runs given URLs
    GITHUB_REPOS="url1,url2" python main.py
    docker compose up orchestrator
"""

import sys
import os
import json
import time
import pathlib
import subprocess

from models import ReproductionResult
from cloner import clone_repo
from agent import run_openhands_agent
from analyzer import analyze_with_ollama

REPOS = [
    "https://github.com/coinse/fonte",
    "https://github.com/Generative-Program-Analysis/icse23-artifact-evaluation",
    "https://github.com/apicad1/artifact",
    "https://zenodo.org/records/7626930",
    "https://github.com/SageSELab/AidUI",
    "https://github.com/soarsmu/Chronos"
]

OPENHANDS_URL  = os.getenv("OPENHANDS_URL", "http://localhost:3000")
OLLAMA_URL     = os.getenv("OLLAMA_URL",    "http://localhost:11434")
ANALYSIS_MODEL = os.getenv("ANALYSIS_MODEL", "llama3.2:1b")
WORKSPACE_PATH = os.getenv("WORKSPACE_PATH", "/workspace")
RESULTS_DIR    = pathlib.Path(os.getenv("RESULTS_DIR", "/results"))
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def _extract_json_object(text: str) -> dict | None:
    start = text.find("{")
    if start == -1:
        return None
    depth, in_str, escape = 0, False, False
    for i, ch in enumerate(text[start:], start):
        if escape:
            escape = False
            continue
        if ch == "\\" and in_str:
            escape = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def reproduce(github_url: str) -> ReproductionResult:
    slug = github_url.rstrip("/").split("/")[-1]
    out_path = RESULTS_DIR / f"{slug}_{int(time.time())}.json"

    repo_path, readme = clone_repo(github_url, WORKSPACE_PATH)
    agent_output, agent_events = run_openhands_agent(OPENHANDS_URL, readme, str(repo_path))

    partial = {
        "repo_url": github_url,
        "agent_output": agent_output,
    }
    out_path.write_text(json.dumps(partial, indent=2))
    print(f"\nPartial result saved → {out_path}")

    analysis = analyze_with_ollama(agent_output, OLLAMA_URL, ANALYSIS_MODEL)

    agent_json = _extract_json_object(agent_output) or {}
    result = ReproductionResult(
        repo_url           = github_url,
        agent_output       = agent_output,
        steps_completed    = agent_json.get("steps_completed", []),
        conversation_trace = agent_events,
        verdict            = analysis.get("verdict", "unknown"),
        error_type         = analysis.get("error_type"),
        metrics_found      = analysis.get("metrics_found", {}),
        analysis           = analysis.get("explanation", ""),
    )

    out_path.write_text(result.model_dump_json(indent=2))
    print(f"Result finalised → {out_path}")
    print(f"Verdict: {result.verdict}")
    print(f"Analysis: {result.analysis}")
    return result


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        urls = sys.argv[1:]
    elif os.environ.get("GITHUB_REPOS"):
        urls = [u.strip() for u in os.environ["GITHUB_REPOS"].split(",") if u.strip()]
    elif os.environ.get("GITHUB_REPO"):
        urls = [os.environ["GITHUB_REPO"]]
    elif REPOS:
        urls = REPOS
    else:
        print("No repos specified.")
        print("Usage: python main.py <url1> [url2] ...")
        print("       or set GITHUB_REPOS=url1,url2 in the environment")
        print("       or edit the REPOS list at the top of main.py")
        sys.exit(1)

    workspace = pathlib.Path(WORKSPACE_PATH)
    subprocess.run(["find", str(workspace), "-mindepth", "1", "-delete"], check=True)
    subprocess.run(["find", str(RESULTS_DIR), "-mindepth", "1", "-delete"], check=True)

    print(f"\n{'='*60}")
    print(f"Running {len(urls)} repo(s)")
    print(f"{'='*60}\n")

    results = []
    for i, url in enumerate(urls, 1):
        print(f"\n[Repo {i}/{len(urls)}] {url}")
        print("-" * 60)
        try:
            results.append(reproduce(url))
        except Exception as exc:
            print(f"  ERROR: {exc}")
            results.append({"repo_url": url, "verdict": "error", "analysis": str(exc)})

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for r in results:
        if isinstance(r, ReproductionResult):
            print(f"  {r.repo_url}")
            print(f"    verdict   : {r.verdict}")
            print(f"    error_type: {r.error_type}")
            print(f"    analysis  : {r.analysis}")
        else:
            print(f"  {r['repo_url']}")
            print(f"    verdict   : {r['verdict']}")
            print(f"    analysis  : {r['analysis']}")
        print()

    summary_path = RESULTS_DIR / f"summary_{int(time.time())}.json"
    summary_path.write_text(
        json.dumps(
            [r.model_dump() if isinstance(r, ReproductionResult) else r for r in results],
            indent=2,
        )
    )
    print(f"Full summary saved → {summary_path}")
