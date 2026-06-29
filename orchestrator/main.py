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
import shutil
import time
import pathlib
import subprocess

from models import ReproductionResult
from cloner import clone_repo
from agent import run_openhands_agent, classify_agent_run, stop_conversation
from analyzer import analyze_with_ollama

REPOS = [
    "https://github.com/SKKU-SecLab/SmartMark",
    "https://github.com/coinse/fonte",
    "https://github.com/SageSELab/AidUI",
    "https://github.com/soarsmu/Chronos",
    "https://zenodo.org/records/7536375#.Y8JfSuxBwUE",
    "https://github.com/UsmanGohar/FairEnsemble",
    "https://zenodo.org/records/7566398?preview_file=ExploratoryCaseStudySpecs.zip",
    "https://github.com/SageSELab/AidUI",
    "https://github.com/jspaper22/bftdetector",
    "https://zenodo.org/records/7622528",
    "https://github.com/Generative-Program-Analysis/icse23-artifact-evaluation",
    "https://github.com/ucd-plse/On-the-Reproducibility",
]

OPENHANDS_URL  = os.getenv("OPENHANDS_URL", "http://localhost:3000")
OLLAMA_URL     = os.getenv("OLLAMA_URL",    "http://localhost:11434")
ANALYSIS_MODEL = os.getenv("ANALYSIS_MODEL", "llama3.2:1b")
WORKSPACE_PATH = os.getenv("WORKSPACE_PATH", "/workspace")
RESULTS_DIR    = pathlib.Path(os.getenv("RESULTS_DIR", "/results"))
RESUME         = os.getenv("RESUME", "false").lower() in ("1", "true", "yes")
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


def find_completed_result(url: str) -> pathlib.Path | None:
    """Return path to a completed result file for this URL, or None.

    A completed result has a 'verdict' field (the full ReproductionResult),
    as opposed to a partial file (written after agent run but before analysis).
    """
    for f in sorted(RESULTS_DIR.glob("*.json")):
        if f.name.startswith("summary_"):
            continue
        try:
            data = json.loads(f.read_text())
        except Exception:
            continue
        if data.get("repo_url") == url and "outcome" in data:
            return f
    return None


def reproduce(url: str) -> ReproductionResult:
    slug = url.rstrip("/").split("/")[-1].replace(".git", "")
    out_path = RESULTS_DIR / f"{slug}_{int(time.time())}.json"

    repo_path, readme = clone_repo(url, WORKSPACE_PATH)
    agent_output, agent_events, conversation_id = run_openhands_agent(
        OPENHANDS_URL, readme, str(repo_path)
    )

    # Stop the OpenHands conversation before cleaning up the workspace.
    # The runtime container stays alive after our polling loop exits, so
    # deleting the repo dir while it's still mounted would pull the rug
    # out from under the agent.
    stop_conversation(OPENHANDS_URL, conversation_id)

    agent_quality = classify_agent_run(agent_events)
    print(f"    Agent quality: {agent_quality['quality']} "
          f"(commands={agent_quality['meaningful_commands']}, "
          f"unique={agent_quality['unique_commands']}, "
          f"repetition={agent_quality['repetition_ratio']:.0%}, "
          f"errors={agent_quality['total_errors']}, "
          f"stuck={agent_quality['agent_stuck']})")

    partial = {
        "repo_url": url,
        "agent_output": agent_output,
        "agent_quality": agent_quality,
    }
    out_path.write_text(json.dumps(partial, indent=2))
    print(f"\nPartial result saved → {out_path}")

    agent_json = _extract_json_object(agent_output) or {}

    # If agent explicitly reported missing prerequisites, pass this info
    prereq_info = None
    if agent_json.get("prereq_missing"):
        prereq_info = agent_json.get("missing_prereqs", [])

    analysis = analyze_with_ollama(agent_output, OLLAMA_URL, ANALYSIS_MODEL,
                                   readme=readme or "",
                                   agent_quality=agent_quality,
                                   prereq_missing=prereq_info)
    result = ReproductionResult(
        repo_url             = url,
        agent_output         = agent_output,
        steps_completed      = analysis.get("steps_completed", ""),
        conversation_trace   = agent_events,
        outcome              = analysis.get("outcome", "unknown"),
        metrics_found        = analysis.get("metrics_found", {}),
        code_modifications   = analysis.get("code_modifications", "none"),
        readme_adherence     = analysis.get("readme_adherence", ""),
        failure_reasons      = analysis.get("failure_reasons"),
        success_factors      = analysis.get("success_factors"),
        detailed_description = analysis.get("detailed_description", ""),
        agent_quality        = agent_quality,
    )

    out_path.write_text(result.model_dump_json(indent=2))
    print(f"Result finalised → {out_path}")
    print(f"Outcome: {result.outcome}")
    print(f"Description: {result.detailed_description[:200]}")

    # Clean up workspace AFTER analysis is complete and conversation is stopped
    shutil.rmtree(repo_path, ignore_errors=True)
    print(f"    Removed repo dir {repo_path}")

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

    if RESUME:
        print("Resume mode: skipping workspace/results cleanup.")
    else:
        subprocess.run(["find", str(workspace), "-mindepth", "1", "-delete"], check=True)
        subprocess.run(["find", str(RESULTS_DIR), "-mindepth", "1", "-delete"], check=True)

    print(f"\n{'='*60}")
    print(f"Running {len(urls)} repo(s){' (resume mode)' if RESUME else ''}")
    print(f"{'='*60}\n")

    results = []
    for i, url in enumerate(urls, 1):
        print(f"\n[Repo {i}/{len(urls)}] {url}")
        print("-" * 60)

        if RESUME:
            completed = find_completed_result(url)
            if completed:
                print(f"  Already completed, loading result from {completed.name}")
                try:
                    results.append(ReproductionResult.model_validate_json(completed.read_text()))
                except Exception as exc:
                    print(f"  WARNING: could not load existing result ({exc}), re-running.")
                else:
                    continue

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
            print(f"    outcome   : {r.outcome}")
            print(f"    description: {r.detailed_description[:200]}...")
        else:
            print(f"  {r['repo_url']}")
            print(f"    outcome   : {r.get('outcome', r.get('verdict', 'unknown'))}")
            print(f"    description: {r.get('detailed_description', r.get('analysis', ''))[:200]}...")
        print()

    summary_path = RESULTS_DIR / f"summary_{int(time.time())}.json"
    summary_path.write_text(
        json.dumps(
            [r.model_dump() if isinstance(r, ReproductionResult) else r for r in results],
            indent=2,
        )
    )
    print(f"Full summary saved → {summary_path}")
