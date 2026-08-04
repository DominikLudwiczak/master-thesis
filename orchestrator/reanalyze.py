"""
Re-analyze existing result files using the fixed _summarize_events
(which now includes command outputs) and re-run Ollama analysis.

Usage:
    python reanalyze.py                          # re-analyze all results
    python reanalyze.py results/file.json        # re-analyze specific file
    python reanalyze.py --dry-run                # show what would be re-analyzed
"""

import sys
import os
import json
import pathlib
import time

# Add orchestrator dir to path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent import _summarize_events, _build_final_output, classify_agent_run, classify_timeout
from analyzer import analyze_with_ollama
from models import ReproductionResult

RESULTS_DIR = pathlib.Path(os.getenv("RESULTS_DIR", "/results"))
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
ANALYSIS_MODEL = os.getenv("ANALYSIS_MODEL", "llama3.2:1b")


def find_result_files(specific_file: str | None = None) -> list[pathlib.Path]:
    """Find result files to re-analyze."""
    if specific_file:
        p = pathlib.Path(specific_file)
        if not p.exists():
            # Try relative to RESULTS_DIR
            p = RESULTS_DIR / specific_file
        if not p.exists():
            print(f"File not found: {specific_file}")
            return []
        return [p]

    files = []
    for f in sorted(RESULTS_DIR.glob("*.json")):
        if f.name.startswith("summary_"):
            continue
        try:
            data = json.loads(f.read_text())
        except Exception:
            continue
        # Only re-analyze files that have conversation_trace
        if "conversation_trace" in data and isinstance(data["conversation_trace"], list):
            files.append(f)
    return files


def extract_readme_from_events(events: list[dict]) -> str:
    """Extract the README from the initial task message in conversation events.

    The agent prompt embeds the README between 'README:\\n---\\n' and '\\n---' markers.
    """
    for ev in events[:5]:
        content = (ev.get("message", "") or
                   ev.get("args", {}).get("content", "") if isinstance(ev.get("args"), dict) else "" or
                   ev.get("content", ""))
        if "README:" in content:
            idx = content.find("README:\n---\n")
            if idx >= 0:
                readme_start = idx + len("README:\n---\n")
                # Find the closing ---
                end_idx = content.rfind("\n---")
                if end_idx > readme_start:
                    return content[readme_start:end_idx]
                return content[readme_start:]
    return ""


def load_readme_from_cloner(url: str) -> str:
    """Try to extract README from workspace if available."""
    slug = url.rstrip("/").split("/")[-1].replace(".git", "")
    workspace = pathlib.Path(os.getenv("WORKSPACE_PATH", "/workspace"))
    repo_path = workspace / slug
    for readme_name in ["README.md", "readme.md", "README.rst", "README.txt", "README"]:
        readme_file = repo_path / readme_name
        if readme_file.exists():
            return readme_file.read_text()
    return ""


def reanalyze_file(filepath: pathlib.Path, dry_run: bool = False) -> bool:
    """Re-analyze a single result file. Returns True if successful."""
    try:
        data = json.loads(filepath.read_text())
    except Exception as e:
        print(f"  ERROR reading {filepath}: {e}")
        return False

    url = data.get("repo_url", "unknown")
    events = data.get("conversation_trace", [])
    old_outcome = data.get("outcome", "unknown")

    if not events:
        print(f"  SKIP {filepath.name}: no conversation_trace")
        return False

    print(f"\n{'='*60}")
    print(f"Re-analyzing: {filepath.name}")
    print(f"  URL: {url}")
    print(f"  Events: {len(events)}")
    print(f"  Old outcome: {old_outcome}")

    if dry_run:
        # Show what the new summary looks like
        new_summary = _summarize_events(events)
        print(f"  New summary length: {len(new_summary)} chars")
        # Count lines with command output (lines not starting with $ or [)
        output_lines = [l for l in new_summary.splitlines()
                       if l.strip() and not l.startswith("$") and not l.startswith("[")]
        print(f"  Output lines in new summary: {len(output_lines)}")
        return True

    # Rebuild the agent output using the fixed _summarize_events
    new_agent_output = _build_final_output(events)

    # Re-classify agent quality
    agent_quality = classify_agent_run(events)
    print(f"  Agent quality: {agent_quality['quality']} "
          f"(commands={agent_quality['meaningful_commands']}, "
          f"unique={agent_quality['unique_commands']}, "
          f"repetition={agent_quality['repetition_ratio']:.0%})")

    # Try to get README — from events first (most reliable), then data, then workspace
    readme = extract_readme_from_events(events)
    if not readme and "readme" in data:
        readme = data["readme"]
    if not readme:
        readme = load_readme_from_cloner(url)
    if readme:
        print(f"  README found ({len(readme)} chars)")

    if not readme:
        print("  WARNING: No README available for re-analysis. "
              "Analysis quality may be reduced.")

    # Check if agent explicitly reported missing prerequisites
    from analyzer import _extract_json_object
    agent_json = _extract_json_object(new_agent_output) or {}
    prereq_info = None
    if agent_json.get("prereq_missing"):
        prereq_info = agent_json.get("missing_prereqs", [])

    # Detect timeout and classify
    timeout_info = None
    if new_agent_output.startswith("TIMEOUT") or data.get("timeout_info"):
        timeout_info = data.get("timeout_info") or classify_timeout(events)
        print(f"  Timeout type: {timeout_info['timeout_type']}, "
              f"phase: {timeout_info['phase_reached']}")

    # Re-run Ollama analysis
    analysis = analyze_with_ollama(
        new_agent_output, OLLAMA_URL, ANALYSIS_MODEL,
        readme=readme or "",
        agent_quality=agent_quality,
        prereq_missing=prereq_info,
        timeout_info=timeout_info,
    )

    # Normalize LLM response
    for key in ("steps_completed", "outcome", "code_modifications",
                "readme_adherence", "failure_reasons", "success_factors",
                "detailed_description", "dependency_pinning"):
        val = analysis.get(key)
        if isinstance(val, list):
            analysis[key] = "\n".join(str(item) for item in val)

    # Build updated result
    result = ReproductionResult(
        repo_url=url,
        agent_output=new_agent_output,
        steps_completed=analysis.get("steps_completed", ""),
        conversation_trace=events,
        outcome=analysis.get("outcome", "unknown"),
        metrics_found=analysis.get("metrics_found", {}),
        code_modifications=analysis.get("code_modifications", "none"),
        readme_adherence=analysis.get("readme_adherence", ""),
        failure_reasons=analysis.get("failure_reasons"),
        success_factors=analysis.get("success_factors"),
        detailed_description=analysis.get("detailed_description", ""),
        agent_quality=agent_quality,
        path_analysis=data.get("path_analysis"),
        timeout_info=timeout_info,
        dependency_pinning=analysis.get("dependency_pinning"),
    )

    # Save with _reanalyzed suffix
    new_path = filepath.with_name(filepath.stem + "_reanalyzed.json")
    new_path.write_text(result.model_dump_json(indent=2))

    print(f"  New outcome: {result.outcome}")
    print(f"  Description: {result.detailed_description[:200]}")
    print(f"  Saved → {new_path}")
    return True


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    args = [a for a in sys.argv[1:] if a != "--dry-run"]

    specific_file = args[0] if args else None
    files = find_result_files(specific_file)

    if not files:
        print("No result files found to re-analyze.")
        print(f"Looked in: {RESULTS_DIR}")
        sys.exit(1)

    print(f"Found {len(files)} file(s) to re-analyze"
          f"{' (dry run)' if dry_run else ''}")

    success = 0
    for f in files:
        if reanalyze_file(f, dry_run=dry_run):
            success += 1

    print(f"\n{'='*60}")
    print(f"Re-analyzed {success}/{len(files)} files successfully.")
