import json
import ollama


ANALYSIS_PROMPT = """\
You are analyzing the output of an automated attempt to reproduce a scientific software artifact.
The goal was to follow the repository's README instructions and run the experiment as described.

Agent activity log:
---
{agent_output}
---

{quality_context}

Based on the log above, classify the reproduction attempt.
Please respond with a JSON object only (no markdown, no preamble):
{{
  "verdict": "reproduced" | "timeout_running" | "partial" | "failed" | "inconclusive" | "prereq_missing",
  "error_type": null | "env" | "code" | "data" | "timeout" | "agent" | "prereq",
  "metrics_found": {{}},
  "steps_completed": "list what was successfully done (e.g. 'cloned repo, installed dependencies, prepared data')",
  "code_modifications": "list any code changes or workarounds the agent had to apply to the project (e.g. 'changed batch_size to 1, commented out GPU check'). Write 'none' if no modifications were made.",
  "failure_reason": "if not reproduced — describe the specific error or blocker that prevented completion. Write null if reproduced or timeout_running.",
  "explanation": "2-3 sentence summary of what happened and why it succeeded or failed"
}}

verdict meanings:
- reproduced: experiment ran to completion and produced plausible results or metrics
- timeout_running: the experiment was successfully launched and was ACTIVELY RUNNING without errors when the time limit expired. Dependencies were installed, data was prepared, and the main experiment script was executing. This is a POSITIVE outcome — the artifact appears reproducible but our time budget was insufficient. This is NOT a failure of the artifact.
- partial: some steps succeeded but others genuinely FAILED — e.g. dependencies installed but the experiment script crashed, or 3 out of 5 sub-experiments ran but 2 errored out. Something actually went wrong that prevented completion.
- failed: the agent genuinely tried to run the experiment (installed dependencies, ran scripts) but the experiment could not be reproduced due to issues in the repository, environment, or data
- prereq_missing: the experiment requires prerequisites that our environment does not have, such as: GPU/CUDA hardware, API tokens (Hugging Face, Weights & Biases, OpenAI), proprietary datasets with restricted access, commercial software licenses. The artifact might be perfectly reproducible with proper prerequisites — we cannot judge it.
- inconclusive: the agent itself malfunctioned — it got stuck in a loop, produced empty responses, could not use its tools, or never meaningfully attempted the experiment. We CANNOT judge the repository's reproducibility from this run.

CRITICAL CLASSIFICATION RULES:
1. Use "inconclusive" when the failure is clearly the agent's fault (tool errors, empty responses, stuck in loop, never ran any experiment commands).
2. Use "failed" only when the agent genuinely attempted the reproduction and the experiment itself has a real bug or broken setup.
3. Use "prereq_missing" when the experiment needs something our environment doesn't provide: GPU, API tokens, restricted data access, specific hardware. This is NOT a failure of the artifact — it's a limitation of our test environment.
4. Use "partial" when some steps FAILED with actual errors — NOT when the agent simply ran out of time while things were working.
5. Use "timeout_running" when the log starts with TIMEOUT AND the last experiment command was actively running or had just completed without errors. The key distinction: if the experiment was executing normally and time ran out, that is "timeout_running" (positive). If something crashed or errored before timeout, that is "partial" or "failed".
6. If the log starts with TIMEOUT but the agent never got past setup (installing deps, downloading data), use "partial" — the experiment itself never started running.

error_type meanings:
- env: missing dependency, wrong Python/Node version, OS incompatibility, package install failure
- code: bug in the repository code itself
- data: missing dataset or download failed (publicly available data that should be accessible)
- timeout: ran out of time
- agent: the AI agent itself malfunctioned (stuck, empty response, tool calling failure)
- prereq: requires GPU, API tokens, credentials, restricted data access, or commercial software that our environment lacks
"""


def analyze_with_ollama(agent_output: str, ollama_url: str, model: str,
                        agent_quality: dict | None = None,
                        prereq_missing: list[str] | None = None) -> dict:
    print("[3/3] Analyzing result with Ollama...")

    # If agent explicitly reported missing prerequisites, short-circuit
    if prereq_missing:
        return {
            "verdict": "prereq_missing",
            "error_type": "prereq",
            "metrics_found": {},
            "explanation": f"Agent identified missing prerequisites: {', '.join(prereq_missing)}. "
                           "The artifact may be reproducible with proper infrastructure.",
        }

    if agent_output.startswith("TIMEOUT"):
        # Timeout with no meaningful work is inconclusive
        if agent_quality and agent_quality.get("quality") == "none":
            return {
                "verdict": "inconclusive",
                "error_type": "agent",
                "metrics_found": {},
                "explanation": "Agent timed out without executing any commands. Cannot assess repository reproducibility.",
            }
        # Agent did meaningful work but ran out of time — let the LLM
        # decide if it's partial, prereq_missing, etc. based on what
        # was accomplished.  Fall through to the LLM analysis below.

    # Build quality context for the LLM
    quality_context = _build_quality_context(agent_quality)

    # If agent quality is clearly "none" (empty response, no actions),
    # short-circuit to inconclusive without LLM call
    if agent_quality and agent_quality.get("quality") == "none":
        return {
            "verdict": "inconclusive",
            "error_type": "agent",
            "metrics_found": {},
            "explanation": "Agent produced no actions — likely a model incompatibility (empty response, no tool calls). Cannot assess repository reproducibility.",
        }

    client = ollama.Client(host=ollama_url)
    response = client.chat(
        model=model,
        messages=[{"role": "user", "content": ANALYSIS_PROMPT.format(
            agent_output=agent_output[:4000],
            quality_context=quality_context,
        )}],
    )
    raw = response["message"]["content"].strip()
    # Strip thinking tags if present (qwen3 models)
    if "<think>" in raw:
        think_end = raw.rfind("</think>")
        if think_end != -1:
            raw = raw[think_end + len("</think>"):].strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"verdict": "unknown", "error_type": None, "metrics_found": {}, "explanation": raw}


def _build_quality_context(agent_quality: dict | None) -> str:
    """Build a context string describing the agent's run quality for the LLM."""
    if not agent_quality:
        return ""

    lines = ["Agent run quality assessment:"]
    q = agent_quality
    lines.append(f"- Meaningful commands executed: {q['meaningful_commands']}")
    lines.append(f"- Tool/validation errors encountered: {q['total_errors']}")

    if q["agent_stuck"]:
        lines.append("- WARNING: Agent got stuck in a loop (AgentStuckInLoopError)")
    if q["empty_response"]:
        lines.append("- WARNING: Agent produced NO actions at all (empty response)")

    if q["quality"] == "none":
        lines.append("- CONCLUSION: Agent never meaningfully attempted the experiment. This should be 'inconclusive'.")
    elif q["quality"] == "poor":
        lines.append("- CONCLUSION: Agent had significant technical problems. Consider 'inconclusive' if the agent never got past basic setup.")
    else:
        lines.append("- CONCLUSION: Agent made a genuine attempt at reproduction.")

    return "\n".join(lines)
