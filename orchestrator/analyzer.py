import json
import ollama


ANALYSIS_PROMPT = """\
You are analyzing the output of an automated attempt to reproduce a scientific software artifact.
The agent was given the repository's README as its instructions and tried to run the experiment as described.

=== README (the instructions the agent was given) ===
{readme}
=== END README ===

=== Agent activity log ===
{agent_output}
=== END Agent activity log ===

{quality_context}

Your task is to produce a thorough, open-ended analysis of what happened during this reproduction attempt.
Do NOT use any predefined verdict taxonomy. Instead, reason from scratch about what you observe.

CRITICAL — grounding rule:
Every claim you make about the outcome must be traceable to specific content in the agent activity log above
(a command, a command's output, an error message, or an explicit statement the agent made). Do NOT infer
success or failure from the agent's tone, from a generic closing remark (e.g. "All done!", "Task complete",
"Let me know if you need anything else"), or from the mere absence of visible errors. A short or upbeat-sounding
final message is NOT evidence of success on its own — commands executed via a pipe (e.g. `| tail`) can mask
non-zero exit codes and real tracebacks, so judge success or failure from the actual command output shown,
not from exit codes or sign-off language.

If the agent activity log does not contain enough concrete detail (specific commands, outputs, error messages,
or file/data references) to determine what actually happened, do NOT guess or fill in a plausible-sounding
narrative. Instead, explicitly say so: set "outcome" to "insufficient log detail to determine outcome" and
explain in "failure_reasons" that the log lacked the evidence needed to assess the run.

Please respond with a JSON object only (no markdown, no preamble):
{{
  "outcome": "A short, free-form label you devise yourself that best captures the result (e.g. 'training script executed successfully', 'dependency installation failed on numpy version conflict', 'experiment timed out mid-epoch', 'agent looped without progress', 'insufficient log detail to determine outcome'). Be specific — avoid generic words like 'failed' or 'success' on their own.",
  "evidence_quote": "A short, verbatim excerpt (one or two lines) copied directly from the agent activity log above — a command output, error message, traceback, or explicit statement by the agent — that directly supports the 'outcome' you gave. This must be an exact quote from the log, not a paraphrase. If the log does not contain any excerpt that supports a specific outcome, write null.",
  "metrics_found": {{}},
  "steps_completed": "Detailed description of every step the agent successfully completed — installation, data download, preprocessing, script execution, etc. Be specific about what worked. Base this only on steps that are explicitly visible in the log, not on assumptions about what 'should' have happened.",
  "code_modifications": "Describe any changes or workarounds the agent applied to the repository code (e.g. patched a version pin, commented out a GPU assertion, reduced batch size). Write 'none' if no modifications were made, or 'unknown' if the log does not show enough detail to tell.",
  "readme_adherence": "Carefully compare what the README instructed with what the agent actually did. Did the agent follow the steps in order? Did it skip any steps? Did it take a different path? Did it repair something broken in the repo? Did it misinterpret any instructions? Be specific and quote relevant parts of both the README and the agent log. If the log doesn't show enough of the agent's actions to compare against the README, say so explicitly rather than guessing.",
  "failure_reasons": "If the experiment did not complete successfully, describe in your own words — without using predefined categories — exactly what went wrong and why, quoting the specific error or symptom from the log. Trace the root cause as specifically as possible (e.g. 'The setup.py required torch==1.9 but the environment had 2.1; pip could not downgrade due to conflicting transitive deps'). If the log lacks enough detail to identify a root cause, say that explicitly instead of speculating. Write null only if the experiment appears to have completed successfully AND the log contains direct evidence of that success.",
  "success_factors": "If the experiment completed or was progressing well, describe what contributed to that outcome, citing specific evidence from the log. Write null if the experiment did not succeed, or if success cannot be confirmed from the log.",
  "detailed_description": "Write a comprehensive narrative (at least 5–8 sentences) of the entire run from start to finish, based strictly on what is visible in the log. Describe what the agent did at each stage, what obstacles it encountered, how it handled them, and what the final state was. If the log cuts off, is truncated, or ends abruptly without showing the actual result of the final steps, state that explicitly rather than assuming an outcome. This section will be read by a human researcher who wants to understand what actually happened without having to read the raw log."
}}
"""


def analyze_with_ollama(agent_output: str, ollama_url: str, model: str,
                        readme: str = "",
                        agent_quality: dict | None = None,
                        prereq_missing: list[str] | None = None) -> dict:
    print("[3/3] Analyzing result with Ollama...")

    # Build quality context for the LLM
    quality_context = _build_quality_context(agent_quality, prereq_missing)

    # If agent quality is clearly "none" (empty response, no actions),
    # short-circuit — no point calling the LLM on an empty log
    if agent_quality and agent_quality.get("quality") == "none":
        return {
            "outcome": "agent produced no actions",
            "metrics_found": {},
            "steps_completed": "none",
            "code_modifications": "none",
            "readme_adherence": "The agent never meaningfully started, so no README steps were followed.",
            "failure_reasons": "The agent produced no tool calls or commands — likely a model incompatibility or empty response. The repository itself was never assessed.",
            "success_factors": None,
            "detailed_description": "The agent session started but the model produced no actions whatsoever. This may indicate an empty LLM response, a tool-calling incompatibility, or the agent getting stuck before issuing any commands. No conclusions about the repository's reproducibility can be drawn from this run.",
        }

    client = ollama.Client(host=ollama_url)
    response = client.chat(
        model=model,
        messages=[{"role": "user", "content": ANALYSIS_PROMPT.format(
            readme=readme[:3000] if readme else "(README not available)",
            agent_output=agent_output[:6000],
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
        return {
            "outcome": "analysis parse error",
            "metrics_found": {},
            "steps_completed": "unknown",
            "code_modifications": "unknown",
            "readme_adherence": "Could not parse LLM response.",
            "failure_reasons": None,
            "success_factors": None,
            "detailed_description": raw,
        }


def _build_quality_context(agent_quality: dict | None,
                           prereq_missing: list[str] | None = None) -> str:
    """Build a context string describing the agent's run quality for the LLM."""
    lines = []

    if prereq_missing:
        lines.append(f"Note: the agent explicitly reported missing prerequisites: {', '.join(prereq_missing)}.")

    if agent_quality:
        lines.append("Agent run quality metrics:")
        q = agent_quality
        lines.append(f"- Meaningful commands executed: {q['meaningful_commands']}")
        lines.append(f"- Tool/validation errors encountered: {q['total_errors']}")
        if q["agent_stuck"]:
            lines.append("- WARNING: Agent got stuck in a loop (AgentStuckInLoopError)")
        if q["empty_response"]:
            lines.append("- WARNING: Agent produced NO actions at all (empty response)")
        if agent_quality.get("quality") == "poor":
            lines.append("- Note: Agent had significant technical problems — consider how much of the log reflects genuine reproduction attempts vs. agent malfunction.")

    return "\n".join(lines)
