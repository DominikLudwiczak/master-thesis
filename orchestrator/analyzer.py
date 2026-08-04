import json
import os
import re
import ollama


def _repair_json(text: str) -> str:
    """Attempt to fix unescaped double quotes within JSON string values.

    LLMs often embed code snippets containing unescaped quotes inside JSON
    string values (e.g. pivot(["pid", "vid"]) instead of escaped form).

    Strategy: use the formatted structure of LLM JSON output (each top-level
    key on its own line) to identify value boundaries, then escape all
    unescaped double quotes within string value regions.
    """
    # Known top-level keys from our analysis prompts
    known_keys = [
        'outcome', 'evidence_quote', 'metrics_found', 'steps_completed',
        'code_modifications', 'readme_adherence', 'failure_reasons',
        'success_factors', 'dependency_pinning', 'detailed_description',
        'env_fixes_applied', 'has_pinned_deps', 'pinning_mechanism',
        'unpinned_packages_installed', 'version_related_errors', 'details',
    ]
    key_alt = '|'.join(re.escape(k) for k in known_keys)
    # Match a line that starts a key: optional whitespace + "key" + optional whitespace + :
    key_line_re = re.compile(
        rf'^(\s*)"({key_alt})"\s*:', re.MULTILINE
    )
    matches = list(key_line_re.finditer(text))

    if len(matches) < 2:
        return text  # Can't determine structure, return as-is

    parts = []
    for idx, m in enumerate(matches):
        # Add text before this key (or between previous value end and this key)
        if idx == 0:
            parts.append(text[:m.start()])
        # Add the key + colon literally
        parts.append(m.group(0))

        value_start = m.end()
        # Value region extends to the start of the next key line (or end of text)
        if idx + 1 < len(matches):
            value_end = matches[idx + 1].start()
        else:
            value_end = len(text)

        value_region = text[value_start:value_end]
        stripped = value_region.lstrip()

        if stripped.startswith('"'):
            # String value — find first and last structural quotes
            first_q = value_start + value_region.index('"')
            # Last quote: strip trailing whitespace/comma/newlines, then find last "
            rstripped = value_region.rstrip()
            if rstripped.endswith(','):
                rstripped = rstripped[:-1].rstrip()
            if rstripped.endswith('"'):
                last_q_offset = len(rstripped) - 1
                last_q = value_start + last_q_offset

                # Content between structural quotes needs escaping
                before = text[value_start:first_q + 1]       # whitespace + opening "
                content = text[first_q + 1:last_q]            # inner content
                after = text[last_q:value_end]                 # closing " + comma + newline

                # Escape unescaped quotes in content
                # Preserve already-escaped quotes
                content = content.replace('\\"', '\x00ESC_Q\x00')
                content = content.replace('"', '\\"')
                content = content.replace('\x00ESC_Q\x00', '\\"')

                parts.append(before + content + after)
            else:
                parts.append(value_region)
        else:
            parts.append(value_region)

    return ''.join(parts)


def _extract_json_object(text: str) -> dict | None:
    """Extract a JSON object from text using brace-depth tracking.

    More robust than json.loads for LLM output that may contain
    unescaped backticks, $-signs, or other characters in string values.
    """
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
                candidate = text[start:i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    # Try repairing unescaped quotes before giving up
                    try:
                        return json.loads(_repair_json(candidate))
                    except json.JSONDecodeError:
                        return None
    return None


README_PATH_ANALYSIS_PROMPT = """\
You are analyzing a scientific repository's README to identify all possible reproduction paths.

Our environment:
- Linux (Ubuntu), no GPU, Docker CLI available, no API tokens (HuggingFace, OpenAI, W&B etc.)
- Python 3.12, pip, conda available
- Internet access available for downloads

README:
---
{readme}
---

Identify ALL distinct paths to reproduce or verify this artifact. Examples of paths:
- Docker-based reproduction (if Dockerfile / docker-compose / docker run instructions exist)
- Manual install + run scripts
- Pre-computed results verification (just checking existing outputs)
- CPU-only path vs GPU path
- Simplified/demo mode
- Notebook execution

Respond with JSON only (no markdown, no preamble):
{{
    "paths": [
        {{
            "name": "short name for this path",
            "description": "what this path does",
            "prerequisites": ["list of requirements"],
            "commands_hint": "key commands mentioned in README for this path",
            "confidence": "high/medium/low - likelihood of working in our env"
        }}
    ],
    "recommended_order": ["path names in order of likelihood of success in our environment"],
    "blockers": ["any hard blockers that affect ALL paths"],
    "notes": "any additional observations relevant to reproduction"
}}
"""


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
  "dependency_pinning": "Analyze whether the repository had its dependency versions pinned or frozen. Specifically assess: (1) Does the repo contain a requirements.txt, environment.yml, setup.py, setup.cfg, pyproject.toml, Pipfile.lock, conda env file, or similar with EXPLICIT version pins (e.g. pandas==1.3.5)? (2) Or does it list bare package names without versions (e.g. just 'pandas')? (3) Does it use a Docker image with pre-installed frozen dependencies? (4) Did the agent end up installing latest/unpinned versions of packages? (5) Did any errors or failures appear to stem from version incompatibility caused by unpinned dependencies (e.g. removed APIs, changed behavior in newer versions)? Provide a structured assessment with fields: 'has_pinned_deps' (true/false/partial), 'pinning_mechanism' (e.g. 'requirements.txt with versions', 'Docker image', 'none', 'bare package list without versions'), 'unpinned_packages_installed' (list of packages installed without version pins, or null), 'version_related_errors' (true/false), 'details' (free-form explanation of what you observed).",
  "detailed_description": "Write a comprehensive narrative (at least 5–8 sentences) of the entire run from start to finish, based strictly on what is visible in the log. Describe what the agent did at each stage, what obstacles it encountered, how it handled them, and what the final state was. If the log cuts off, is truncated, or ends abruptly without showing the actual result of the final steps, state that explicitly rather than assuming an outcome. This section will be read by a human researcher who wants to understand what actually happened without having to read the raw log."
}}
"""


def analyze_readme_paths(readme: str, ollama_url: str, model: str) -> dict:
    """Pre-analyze a README to identify reproduction paths and fallback order."""
    if not readme:
        return {"paths": [], "recommended_order": [], "blockers": [], "notes": "No README available"}

    readme_max = int(os.getenv("PATH_ANALYSIS_README_MAX_CHARS", "15000"))
    client = ollama.Client(host=ollama_url, timeout=1800)
    try:
        response = client.chat(
            model=model,
            messages=[{"role": "user", "content": README_PATH_ANALYSIS_PROMPT.format(
                readme=readme[:readme_max],
            )}],
        )
    except Exception as e:
        print(f"    Path analysis error: {e}")
        return {"paths": [], "recommended_order": [], "blockers": [], "notes": f"Pre-analysis failed: {e}"}

    raw = response["message"]["content"].strip()
    if "<think>" in raw:
        think_end = raw.rfind("</think>")
        if think_end != -1:
            raw = raw[think_end + len("</think>"):].strip()
    raw = raw.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        parsed = _extract_json_object(raw)
        if parsed:
            return parsed
        return {"paths": [], "recommended_order": [], "blockers": [], "notes": "Could not parse path analysis"}


def analyze_with_ollama(agent_output: str, ollama_url: str, model: str,
                        readme: str = "",
                        agent_quality: dict | None = None,
                        prereq_missing: list[str] | None = None,
                        timeout_info: dict | None = None) -> dict:
    print("[3/3] Analyzing result with Ollama...")

    # Build quality context for the LLM
    quality_context = _build_quality_context(agent_quality, prereq_missing, timeout_info)

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

    readme_max = int(os.getenv("ANALYZER_README_MAX_CHARS", "15000"))
    output_max = int(os.getenv("ANALYZER_OUTPUT_MAX_CHARS", "20000"))

    client = ollama.Client(host=ollama_url, timeout=1800)
    try:
        response = client.chat(
            model=model,
            messages=[{"role": "user", "content": ANALYSIS_PROMPT.format(
                readme=readme[:readme_max] if readme else "(README not available)",
                agent_output=agent_output[:output_max],
                quality_context=quality_context,
            )}],
        )
    except Exception as e:
        print(f"    Ollama error: {e}")
        return {
            "outcome": "analysis error",
            "metrics_found": {},
            "steps_completed": agent_output[:500],
            "code_modifications": "unknown",
            "readme_adherence": "unknown",
            "failure_reasons": f"Ollama analysis failed: {e}",
            "success_factors": None,
            "detailed_description": f"Analysis could not be completed due to Ollama error: {e}. Agent output was: {agent_output[:1000]}",
        }
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
        # Fallback 1: extract JSON using brace-depth tracking
        parsed = _extract_json_object(raw)
        if parsed:
            return parsed
        # Fallback 2: repair unescaped quotes then parse
        try:
            repaired = _repair_json(raw)
            parsed = json.loads(repaired)
            if parsed:
                return parsed
        except (json.JSONDecodeError, Exception):
            pass
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
                           prereq_missing: list[str] | None = None,
                           timeout_info: dict | None = None) -> str:
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

    if timeout_info:
        lines.append("\nTimeout analysis:")
        lines.append(f"- Timeout type: {timeout_info['timeout_type']}")
        phase_names = {1: "Reconnaissance", 2: "Install Dependencies",
                       3: "Prepare Data", 4: "Run Experiment", 5: "Report"}
        phase = timeout_info.get("phase_reached", 0)
        lines.append(f"- Highest phase reached: {phase} ({phase_names.get(phase, 'Unknown')})")
        if timeout_info.get("experiment_script_detected"):
            lines.append(f"- Main experiment script: {timeout_info['experiment_script_detected']}")
        if timeout_info.get("experiment_was_running"):
            lines.append("- IMPORTANT: The experiment script was actively running when "
                         "the timeout occurred. This suggests the artifact may be "
                         "reproducible but requires more execution time.")
        if timeout_info.get("last_command") and not timeout_info.get("last_command_had_output"):
            cmd = timeout_info["last_command"][:150]
            lines.append(f"- Last command (still running at timeout): {cmd}")
            if timeout_info.get("last_command_running_seconds"):
                lines.append(f"  (had been running for {timeout_info['last_command_running_seconds']}s)")
        lines.append(f"- Progress summary: {timeout_info.get('progress_summary', 'N/A')}")
        lines.append("")
        lines.append("Timeout interpretation guidance:")
        lines.append("- 'timeout_during_experiment' with experiment script still running is a POSITIVE signal — "
                     "the outcome should reflect that the artifact was progressing toward reproduction but needed "
                     "more time, not that it 'failed'.")
        lines.append("- 'timeout_during_setup' suggests complex dependencies but the experiment was never tested.")
        lines.append("- 'timeout_agent_idle' suggests the agent got stuck; the timeout is not informative about "
                     "the artifact itself.")

    return "\n".join(lines)
