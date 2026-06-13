import json
import ollama


ANALYSIS_PROMPT = """\
You are analyzing the output of an automated attempt to reproduce a scientific software artifact.
The goal was to follow the repository's README instructions and run the experiment as described.

Agent activity log:
---
{agent_output}
---

Based on the log above, classify the reproduction attempt.
Please respond with a JSON object only (no markdown, no preamble):
{{
  "verdict": "reproduced" | "partial" | "failed",
  "error_type": null | "env" | "code" | "data" | "timeout",
  "metrics_found": {{}},
  "explanation": "2-3 sentence summary of what happened and why it succeeded or failed"
}}

verdict meanings:
- reproduced: experiment ran to completion and produced plausible results or metrics
- partial: some steps succeeded (e.g. dependencies installed, some scripts ran) but the final experiment result was not obtained
- failed: could not run the experiment at all (setup crashed, critical dependencies missing, agent stuck)

error_type meanings (if verdict != reproduced):
- env: missing dependency, wrong Python/Node version, OS incompatibility, package install failure
- code: bug in the repository code itself
- data: missing dataset or download failed
- timeout: ran out of time
"""


def analyze_with_ollama(agent_output: str, ollama_url: str, model: str) -> dict:
    print("[3/3] Analyzing result with Ollama...")

    if agent_output.startswith("TIMEOUT"):
        return {
            "verdict": "failed",
            "error_type": "timeout",
            "metrics_found": {},
            "explanation": "Agent did not finish within the time limit. No experiment output to analyze.",
        }

    client = ollama.Client(host=ollama_url)
    response = client.chat(
        model=model,
        messages=[{"role": "user", "content": ANALYSIS_PROMPT.format(agent_output=agent_output[:4000])}],
    )
    raw = response["message"]["content"].strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"verdict": "unknown", "error_type": None, "metrics_found": {}, "explanation": raw}
