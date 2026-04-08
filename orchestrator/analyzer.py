import json
import ollama


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


def analyze_with_ollama(agent_output: str, ollama_url: str, model: str) -> dict:
    print("[3/3] Analyzing result with Ollama...")
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