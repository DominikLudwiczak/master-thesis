import os
import time
import pathlib
import requests


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


def ensure_openhands_settings(openhands_url: str) -> None:
    agent_model = os.getenv("AGENT_MODEL", "qwen2.5-coder:14b")
    ollama_base = os.getenv("OLLAMA_URL", "http://ollama:11434")
    payload = {
        "llm_model":    f"ollama/{agent_model}",
        "llm_base_url": ollama_base,
        "llm_api_key":  "ollama",
    }
    r = requests.post(f"{openhands_url}/api/settings", json=payload, timeout=10)
    if r.status_code not in (200, 201):
        print(f"    [warn] settings POST returned {r.status_code}: {r.text[:300]}")
    else:
        print("    settings saved OK")


def run_openhands_agent(openhands_url: str, readme: str) -> str:
    readme_trimmed = _smart_truncate_readme(readme, max_chars=6000)
    task = AGENT_TASK_TEMPLATE.format(readme=readme_trimmed)

    print("[2/3] Sending task to OpenHands agent...")
    ensure_openhands_settings(openhands_url)

    resp = requests.post(
        f"{openhands_url}/api/conversations",
        json={"initial_user_msg": task},
        timeout=30,
    )
    if not resp.ok:
        raise RuntimeError(
            f"OpenHands POST /api/conversations failed {resp.status_code}: {resp.text[:300]}"
        )
    conversation_id = resp.json()["conversation_id"]
    print(f"    conversation id: {conversation_id}")

    deadline = time.time() + 1200
    while time.time() < deadline:
        time.sleep(10)
        status_resp = requests.get(
            f"{openhands_url}/api/conversations/{conversation_id}",
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


def _smart_truncate_readme(readme: str, max_chars: int) -> str:
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