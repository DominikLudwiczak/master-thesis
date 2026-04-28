import os
import time
import pathlib
import requests


AGENT_TASK_TEMPLATE = """\
You are reproducing a scientific experiment from a GitHub repository.

The repository has been cloned to {repo_path}.
All your work must be done inside that directory — start by running: cd {repo_path}

Your goal:
1. cd into {repo_path} first.
2. Read the README carefully and identify the reproduction steps.
3. Install all dependencies.
4. Run the experiment exactly as described.
5. Capture ALL output (stdout, stderr, metrics, figures saved, etc.).
6. If a step fails, diagnose why and try to fix it (once).
7. At the end, output a JSON block like this:

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
    agent_model = os.getenv("AGENT_MODEL", "qwen2.5-coder:3b")
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


def run_openhands_agent(openhands_url: str, readme: str, repo_path: str) -> tuple[str, list[dict]]:
    readme_trimmed = _smart_truncate_readme(readme, max_chars=6000)
    task = AGENT_TASK_TEMPLATE.format(repo_path=repo_path, readme=readme_trimmed)

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

    all_events: list[dict] = []
    last_event_id = -1
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
        runtime_status = data.get("runtime_status", "")
        print(f"[debug] status={state!r} runtime_status={runtime_status!r} events={len(all_events)}")

        new_events = _fetch_new_events(openhands_url, conversation_id, last_event_id)
        for event in new_events:
            all_events.append(event)
            last_event_id = max(last_event_id, event.get("id", last_event_id))
            description = _describe_event(event)
            if description:
                print(f"  {description}")

        if state.lower() in ("finished", "error", "stopped", "awaiting_user_input"):
            final_message = _last_assistant_message(all_events) or str(data)
            return final_message, all_events

    return "TIMEOUT: agent did not finish within 20 minutes", all_events


def _fetch_new_events(openhands_url: str, conversation_id: str, last_id: int) -> list[dict]:
    """Fetch only events with id > last_id."""
    try:
        resp = requests.get(
            f"{openhands_url}/api/conversations/{conversation_id}/events",
            params={"start_id": last_id + 1, "limit": 100},
            timeout=60,
        )
        if resp.ok:
            data = resp.json()
            events = data if isinstance(data, list) else data.get("events", [])
            return [e for e in events if e.get("id", -1) > last_id]
    except Exception as e:
        print(f"    [warn] could not fetch events: {e}")
    return []


def _describe_event(event: dict) -> str | None:
    """Return a human-readable one-liner for an event, or None to skip it."""
    source = event.get("source", "")
    action = event.get("action", "")
    observation = event.get("observation", "")
    extras = event.get("extras", {})
    content = event.get("content", "")
    message = event.get("message", "")
    args = event.get("args", {}) if isinstance(event.get("args"), dict) else {}

    # Agent state transitions
    if observation == "agent_state_changed":
        state = extras.get("agent_state", "")
        reason = extras.get("reason", "")
        return f"[state] {state}" + (f" — {reason}" if reason else "")

    if action == "change_agent_state":
        state = extras.get("agent_state", "")
        return f"[state] → {state}"

    # Skip the verbose system prompt injected at conversation start
    if source == "agent" and action == "system":
        return None

    # The user task message
    if source == "user" and action == "message":
        text = message or args.get("content", "")
        preview = text[:120].replace("\n", " ")
        return f"[task] {preview}{'...' if len(text) > 120 else ''}"

    # Agent prose response
    if source == "agent" and action == "message":
        text = message or content
        if text:
            preview = text[:200].replace("\n", " ")
            return f"[agent] {preview}{'...' if len(text) > 200 else ''}"

    # Bash command issued by the agent
    if source == "agent" and action == "run":
        cmd = args.get("command", "")
        return f"[bash] {cmd[:200]}"

    # Bash command output from the environment
    if source == "environment" and observation == "run":
        if content:
            preview = content[:200].replace("\n", " ")
            return f"[output] {preview}{'...' if len(content) > 200 else ''}"

    # File read / write / edit by the agent
    if source == "agent" and action in ("read", "write", "edit"):
        path = args.get("path", "")
        return f"[{action}] {path}"

    # Workspace context recall
    if action == "recall":
        return "[recall] retrieving workspace context"

    if source == "environment" and observation == "recall":
        return "[recall] workspace context loaded"

    # Error observation
    if source == "environment" and observation == "error":
        preview = content[:200].replace("\n", " ")
        return f"[error] {preview}"

    return None


def _last_assistant_message(events: list[dict]) -> str:
    for event in reversed(events):
        source = event.get("source") or event.get("role", "")
        print(f"    [debug] checking event from {source} with action {event.get('action', '')}")
        if source == "agent":
            content = event.get("message") or event.get("content", "")
            if content:
                return content
    return ""


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