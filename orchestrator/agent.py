import os
import time
import pathlib
import requests
import docker


AGENT_TASK_TEMPLATE = """\
You are assessing whether a scientific software artifact is reproducible.
IMPORTANT: Do NOT ask for confirmation. Do NOT wait for user input. Execute all steps immediately and autonomously.

The repository has been cloned to {repo_path}.
You have a BUDGET of 25 commands. Use them wisely — do not waste commands repeating things.

Follow these phases IN ORDER. Move to the next phase as soon as the current one is done.

== PHASE 1: RECONNAISSANCE (use max 3 commands) ==
Run these commands:
1. cd {repo_path} && ls -la
2. cat README.md (or whatever README file exists — check ls output)
3. nvidia-smi 2>/dev/null || echo "NO GPU"

After these 3 commands, DECIDE IMMEDIATELY:
- If the README mentions GPU/CUDA/Docker --gpus and nvidia-smi shows "NO GPU" → skip to PHASE 5 with prereq_missing=true
- If API tokens are REQUIRED (HuggingFace, OpenAI, W&B) → skip to PHASE 5 with prereq_missing=true
- Otherwise → continue to PHASE 2

== PHASE 2: INSTALL DEPENDENCIES (use max 5 commands) ==
Install what the README requires:
- pip install <packages> (pip does NOT need -y flag)
- apt-get install -y <packages>
- If a package fails, try ONE alternative, then skip it and note it in your report
- Pipe long outputs: pip install ... 2>&1 | tail -20

== PHASE 3: PREPARE DATA (use max 3 commands) ==
If the experiment needs data preparation (download datasets, preprocess, etc.):
- Run the data preparation commands from the README
- If a download fails or takes too long, skip and note it

== PHASE 4: RUN EXPERIMENT (use max 5 commands) ==
Run the experiment exactly as the README describes:
- For Python scripts: python script.py 2>&1 | tail -50
- For notebooks: jupyter nbconvert --to notebook --execute notebook.ipynb 2>&1 | tail -30
- For shell scripts: bash script.sh 2>&1 | tail -50
- Check for output files: ls -la output/ results/ *.png *.csv 2>/dev/null

== PHASE 5: REPORT (MANDATORY — you MUST do this) ==
Output this JSON block. This is REQUIRED regardless of whether the experiment succeeded or failed:

```json
{{
  "success": true or false,
  "prereq_missing": false,
  "missing_prereqs": [],
  "metrics": {{}},
  "error": null or "description of what went wrong",
  "steps_completed": ["list", "of", "completed", "steps"]
}}
```

If prerequisites are missing (GPU, tokens, etc.):
- "success": false, "prereq_missing": true, "missing_prereqs": ["GPU/CUDA", ...]

CRITICAL RULES:
- NEVER run the same command twice. You already have its output in your conversation history. Re-read the output instead of re-running.
- NEVER run nvidia-smi more than once — the result will NOT change.
- NEVER run pip install for the same package more than once — it is already installed.
- Your goal is to ASSESS reproducibility, not to FIX broken code. If something fails, report it — do not spend multiple attempts trying to repair the project.
- If a step fails, try ONE different approach. If that also fails, move to the next phase.
- ALWAYS use non-interactive flags: apt-get -y, conda -y. pip installs non-interactively by default.
- You MUST reach PHASE 5 and output the JSON. If you run out of ideas, go to PHASE 5 immediately.

README:
---
{readme}
---
"""


def ensure_openhands_settings(openhands_url: str) -> None:
    agent_model = os.getenv("AGENT_MODEL", "qwen2.5-coder:32b")
    ollama_base = os.getenv("OLLAMA_URL", "http://host.docker.internal:11434")
    payload = {
        "llm_model":    f"ollama/{agent_model}",
        "llm_base_url": ollama_base,
        "llm_api_key":  "ollama",
        "enable_default_condenser": False,
    }
    r = requests.post(f"{openhands_url}/api/settings", json=payload, timeout=10)
    if r.status_code not in (200, 201):
        print(f"    [warn] settings POST returned {r.status_code}: {r.text[:300]}")
    else:
        print("    settings saved OK")


def stop_conversation(openhands_url: str, conversation_id: str) -> None:
    """Stop a running OpenHands conversation and its runtime container."""
    try:
        r = requests.post(
            f"{openhands_url}/api/conversations/{conversation_id}/stop",
            timeout=15,
        )
        if r.ok:
            print(f"    Conversation {conversation_id} stopped OK")
        else:
            print(f"    [warn] stop returned {r.status_code}: {r.text[:200]}")
    except Exception as e:
        print(f"    [warn] could not stop conversation: {e}")

    # Clean up the runtime container left behind by OpenHands
    _cleanup_runtime_container(conversation_id)


def _cleanup_runtime_container(conversation_id: str) -> None:
    """Remove the OpenHands runtime container for a finished conversation."""
    container_name = f"openhands-runtime-{conversation_id}"
    try:
        client = docker.from_env()
        container = client.containers.get(container_name)
        container.remove(force=True)
        print(f"    Removed runtime container {container_name}")
    except docker.errors.NotFound:
        pass  # Container already gone
    except Exception as e:
        print(f"    [warn] could not remove container {container_name}: {e}")


def run_openhands_agent(openhands_url: str, readme: str, repo_path: str) -> tuple[str, list[dict], str]:
    readme_trimmed = _smart_truncate_readme(readme, max_chars=6000)
    task = AGENT_TASK_TEMPLATE.format(repo_path=repo_path, readme=readme_trimmed)

    max_timeout_min = int(os.getenv("MAX_TIMEOUT_MINUTES", "60"))
    print(f"[2/3] Sending task to OpenHands agent (max {max_timeout_min} min)...")
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
    agent_error_state: str | None = None

    # Adaptive timeout with 20-min checkpoints.
    # MAX_TIMEOUT_MINUTES is configurable via .env (default 60).
    # At each checkpoint the orchestrator evaluates progress — if the
    # agent is doing useful work it gets another 20 min, but each
    # successive check is stricter.
    CHECKPOINT_INTERVAL = 1200  # 20 minutes
    max_timeout = int(os.getenv("MAX_TIMEOUT_MINUTES", "60")) * 60
    MAX_EXTENSIONS = max(0, (max_timeout // CHECKPOINT_INTERVAL) - 1)

    start_time = time.time()
    next_checkpoint = start_time + CHECKPOINT_INTERVAL
    hard_deadline = start_time + max_timeout
    extensions_used = 0
    last_checkpoint_cmds = 0

    while time.time() < hard_deadline:
        time.sleep(10)
        status_resp = requests.get(
            f"{openhands_url}/api/conversations/{conversation_id}",
            timeout=60,
        )
        status_resp.raise_for_status()
        data = status_resp.json()
        state = data.get("status", "running")
        runtime_status = data.get("runtime_status", "")
        elapsed = int(time.time() - start_time)
        print(f"[debug] status={state!r} runtime_status={runtime_status!r} "
              f"events={len(all_events)} elapsed={elapsed // 60}m{elapsed % 60:02d}s")

        new_events = _fetch_new_events(openhands_url, conversation_id, last_event_id)
        for event in new_events:
            all_events.append(event)
            last_event_id = max(last_event_id, event.get("id", last_event_id))
            description = _describe_event(event)
            if description:
                print(f"  {description}")

            # Detect terminal agent states from events (the conversation
            # status may stay RUNNING even after the agent errors out).
            if event.get("observation") == "agent_state_changed":
                agent_state = event.get("extras", {}).get("agent_state", "")
                if agent_state in ("error", "stopped", "finished", "awaiting_user_input"):
                    agent_error_state = agent_state

        # Check conversation-level status first
        if state.lower() in ("finished", "error", "stopped", "awaiting_user_input"):
            final_message = _last_assistant_message(all_events) or str(data)
            return final_message, all_events, conversation_id

        # Fall back to agent-level state detected from events
        if agent_error_state:
            print(f"[info] Agent reached terminal state '{agent_error_state}' "
                  f"(conversation status still '{state}')")

            if agent_error_state == "finished":
                # Normal completion — use the last assistant message
                final_message = _last_assistant_message(all_events) or _summarize_events(all_events)
            elif agent_error_state == "awaiting_user_input":
                summary = _summarize_events(all_events)
                final_message = f"Agent stopped (awaiting user input) after these steps:\n\n{summary}"
            else:
                reason = _agent_error_reason(all_events)
                final_message = (f"AGENT_ERROR ({agent_error_state}): {reason}"
                                 f"\n\n{_summarize_events(all_events)}")
            return final_message, all_events, conversation_id

        # ── Adaptive checkpoint ──────────────────────────────────────
        if time.time() >= next_checkpoint:
            current_quality = classify_agent_run(all_events)
            new_cmds = current_quality["meaningful_commands"] - last_checkpoint_cmds

            if extensions_used >= MAX_EXTENSIONS:
                print(f"[checkpoint] Hard deadline reached ({MAX_EXTENSIONS + 1} × "
                      f"{CHECKPOINT_INTERVAL // 60} min = "
                      f"{(MAX_EXTENSIONS + 1) * CHECKPOINT_INTERVAL // 60} min)")
                break

            should_extend, reason = _should_extend(
                new_cmds, current_quality, extensions_used + 1
            )

            if should_extend:
                extensions_used += 1
                last_checkpoint_cmds = current_quality["meaningful_commands"]
                next_checkpoint += CHECKPOINT_INTERVAL
                print(f"[checkpoint {extensions_used}/{MAX_EXTENSIONS}] "
                      f"Extending +{CHECKPOINT_INTERVAL // 60} min: {reason}")
            else:
                print(f"[checkpoint] Stopping early: {reason}")
                break

    elapsed_min = int((time.time() - start_time) / 60)
    summary = _summarize_events(all_events)
    timeout_msg = (f"TIMEOUT: agent did not finish within {elapsed_min} minutes.\n\n"
                   f"Work done before timeout:\n{summary}")
    return timeout_msg, all_events, conversation_id


def _should_extend(new_cmds: int, quality: dict, extension_num: int) -> tuple[bool, str]:
    """Decide whether to grant more time at a checkpoint.

    Each successive checkpoint is stricter:
      extension 1 (20→40 min): lenient — any meaningful new work
      extension 2+ (40→60→... min): strict — must show continued progress
    """
    if quality["agent_stuck"]:
        return False, "agent stuck in loop"

    if quality["empty_response"]:
        return False, "agent produced no actions"

    # High repetition means the agent is looping — don't give it more time
    rep_ratio = quality.get("repetition_ratio", 0.0)
    if rep_ratio > 0.5:
        return False, (f"agent is looping (repetition ratio {rep_ratio:.0%}, "
                       f"{quality['unique_commands']} unique / "
                       f"{quality['meaningful_commands']} total commands)")

    if extension_num == 1:
        # Lenient: at least 1 meaningful command in the first 20 min
        if new_cmds >= 1:
            return True, f"agent active ({new_cmds} meaningful commands so far)"
        return False, "no meaningful commands executed in first 20 min"

    # Extension 2+: strict — must show continued progress
    if new_cmds >= 2:
        return True, f"agent still progressing ({new_cmds} new commands since last checkpoint)"
    if new_cmds == 1:
        # 1 new command in 20 min is borderline — allow if no errors
        if quality["total_errors"] == 0:
            return True, "slow but error-free progress (1 new command)"
        return False, f"minimal progress with errors ({quality['total_errors']} errors)"
    return False, "no new commands since last checkpoint"


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


def _agent_error_reason(events: list[dict]) -> str:
    """Extract the error reason from agent_state_changed events."""
    for event in reversed(events):
        if event.get("observation") == "agent_state_changed":
            reason = event.get("extras", {}).get("reason", "")
            if reason:
                return reason
    return "unknown error"


def _summarize_events(events: list[dict], max_chars: int = 4000) -> str:
    """Build a readable summary of what the agent did from event history."""
    lines = []
    for event in events:
        source = event.get("source", "")
        action = event.get("action", "")
        observation = event.get("observation", "")
        args = event.get("args", {}) if isinstance(event.get("args"), dict) else {}
        content = event.get("content", "")

        if source == "agent" and action == "run":
            cmd = args.get("command", "")
            lines.append(f"$ {cmd}")
        elif source == "environment" and observation == "run":
            if content:
                # Keep last 300 chars of output to capture errors/results
                trimmed = content[-300:] if len(content) > 300 else content
                lines.append(trimmed)
        elif source == "agent" and action == "message":
            msg = event.get("message") or content
            if msg:
                lines.append(f"[agent] {msg}")
        elif observation == "agent_state_changed":
            state = event.get("extras", {}).get("agent_state", "")
            reason = event.get("extras", {}).get("reason", "")
            if state:
                lines.append(f"[state] {state}" + (f": {reason}" if reason else ""))

    summary = "\n".join(lines)
    if len(summary) > max_chars:
        summary = summary[-max_chars:]
    return summary


def _normalize_cmd(cmd: str) -> str:
    """Normalize a command for deduplication (strip whitespace, collapse paths)."""
    return " ".join(cmd.split())


def classify_agent_run(events: list[dict]) -> dict:
    """Classify the quality of the agent's attempt based on event history.

    Returns a dict with:
      - meaningful_commands: int — number of non-trivial bash commands executed
      - unique_commands: int — number of DISTINCT meaningful commands
      - total_errors: int — number of ErrorObservation events
      - agent_stuck: bool — whether AgentStuckInLoopError occurred
      - repetition_ratio: float — 0.0 (no repetition) to 1.0 (all commands repeated)
      - empty_response: bool — agent produced no actions at all
      - quality: "good" | "poor" | "none" — overall assessment
    """
    meaningful_cmds = 0
    total_errors = 0
    agent_stuck = False
    has_any_action = False
    trivial_cmds = {"cd", "pwd", "ls", "echo", "whoami", "cat"}
    seen_commands: set[str] = set()

    for event in events:
        source = event.get("source", "")
        action = event.get("action", "")
        observation = event.get("observation", "")
        args = event.get("args", {}) if isinstance(event.get("args"), dict) else {}
        extras = event.get("extras", {})

        # Count meaningful bash commands (not just cd/ls)
        if source == "agent" and action == "run":
            has_any_action = True
            cmd = args.get("command", "").strip()
            # For compound commands (cd X && pip install Y), check all parts
            parts = [p.strip() for p in cmd.replace("&&", ";").replace("||", ";").split(";")]
            has_meaningful = False
            for part in parts:
                first_word = part.split()[0] if part.split() else ""
                if first_word and first_word not in trivial_cmds:
                    has_meaningful = True
                    break
            if has_meaningful:
                meaningful_cmds += 1
                seen_commands.add(_normalize_cmd(cmd))

        # Count error observations
        if source == "environment" and observation == "error":
            total_errors += 1

        # Detect stuck-in-loop
        if observation == "agent_state_changed":
            reason = extras.get("reason", "")
            if "StuckInLoop" in reason:
                agent_stuck = True

    empty_response = not has_any_action
    unique_cmds = len(seen_commands)

    # Repetition ratio: 0.0 = every command is unique, 1.0 = all are duplicates
    if meaningful_cmds > 0:
        repetition_ratio = round(1.0 - unique_cmds / meaningful_cmds, 2)
    else:
        repetition_ratio = 0.0

    # Classify quality — repetition above 0.7 means >70% of commands are duplicates
    if empty_response:
        quality = "none"
    elif meaningful_cmds == 0 or agent_stuck or repetition_ratio > 0.7:
        quality = "poor"
    else:
        quality = "good"

    return {
        "meaningful_commands": meaningful_cmds,
        "unique_commands": unique_cmds,
        "total_errors": total_errors,
        "agent_stuck": agent_stuck,
        "repetition_ratio": repetition_ratio,
        "empty_response": empty_response,
        "quality": quality,
    }


def _last_assistant_message(events: list[dict]) -> str:
    for event in reversed(events):
        source = event.get("source") or event.get("role", "")
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