import os
import re
import time
import pathlib
import requests
import docker


AGENT_TASK_TEMPLATE = """\
You are assessing whether a scientific software artifact is reproducible.
IMPORTANT: Do NOT ask for confirmation. Do NOT wait for user input. Execute all steps immediately and autonomously.

The repository has already been downloaded and is available at {repo_path}.
Do NOT clone or download the repository yourself — it is already there, ready to use.
Your working directory (/workspace) is a separate sandbox; all repo files are under {repo_path}.

Follow these phases IN ORDER. Move to the next phase as soon as the current one is done.
{path_analysis}
== PHASE 1: RECONNAISSANCE (use max 3 commands) ==
Run these commands:
1. ls -la {repo_path}
2. cat {repo_path}/README.md (or whatever README file exists — check ls output)
3. nvidia-smi 2>/dev/null || echo "NO GPU"

After these 3 commands, DECIDE which path to take:
- If the README mentions GPU/CUDA and nvidia-smi shows "NO GPU" → check if the README also offers a CPU-only path, a validation-only path, or pre-computed results to verify. If yes, use that path. If GPU is truly required for ALL paths, skip to PHASE 5 with prereq_missing=true.
- If API tokens are REQUIRED for ALL paths (HuggingFace, OpenAI, W&B) → skip to PHASE 5 with prereq_missing=true
- Docker CLI is available. You can run `docker pull` and `docker run` commands. Do NOT attempt to start dockerd — the Docker daemon runs on the host and is already accessible. If the README uses Docker, follow its instructions directly (docker pull, docker run, docker-compose, etc.).
- Otherwise → continue to PHASE 2

== PHASE 2: INSTALL DEPENDENCIES (use max 8 commands) ==
First cd into the repo: cd {repo_path}
Install what the README requires:
- pip install <packages> (pip does NOT need -y flag)
- apt-get install -y <packages>
- If a requirements.txt or setup.py exists, use it: pip install -r requirements.txt 2>&1 | tail -20
- If a package fails, try ONE alternative, then skip it and note it in your report
- Pipe long outputs: pip install ... 2>&1 | tail -20
- Do NOT edit setup.py, requirements.txt, pyproject.toml, or any dependency-pin file to work around a version conflict, unless the README explicitly tells you to change it. If a pinned version fails to install, report that as the failure — do not loosen or rewrite the pin yourself.
- If dependency installation takes more than 3 commands and still fails, stop and move to the next phase — do not keep retrying with different flags or installing sub-dependencies one by one.
- VERSION FALLBACK RULE: If the README or dependency files do NOT pin specific versions (no == in requirements.txt, or just bare package names like "pandas, numpy"), first install the latest version. If this causes errors later in PHASE 4 (API incompatibility, deprecated/removed functions, TypeError from changed signatures, etc.), you may come back here and install an older version of the problematic package. Estimate the appropriate version based on when the repository was created (check git log --format=%ai -1 or file timestamps with ls -la). For example, if the repo was created in 2021 and pandas 3.x removed a function the code uses, try pandas 1.x. Document every such version change in your report under "env_fixes_applied".
- This fallback rule ONLY applies to unpinned dependencies. If versions ARE pinned (e.g. pandas==1.3.5) and fail to install, report that failure — do not change the pinned version.

== PHASE 3: PREPARE DATA (use max 3 commands) ==
If the experiment needs data preparation (download datasets, preprocess, etc.):
- Run the data preparation commands from the README
- If a download fails or takes too long, skip and note it

== PHASE 4: RUN EXPERIMENT (use max 8 commands) ==
Run the experiment exactly as the README describes:
- For Python scripts: python script.py 2>&1 | tail -50
- For notebooks: jupyter nbconvert --to notebook --execute notebook.ipynb 2>&1 | tail -30
- For shell scripts: bash script.sh 2>&1 | tail -50
- Check for output files: ls -la output/ results/ *.png *.csv 2>/dev/null
- If the script raises an error (AttributeError, ImportError, deprecated API call, etc.), do NOT open the file and patch the code to work around it. Capture the exact error and move on — a human trying to reproduce this artifact by hand would hit the same error, and your report needs to reflect that, not a version of the artifact you silently modified.
- EXCEPTION — version incompatibility with unpinned deps: If a script fails with an error that clearly indicates a version incompatibility (AttributeError on a function removed in newer versions, ImportError for a restructured submodule, TypeError from changed API signatures), AND the dependency was NOT pinned to a specific version in the repo, you MAY go back to PHASE 2 to install a compatible older version of that specific package and then retry the script. This is NOT a code modification — you are fixing the runtime environment, not the code. Do this at most ONCE per package. Record every such change under "env_fixes_applied" in your report.

== PHASE 5: REPORT (MANDATORY — you MUST do this) ==
Before calling the finish tool, you MUST first execute a bash command that echoes the JSON report below.
This is the ONLY way your report will be captured. Do NOT just put it in the finish message.

Run this command (fill in the values):
echo '{{
  "success": true or false,
  "prereq_missing": false,
  "missing_prereqs": [],
  "metrics": {{}},
  "error": null or "description of what went wrong, including the exact error message or traceback line if one was produced",
  "code_modified": true or false,
  "code_modifications": "If code_modified is true, describe exactly what you changed and why the README instructed it. Write 'none' if code_modified is false.",
  "env_fixes_applied": ["list of environment/dependency version changes you made, e.g. 'downgraded pandas from 3.0.3 to 1.5.3 because DataFrame.append() was removed in 2.0'. Write empty list [] if no version changes were needed"],
  "steps_completed": ["list", "of", "completed", "steps"]
}}'

Then call finish with a summary.

If prerequisites are missing (GPU, tokens, Docker, etc.):
- "success": false, "prereq_missing": true, "missing_prereqs": ["GPU/CUDA", ...]

CRITICAL RULES:
- NEVER run the same command twice. If you already ran a command and saw its output, do NOT run it again. Re-read the existing output from your conversation history instead.
- NEVER run nvidia-smi more than once — the result will NOT change.
- NEVER run pip install for the same package more than once — it is already installed.
- NEVER run ls on the same directory more than once — the listing does not change unless you created new files.
- Your goal is to ASSESS reproducibility by following the README instructions, not to FIX broken code. If something fails, report it — do not spend multiple attempts trying to repair the project.
- NEVER edit, patch, or rewrite any file inside the repository (source code, setup.py, config files, notebooks, etc.) unless the README explicitly instructs you to make that exact change as a documented setup step. Encountering a bug or a compatibility error is a valid, reportable outcome — it is not something for you to fix. If you are unsure whether a change is "required by the README" versus "a workaround you invented," treat it as invented and do not make it.
- The one exception is creating a missing directory that a script needs to write output into (e.g. `mkdir -p output/`) when the README's instructions assume it already exists — this is not a code change and is fine to do.
- If a step fails, try ONE different approach that does not involve modifying repository files (e.g., a different install flag, a different package manager). If that also fails, move to the next phase.
- ALWAYS use non-interactive flags: apt-get -y, conda -y. pip installs non-interactively by default.
- You MUST reach PHASE 5 and output the JSON. If you run out of ideas, go to PHASE 5 immediately.
- Do NOT attempt to start the Docker daemon (dockerd) — it runs on the host and is already accessible. Just use `docker pull` / `docker run` directly.

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
    for attempt in range(1, 4):
        try:
            r = requests.post(f"{openhands_url}/api/settings", json=payload, timeout=30)
            if r.status_code not in (200, 201):
                print(f"    [warn] settings POST returned {r.status_code}: {r.text[:300]}")
            else:
                print("    settings saved OK")
            return
        except requests.exceptions.RequestException as e:
            print(f"    [retry {attempt}/3] OpenHands not ready: {e}")
            time.sleep(10 * attempt)


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

    # Wait for OpenHands to fully shut down the runtime container
    # before attempting removal — the stop API is async.
    time.sleep(5)
    _cleanup_runtime_container(conversation_id)


def cleanup_stale_runtime_containers() -> None:
    """Remove ALL openhands-runtime-* containers (running or exited).

    Called at orchestrator startup to catch any containers left over
    from previous runs that slipped through per-conversation cleanup.
    """
    try:
        client = docker.from_env()
        containers = client.containers.list(
            all=True, filters={"name": "openhands-runtime-"}
        )
        if not containers:
            return
        print(f"    Cleaning up {len(containers)} stale runtime container(s)...")
        for c in containers:
            try:
                c.remove(force=True)
                print(f"    Removed {c.name}")
            except Exception:
                pass
    except Exception as e:
        print(f"    [warn] could not list/remove stale containers: {e}")


def _cleanup_runtime_container(conversation_id: str) -> None:
    """Remove the OpenHands runtime container for a finished conversation."""
    container_name = f"openhands-runtime-{conversation_id}"
    try:
        client = docker.from_env()
        container = client.containers.get(container_name)
        container.remove(force=True)
        # Verify removal — docker rm -f can silently fail on WSL2
        try:
            client.containers.get(container_name)
            print(f"    [warn] container {container_name} still exists after remove(force=True)")
        except docker.errors.NotFound:
            print(f"    Removed runtime container {container_name}")
    except docker.errors.NotFound:
        pass  # Container already gone
    except Exception as e:
        print(f"    [warn] could not remove container {container_name}: {e}")


def run_openhands_agent(openhands_url: str, readme: str, repo_path: str,
                        path_analysis: dict | None = None) -> tuple[str, list[dict], str]:
    readme_max = int(os.getenv("AGENT_README_MAX_CHARS", "30000"))
    readme_trimmed = _smart_truncate_readme(readme, max_chars=readme_max)
    path_analysis_text = _format_path_analysis(path_analysis)
    task = AGENT_TASK_TEMPLATE.format(
        repo_path=repo_path, readme=readme_trimmed, path_analysis=path_analysis_text
    )

    max_timeout_min = int(os.getenv("MAX_TIMEOUT_MINUTES", "60"))
    print(f"[2/3] Sending task to OpenHands agent (max {max_timeout_min} min)...")
    ensure_openhands_settings(openhands_url)

    for attempt in range(1, 4):
        try:
            resp = requests.post(
                f"{openhands_url}/api/conversations",
                json={"initial_user_msg": task},
                timeout=60,
            )
            if resp.ok:
                break
            print(f"    [retry {attempt}/3] conversations POST returned {resp.status_code}")
            time.sleep(10 * attempt)
        except requests.exceptions.RequestException as e:
            print(f"    [retry {attempt}/3] conversations POST failed: {e}")
            time.sleep(10 * attempt)
    else:
        raise RuntimeError(
            f"OpenHands POST /api/conversations failed after 3 retries"
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
            final_message = _build_final_output(all_events, str(data))
            return final_message, all_events, conversation_id

        # Fall back to agent-level state detected from events
        if agent_error_state:
            print(f"[info] Agent reached terminal state '{agent_error_state}' "
                  f"(conversation status still '{state}')")

            if agent_error_state == "finished":
                final_message = _build_final_output(all_events)
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
                new_cmds, current_quality, extensions_used + 1, all_events
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
    timeout_class = classify_timeout(all_events)
    timeout_type = timeout_class["timeout_type"]
    progress = timeout_class["progress_summary"]

    timeout_parts = [
        f"TIMEOUT ({timeout_type}): agent did not finish within {elapsed_min} minutes.",
        f"Progress at timeout: {progress}",
    ]

    if timeout_class.get("last_command") and not timeout_class.get("last_command_had_output"):
        timeout_parts.append(
            f"Last command still executing: {timeout_class['last_command'][:200]}"
        )
        if timeout_class.get("last_command_running_seconds"):
            timeout_parts.append(
                f"Command had been running for {timeout_class['last_command_running_seconds']}s"
            )

    if timeout_class.get("experiment_was_running"):
        timeout_parts.append(
            "NOTE: The main experiment script was actively running at timeout. "
            "This is a POSITIVE signal — the artifact may be reproducible but "
            "requires more time than the budget allows."
        )

    timeout_parts.append(f"\nWork done before timeout:\n{summary}")
    timeout_msg = "\n".join(timeout_parts)
    return timeout_msg, all_events, conversation_id


def _has_pending_command(events: list[dict]) -> bool:
    """True if the last agent action is a bash command with no result yet."""
    last_action = None
    last_obs = None
    for ev in reversed(events):
        src = ev.get("source", "")
        if src == "agent" and ev.get("action") == "run" and last_action is None:
            last_action = ev
        if src == "environment" and ev.get("observation") == "run" and last_obs is None:
            last_obs = ev
        if last_action and last_obs:
            break
    if last_action is None:
        return False
    # If there's no observation at all, or the last action came AFTER the
    # last observation, a command is still running.
    if last_obs is None:
        return True
    return last_action.get("id", 0) > last_obs.get("id", 0)


def _should_extend(new_cmds: int, quality: dict, extension_num: int,
                   events: list[dict] | None = None) -> tuple[bool, str]:
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

    # If a command is currently running (no output yet), always extend —
    # killing the agent while a long script is executing wastes all progress.
    if events and _has_pending_command(events):
        return True, "command still executing"

    if extension_num == 1:
        # Lenient: at least 1 meaningful command in the first 20 min
        if new_cmds >= 1:
            return True, f"agent active ({new_cmds} meaningful commands so far)"
        return False, "no meaningful commands executed in first 20 min"

    # Extension 2+: require continued progress, but be lenient for slow models
    # (a slow LLM may only produce 2-3 commands per 20-minute window)
    if new_cmds >= 1:
        return True, f"agent still progressing ({new_cmds} new commands since last checkpoint)"
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

    # Bash command output (source can be "environment" or "agent" depending on API version)
    if observation == "run" and not action:
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

    if observation == "recall" and not action:
        return "[recall] workspace context loaded"

    # Error observation
    if observation == "error" and not action:
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


def _summarize_events(events: list[dict], max_chars: int | None = None,
                      event_output_max_chars: int | None = None) -> str:
    """Build a readable summary of what the agent did from event history."""
    if max_chars is None:
        max_chars = int(os.getenv("EVENT_SUMMARY_MAX_CHARS", "8000"))
    if event_output_max_chars is None:
        event_output_max_chars = int(os.getenv("EVENT_OUTPUT_MAX_CHARS", "1000"))
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
        elif observation == "run" and not action:
            if content:
                trimmed = content[-event_output_max_chars:] if len(content) > event_output_max_chars else content
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
        if observation == "error" and not action:
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


def _build_final_output(events: list[dict], fallback: str = "") -> str:
    """Combine the agent's last message with the event summary.

    The last assistant message often contains the structured JSON report,
    but some models just say "All done!". By always appending the event
    summary the analyzer has concrete evidence of what happened.
    """
    last_msg = _last_assistant_message(events) or fallback
    summary = _summarize_events(events)
    return f"{last_msg}\n\n=== Agent activity log ===\n{summary}"


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


# ── Path analysis formatting ────────────────────────────────────────────────

def _format_path_analysis(analysis: dict | None) -> str:
    """Format the path analysis dict into a concise text block for the agent prompt."""
    if not analysis or not analysis.get("paths"):
        return ""

    lines = ["\n== PRE-ANALYSIS: Reproduction Path Map ==",
             "The following reproduction paths were identified from the README."]

    for i, path_name in enumerate(analysis.get("recommended_order", []), 1):
        detail = next((p for p in analysis["paths"] if p["name"] == path_name), None)
        if detail:
            prereqs = ", ".join(detail.get("prerequisites", [])) or "none"
            lines.append(f"{i}. {path_name} (confidence: {detail.get('confidence', '?')})")
            lines.append(f"   Prerequisites: {prereqs}")
            hint = detail.get("commands_hint", "see README")
            if hint:
                lines.append(f"   Hint: {hint}")

    # Include any paths not in recommended_order
    listed = set(analysis.get("recommended_order", []))
    for p in analysis["paths"]:
        if p["name"] not in listed:
            prereqs = ", ".join(p.get("prerequisites", [])) or "none"
            lines.append(f"- {p['name']} (confidence: {p.get('confidence', '?')})")
            lines.append(f"   Prerequisites: {prereqs}")

    if analysis.get("blockers"):
        lines.append(f"\nBlockers: {'; '.join(analysis['blockers'])}")
    if analysis.get("notes"):
        lines.append(f"Notes: {analysis['notes']}")

    lines.append("")
    lines.append("Use this map to guide your approach:")
    lines.append("- Try paths in the recommended order above")
    lines.append("- If your current path fails after 3 commands, switch to the next one")
    lines.append("- Do NOT spend time fixing a broken path when alternatives exist")
    lines.append("")

    return "\n".join(lines)


# ── Timeout classification ──────────────────────────────────────────────────

def classify_timeout(events: list[dict]) -> dict:
    """Classify what was happening when the agent timed out.

    Analyzes the event history to determine:
    - Which reproduction phase the agent reached
    - Whether a command was still running
    - Whether it was the main experiment script
    """
    phase = _detect_phase(events)
    last_cmd = _get_last_command_info(events)
    experiment_script = _detect_experiment_script(events)
    timeout_type = _classify_timeout_type(phase, last_cmd, events)

    # Determine if experiment was actively running at timeout
    experiment_was_running = (
        phase >= 4
        and last_cmd["command"] is not None
        and not last_cmd["had_output"]
        and experiment_script is not None
        and experiment_script in (last_cmd["command"] or "")
    )

    # Build progress summary
    phase_names = {1: "Reconnaissance", 2: "Install Dependencies",
                   3: "Prepare Data", 4: "Run Experiment", 5: "Report"}
    summary_parts = [f"Reached Phase {phase} ({phase_names.get(phase, 'Unknown')})"]

    if last_cmd["command"] and not last_cmd["had_output"]:
        summary_parts.append(f"Last command still running: {last_cmd['command'][:100]}")
        if last_cmd["running_seconds"]:
            summary_parts.append(f"Running for {last_cmd['running_seconds']}s at timeout")

    if experiment_script:
        summary_parts.append(f"Main experiment script identified: {experiment_script}")
        if experiment_was_running:
            summary_parts.append("POSITIVE SIGNAL: Experiment script was actively executing at timeout")

    return {
        "timeout_type": timeout_type,
        "phase_reached": phase,
        "last_command": last_cmd["command"],
        "last_command_had_output": last_cmd["had_output"],
        "last_command_running_seconds": last_cmd["running_seconds"],
        "experiment_script_detected": experiment_script,
        "experiment_was_running": experiment_was_running,
        "progress_summary": ". ".join(summary_parts),
    }


def _detect_phase(events: list[dict]) -> int:
    """Detect the highest reproduction phase the agent reached."""
    phase = 1
    has_install = False

    for ev in events:
        if ev.get("source") != "agent" or ev.get("action") != "run":
            continue
        args = ev.get("args", {}) if isinstance(ev.get("args"), dict) else {}
        cmd = args.get("command", "")
        cmd_lower = cmd.lower()

        # Phase 2: dependency installation
        if any(kw in cmd_lower for kw in [
            "pip install", "apt-get install", "conda install",
            "requirements.txt", "setup.py install", "npm install",
            "yarn install", "pip3 install",
        ]):
            phase = max(phase, 2)
            has_install = True

        # Phase 3: data download / preparation
        first_word = cmd.split()[0] if cmd.split() else ""
        if first_word in ("wget", "curl", "gdown", "kaggle") or "download" in cmd_lower:
            phase = max(phase, 3)

        # Phase 4: running actual experiment scripts
        is_runner = any(runner in cmd_lower for runner in [
            "python ", "python3 ", "bash ", "sh ", "jupyter",
            "./run", "make ", "Rscript ",
        ])
        is_setup = any(skip in cmd_lower for skip in [
            "setup.py", "pip", "install", "download", "apt-get",
        ])
        # docker run is always Phase 4 (running experiment in container)
        if "docker run" in cmd_lower or "docker exec" in cmd_lower:
            phase = max(phase, 4)
        elif is_runner and not is_setup:
            if has_install or phase >= 2:
                phase = max(phase, 4)

        # Phase 5: agent generating JSON report
        if '"success"' in cmd or "'success'" in cmd:
            phase = max(phase, 5)

    return phase


def _get_last_command_info(events: list[dict]) -> dict:
    """Analyze the last bash command: was it still running at timeout?"""
    last_action = None
    last_action_idx = -1
    last_obs_idx = -1

    for i, ev in enumerate(events):
        if ev.get("source") == "agent" and ev.get("action") == "run":
            last_action = ev
            last_action_idx = i
        if ev.get("observation") == "run" and not ev.get("action"):
            last_obs_idx = i

    if last_action is None:
        return {"command": None, "had_output": True, "running_seconds": None}

    args = last_action.get("args", {}) if isinstance(last_action.get("args"), dict) else {}
    cmd = args.get("command", "")
    still_running = last_obs_idx < last_action_idx

    running_seconds = None
    if still_running:
        # Estimate from event timestamps if available
        action_ts = last_action.get("timestamp", "")
        if action_ts and events:
            last_ts = events[-1].get("timestamp", "")
            if action_ts and last_ts:
                from datetime import datetime
                try:
                    t1 = datetime.fromisoformat(action_ts.replace("Z", "+00:00"))
                    t2 = datetime.fromisoformat(last_ts.replace("Z", "+00:00"))
                    running_seconds = int((t2 - t1).total_seconds())
                except Exception:
                    pass

    return {
        "command": cmd,
        "had_output": not still_running,
        "running_seconds": running_seconds,
    }


def _detect_experiment_script(events: list[dict]) -> str | None:
    """Try to identify the main experiment script from event history."""
    patterns = [
        r"python3?\s+(\S+\.py)",
        r"bash\s+(\S+\.sh)",
        r"sh\s+(\S+\.sh)",
        r"Rscript\s+(\S+\.R)",
        r"jupyter\s+nbconvert.*?(\S+\.ipynb)",
        r"\./(\S+\.(?:py|sh))",
    ]
    skip_keywords = ["setup", "install", "requirements", "download", "pip", "conda"]
    candidates = []

    for ev in events:
        if ev.get("source") != "agent" or ev.get("action") != "run":
            continue
        args = ev.get("args", {}) if isinstance(ev.get("args"), dict) else {}
        cmd = args.get("command", "")
        for pattern in patterns:
            match = re.search(pattern, cmd)
            if match:
                script = match.group(1)
                if not any(skip in script.lower() for skip in skip_keywords):
                    candidates.append(script)

    return candidates[-1] if candidates else None


def _classify_timeout_type(phase: int, last_cmd: dict, events: list[dict]) -> str:
    """Classify the timeout into a category."""
    if not last_cmd["command"]:
        return "timeout_agent_idle"

    # Check if agent was idle at the end (no recent commands)
    recent_actions = [ev for ev in events[-10:]
                      if ev.get("source") == "agent" and ev.get("action") == "run"]
    if not recent_actions:
        return "timeout_agent_idle"

    if phase <= 2:
        return "timeout_during_setup"
    elif phase == 3:
        return "timeout_during_data_prep"
    elif phase >= 4:
        return "timeout_during_experiment"

    return "timeout_unknown"