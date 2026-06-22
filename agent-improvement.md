# Agent Improvement Log — Scientific Artifact Reproducer

Chronological record of the iterative development process, problems encountered, and solutions applied to the automated scientific artifact reproduction system.

## System Overview

The system automatically assesses whether scientific software artifacts from academic papers are reproducible. Architecture:
- **Orchestrator** (Python) — clones repos, sends tasks to the agent, analyzes results
- **OpenHands** (CodeActAgent) — AI agent that follows README instructions to reproduce experiments
- **Ollama** (local LLM) — both drives the agent and performs final verdict analysis
- **Infrastructure** — WSL2, Docker containers, no GPU

---

## Phase 1: Initial Setup

### Architecture Decisions
- OpenHands 1.4 as the agent platform (provides sandboxed runtime containers)
- Ollama for local LLM inference (no API costs, full control)
- Docker Compose to orchestrate all services
- Initial model: `qwen2.5-coder:14b`

### Initial Pipeline
1. `cloner.py` — downloads repo (GitHub ZIP, Zenodo, git clone)
2. `agent.py` — sends README + task prompt to OpenHands, polls for completion
3. `analyzer.py` — sends agent output to Ollama for verdict classification
4. `main.py` — orchestrates the pipeline, saves results as JSON

### Initial Agent Prompt
A freeform prompt asking the agent to follow the README and reproduce the experiment. No structure, no command budget, no phase separation.

### Initial Configuration
- Timeout: fixed 30 minutes
- Condenser: `observation_masking` with `attention_window = 5`
- `max_message_chars`: 30,000 (OpenHands default)
- Truncation: 50/50 head+tail split (OpenHands default)

---

## Phase 2: First Test Runs & Early Problems

### Problem: Workspace Disappearing During Agent Run
**Observation:** The repository directory disappeared from the OpenHands runtime container while the agent was still working.

**Root Cause:** `./workspace` is a shared Docker bind mount between the orchestrator and OpenHands runtime containers. After `run_openhands_agent()` returned, `shutil.rmtree()` ran immediately in the orchestrator — but the OpenHands runtime container was still alive and had the directory mounted.

**Fix:**
- Added `stop_conversation()` function calling `POST /api/conversations/{id}/stop`
- Reordered pipeline: agent finishes → stop conversation → run analysis → delete workspace
- Changed `run_openhands_agent` return type to include `conversation_id`

### Problem: Fixed Timeout Too Short for pip install
**Observation:** Many repos need long `pip install` times. 30 minutes was insufficient for repos with heavy dependencies.

**User Insight:** Timeout shouldn't automatically mean failed — the agent may have done meaningful work before time ran out.

**Fix (v1):** Changed timeout message to include a summary of what the agent accomplished. Updated analyzer to not hard-code "failed" for timeouts — instead falls through to LLM analysis when the agent did meaningful work.

---

## Phase 3: Adaptive Timeout System

### Problem: Fixed Timeout is Either Too Short or Too Long
Some repos finish in 5 minutes, others need 60+. A fixed timeout wastes time on stuck agents or cuts short productive ones.

**User Idea:** Every 20 minutes the orchestrator should check the agent's output to determine if it's making progress. If so, extend the time. If not (stuck, looping), stop early. Each successive check should be stricter. Maximum 60 minutes total.

### Implementation: Checkpoint-Based Adaptive Timeout
- `CHECKPOINT_INTERVAL = 1200` (20 minutes)
- `MAX_EXTENSIONS = 2` (initially up to 60 min total)
- `_should_extend()` function with progressive strictness:
  - **Checkpoint 1** (20 min): lenient — at least 1 meaningful command
  - **Checkpoint 2** (40 min): strict — at least 2 new commands, or 1 with no errors
- Orchestrator logs checkpoint decisions for transparency

### Test: nanoGPT Repository
First test with adaptive timeout. Agent performed well:
- Ran `cd`, `nvidia-smi`, config analysis, `pip install`, data preparation, training
- Checkpoint 1 at 20m03s: "Extending +20 min: agent active (12 meaningful commands so far)"
- Agent edited config, ran training, checked logs
- Verdict: `partial` (training started but didn't complete in 60 min)

---

## Phase 4: Truncation & Context Window Issues

### Problem: Truncated Output Causes Misdiagnosis
**Observation:** Agent saw truncated pip install output and misdiagnosed errors. Stack traces were cut in half, so the agent couldn't see the actual error at the bottom.

**User Insight:** The agent thinks the error is GPU-related, but the real cause is likely something else — the end of the stack trace is truncated and invisible to the agent.

### OpenHands Truncation Architecture (Investigated)
Two-layer truncation pipeline:
1. **Layer 1:** `CmdOutputObservation._maybe_truncate()` — limits raw command output
2. **Layer 2:** `truncate_content()` in `event.py` — 50/50 head+tail split before sending to LLM

Both use `max_message_chars` (default 30,000).

### Fix: Asymmetric Truncation
**User Proposal:** Use an asymmetric split — e.g. 1000 chars from the beginning, 2000 from the middle, and 4000 from the end — to favor the tail where errors appear.

Created `patch_truncation.py` to replace the default 50/50 split:
- **Head:** 14% (~2,800 chars of 20,000)
- **Middle:** 29% (~5,800 chars)
- **Tail:** 57% (~11,400 chars)

Errors and final output typically appear at the end, so the tail gets the most space.

### Fix: Adjusted max_message_chars
- Changed from 30,000 (default) to 5,000 initially
- User questioned: with such a large context window (262K), do we even need to truncate at all?
- Settled on **20,000** — good balance for 262K context models

### Bug: patch_truncation.py Caused SyntaxError
First version used regex-based patching (`re.compile(r"def truncate_content...")`). This corrupted escape sequences in `event.py`, causing `SyntaxError: unterminated string literal (detected at line 184)` on OpenHands startup.

**Fix:** Rewrote to use exact string matching with a relaxed fallback. Confirmed: "Patched truncate_content() (exact match)".

---

## Phase 5: Analysis Improvements

### Problem: LLM Verdict Lacked Detail
The analysis only produced `verdict` and `explanation`. No information about what was accomplished, what code changes were needed, or specific failure reasons.

**User Request:** The LLM should summarize what was actually accomplished, what code modifications the agent had to make, and the specific reason for failure if the experiment didn't succeed.

### Fix: Extended Analysis Output
Added three new fields to the analysis prompt and `ReproductionResult` model:
- `steps_completed` — what was successfully done
- `code_modifications` — what code changes the agent applied to the project
- `failure_reason` — specific error or blocker if not reproduced

---

## Phase 6: Catastrophic Agent Looping — Deep Analysis

### Problem: All 5 Test Repos → "inconclusive"
After running 5 repos from the ICSE papers list, analysis of results revealed catastrophic command repetition:

| Repository | Repeated Command | Count |
|------------|-----------------|-------|
| SmartMark | `pip install pycryptodome` | 88x |
| Fonte | `pip install lib/SBFL` | 70x |
| AidUI | `cat env file` | 87x |
| Chronos | `nvidia-smi` | 108x |
| Zenodo record | `nvidia-smi` | 108x |

**All 5 verdicts: inconclusive.** The agent never produced the required JSON output in any case.

### Root Cause Analysis
Three contributing factors identified:
1. **Model loses context (60%)** — the 27B model (running with `attention_window=5`) forgets what it already executed
2. **Condenser too aggressive (25%)** — `attention_window=5` masks all but the last 5 event outputs, so the agent can't see its own previous commands
3. **OpenHands TROUBLESHOOTING prompt bias (15%)** — the default system prompt encourages the agent to "try again" and "fix errors", reinforcing looping behavior

### Solution: Four-Part Fix

#### 1. Increased attention_window: 5 → 20
In `openhands/Dockerfile`:
```
RUN printf '...\nattention_window = 20\n' > /app/config.toml
```
The agent can now see the last 20 event outputs instead of 5, dramatically reducing context amnesia.

#### 2. Phase-Based Agent Prompt with Command Budget
Replaced the freeform prompt with a structured 5-phase template:

- **PHASE 1: RECONNAISSANCE** (max 3 commands) — `ls`, `cat README`, `nvidia-smi`
- **PHASE 2: INSTALL DEPENDENCIES** (max 5 commands)
- **PHASE 3: PREPARE DATA** (max 3 commands)
- **PHASE 4: RUN EXPERIMENT** (max 5 commands)
- **PHASE 5: REPORT** (mandatory JSON output)

Total budget: 25 commands. Explicit anti-loop rules:
- "NEVER run the same command twice"
- "NEVER run nvidia-smi more than once"
- "NEVER run pip install for the same package more than once"
- "Your goal is to ASSESS reproducibility, not to FIX broken code"

#### 3. Repetition Detection in classify_agent_run
Added metrics to detect command repetition:
- `unique_commands` — count of distinct commands
- `repetition_ratio` — 0.0 (all unique) to 1.0 (all duplicates)
- `quality = "poor"` if `repetition_ratio > 0.7`

#### 4. Repetition Check in Checkpoint Logic
Updated `_should_extend()` to reject time extensions when `repetition_ratio > 0.5`:
```python
if rep_ratio > 0.5:
    return False, "agent is looping"
```

### Result: Dramatic Improvement
Test with SmartMark (previously 88x `pip install pycryptodome`):

| Metric | Before | After |
|--------|--------|-------|
| Commands executed | 88 | 6 |
| Unique commands | ~1 | 6 (100%) |
| Repetition ratio | ~99% | 0% |
| Quality | poor | **good** |
| Verdict | inconclusive | **reproduced** |

The agent followed all 5 phases correctly and completed the experiment.

---

## Phase 7: Agent State Handling Bug

### Problem: "finished" Treated as Error
**Observation:** Agent output started with `AGENT_ERROR (finished): unknown error` even when the agent completed successfully.

**Root Cause:** In the polling loop, the `finished` agent state was grouped with `error` and `stopped` in the terminal state detection. When the conversation-level status hadn't caught up yet (still `RUNNING`), the code constructed an `AGENT_ERROR` message for the `finished` state.

This also broke JSON extraction from agent output — the `AGENT_ERROR` prefix prevented `_extract_json_object()` from finding the agent's JSON report, so `steps_completed` was always empty.

**Fix:** Added special handling for `finished` state — use `_last_assistant_message()` instead of constructing an error message:
```python
if agent_error_state == "finished":
    final_message = _last_assistant_message(all_events) or _summarize_events(all_events)
```

---

## Phase 8: Runtime Container Cleanup

### Problem: Orphaned Docker Containers
**Observation:** Runtime containers are not being removed and keep accumulating — 9 leftover `openhands-runtime-*` containers from previous runs, consuming resources.

**Root Cause:** OpenHands creates a runtime container per conversation (`openhands-runtime-{conversation_id}`). The `stop_conversation()` API call doesn't always remove the container.

### Fix: Automatic Container Cleanup
- Added `_cleanup_runtime_container()` using the Python `docker` SDK
- Called from `stop_conversation()` after the API stop call
- Added Docker socket mount (`/var/run/docker.sock`) to orchestrator in `docker-compose.yml`
- Added `docker==7.1.0` to `requirements.txt`

---

## Phase 9: Verdict Taxonomy Refinement

### Problem: "partial" Doesn't Capture Timeout-While-Running
**Observation:** When an experiment was successfully launched and actively running but our timeout expired, the verdict was `partial`. But `partial` implies something went wrong — "some steps failed."

**User Insight:** If the final experiment script was launched and running without errors when the timeout hit, that's actually a positive outcome. "Partial" means some parts succeeded and some failed — but here nothing actually failed, it's our time budget that's insufficient, not the artifact's fault. These are fundamentally different situations.

### Fix: New Verdict `timeout_running`
Added a sixth verdict category:

| Verdict | Meaning |
|---------|---------|
| `reproduced` | Experiment completed, results produced |
| `timeout_running` | Experiment was running successfully when time expired — positive signal, our time limit, not artifact's fault |
| `partial` | Some steps genuinely FAILED with errors |
| `failed` | Agent tried, but artifact has real bugs |
| `prereq_missing` | Needs GPU/tokens/data we don't have |
| `inconclusive` | Agent malfunctioned, can't judge |

### Configurable Timeout
Extracted `MAX_TIMEOUT_MINUTES` to `.env` (default 60). Checkpoint count calculated automatically:
```python
max_timeout = int(os.getenv("MAX_TIMEOUT_MINUTES", "60")) * 60
MAX_EXTENSIONS = max(0, (max_timeout // CHECKPOINT_INTERVAL) - 1)
```
Setting `MAX_TIMEOUT_MINUTES=180` gives max 3 hours with checkpoints every 20 minutes.

### Agent Quality in Results
Added `agent_quality` field to `ReproductionResult` — now saved alongside verdict for post-analysis:
- `meaningful_commands`, `unique_commands`, `repetition_ratio`
- `total_errors`, `agent_stuck`, `quality`

---

## Phase 10: Second Test Run — Validation

Ran 3 repositories with all improvements applied:

| Repository | Verdict | Commands | Unique | Repetition | Analysis |
|------------|---------|----------|--------|------------|----------|
| SmartMark | **reproduced** | 6 | 6 | 0% | Watermark embedding and verification completed without errors |
| Fonte | **failed** (env) | — | — | — | SparseDtype error — needs Python 3.9, runtime has 3.12 |
| AidUI | **partial** (timeout) | — | — | — | Deps installed, model downloaded, but timed out before main script |

All three verdicts are correct and meaningful — a dramatic improvement from the initial "all 5 inconclusive" state.

---

## Model History

| Period | Model | Context | Notes |
|--------|-------|---------|-------|
| Initial | `qwen2.5-coder:14b` | ~32K | First tests, severe looping |
| Mid | `qwen2.5-coder:32b` | ~32K | Better reasoning, still looped |
| Current | `fredrezones55/Qwopus3.6` (27B) | 262K | Large context helps with condenser, much better phase following |

---

## Current Architecture Summary

```
.env                          # Model config, timeout, repo list
docker-compose.yml            # Ollama + OpenHands + Orchestrator
orchestrator/
  main.py                     # Pipeline: clone → agent → analyze → save
  agent.py                    # Phase-based prompt, adaptive timeout, repetition detection
  analyzer.py                 # LLM verdict with 6 categories
  models.py                   # ReproductionResult Pydantic model
  cloner.py                   # GitHub ZIP / Zenodo / git clone strategies
openhands/
  Dockerfile                  # Custom OpenHands: attention_window=20, truncation patch
  patch_truncation.py         # Asymmetric head+mid+tail truncation
```

### Key Parameters
- `attention_window = 20` — agent sees last 20 event outputs
- `max_message_chars = 20,000` — per-observation truncation limit
- Truncation split: 14% head / 29% mid / 57% tail
- Checkpoint interval: 20 minutes
- Max timeout: configurable via `MAX_TIMEOUT_MINUTES` (default 60)
- Command budget: 25 commands across 5 phases

---

## Lessons Learned

1. **Context window management is critical for agent quality.** The single biggest improvement came from increasing `attention_window` from 5 to 20. When the agent can't see its own previous actions, it loops catastrophically.

2. **Structured prompts prevent aimless behavior.** The phase-based prompt with explicit command budgets transformed the agent from "repeat pip install 88 times" to "follow 5 phases, finish in 6 commands."

3. **Truncation strategy matters.** The default 50/50 split loses error messages at the end of output. Asymmetric truncation (favoring the tail) preserves stack traces and error messages that the agent needs for diagnosis.

4. **Timeout is not a binary outcome.** Distinguishing "experiment was running successfully" (`timeout_running`) from "something failed" (`partial`) gives much more useful information about artifact reproducibility.

5. **Repetition detection as a circuit breaker.** Monitoring unique command ratio lets the orchestrator stop wasting time on a looping agent instead of waiting for the full timeout.

6. **Container lifecycle management is essential.** Without explicit cleanup, runtime containers accumulate and consume resources. The orchestrator needs Docker socket access to clean up after itself.

7. **Iterative testing on real repos reveals problems that unit tests miss.** Every improvement in this log came from observing actual agent behavior on real scientific repositories, not from theoretical analysis.
