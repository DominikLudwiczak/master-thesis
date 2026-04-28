# Scientific Artifact Reproducer

Automatically reproduces computational experiments from GitHub repositories using a fully local, zero-cost AI stack. Point it at one or more repos, walk away, and come back to a JSON verdict for each one.

---

## How it works

```
GitHub URL(s)
     │
     ▼
Orchestrator          clones repo · reads README · sends task
     │
     ▼
OpenHands agent       installs deps · runs experiment · retries on errors
(qwen2.5-coder)       sandboxed inside Docker
     │
     ▼
Ollama analyzer       classifies result · extracts metrics · writes verdict
(llama3.2:1b)
     │
     ▼
results/<repo>.json   reproduced | partial | failed
```

Everything runs locally. No API keys, no cloud, no cost.

---

## Requirements

- Docker + Docker Compose
- ~30 GB free disk (models + workspace)
- NVIDIA GPU recommended (`qwen2.5-coder:14b` needs ~10 GB VRAM); CPU works but is slow
- macOS users: Docker Desktop with default file sharing settings (startup takes ~3–4 min per run due to volume mount latency — this is expected)

---

## Project structure

```
agent/
├── docker-compose.yml
├── openhands/
│   └── Dockerfile          # custom OpenHands image (extended startup timeout)
├── orchestrator/
│   ├── Dockerfile
│   ├── main.py             # entry point — define REPOS list here
│   ├── agent.py            # OpenHands conversation driver
│   ├── analyzer.py         # Ollama result analyzer
│   ├── cloner.py           # GitHub repo cloner
│   ├── models.py           # ReproductionResult schema
│   └── requirements.txt
├── workspace/              # repos cloned here (auto-created)
└── results/                # JSON verdicts written here (auto-created)
```

---

## Setup

**1. Pull images and models**

```bash
docker compose up ollama ollama-pull openhands --build -d
```

This builds the custom OpenHands image, starts Ollama, and downloads both models. The first run takes a while depending on your connection. Models are cached in a Docker volume so subsequent runs are instant.

**2. Verify OpenHands is up**

Open [http://localhost:3000](http://localhost:3000) — you should see the OpenHands UI.

---

## Configuration

Models can be changed via environment variables (defaults shown):

```bash
AGENT_MODEL=qwen2.5-coder:14b    # runs the experiment — needs strong tool-use ability
ANALYSIS_MODEL=llama3.2:1b       # analyses the result — lightweight is fine
```

Pass them inline or export before running:

```bash
AGENT_MODEL=qwen2.5-coder:7b docker compose run orchestrator
```

The REPOS list to run by default is defined at the top of `orchestrator/main.py`.

---

## Running

### Single repo

```bash
GITHUB_REPO=https://github.com/org/repo docker compose run orchestrator
```

### Multiple repos

```bash
GITHUB_REPOS=https://github.com/org/repo1,https://github.com/org/repo2 docker compose run orchestrator
```

### Default batch (REPOS list in main.py)

```bash
docker compose run orchestrator
```

The orchestrator processes repos sequentially and prints a live summary:

```
============================================================
Running 2 repo(s)
============================================================

[Repo 1/2] https://github.com/org/repo1
------------------------------------------------------------
[1/3] Cloning ...
[2/3] Sending task to OpenHands agent...
[debug] status='RUNNING' runtime_status='STATUS$ACTIVE' events=12
...
[3/3] Analyzing result with Ollama...
Verdict: reproduced
```

---

## Results

Each repo produces a timestamped JSON file in `./results/`:

```json
{
  "repo_url": "https://github.com/org/repo",
  "agent_output": "...",
  "verdict": "reproduced",
  "error_type": null,
  "metrics_found": {
    "accuracy": 0.923,
    "f1": 0.891
  },
  "analysis": "The experiment completed successfully. The reported accuracy of 0.923 matches the paper's Table 2 results within rounding error."
}
```

**Verdict values:**

| Verdict | Meaning |
|---|---|
| `reproduced` | Experiment ran to completion with plausible results |
| `partial` | Some steps succeeded but final result was not obtained |
| `failed` | Could not run the experiment at all |

**Error types** (when verdict is not `reproduced`):

| Type | Meaning |
|---|---|
| `env` | Missing dependency, wrong Python version, CUDA issue |
| `code` | Bug in the repo's own code |
| `data` | Missing dataset or download failed |
| `timeout` | Agent ran out of time (20 min limit) |

---

## Tips

**macOS startup delay** — on macOS with Docker Desktop, the OpenHands runtime sandbox takes 3–4 minutes to initialize on first use per conversation (due to Docker volume mount latency). This is normal. The custom `openhands/Dockerfile` extends the startup timeout to 600 s to accommodate this.

**Disk space** — large repos with bundled datasets can fill your disk during cloning. The orchestrator automatically retries with a shallow sparse clone if it runs out of space. You can also free Docker cache manually:

```bash
docker image prune -a
docker builder prune -a
```

**No GPU** — remove the `deploy:` block from the `ollama` service in `docker-compose.yml`. Inference will be slow but functional. Consider smaller models (`qwen2.5-coder:7b` and `llama3.2:1b`).

**Monitoring a run** — tail the orchestrator logs in another terminal:

```bash
docker compose logs -f orchestrator
```

**OpenHands UI** — you can watch the agent work in real time at [http://localhost:3000](http://localhost:3000) while the orchestrator is running.

**Re-running a repo** — the orchestrator always deletes and re-clones the workspace before each run, so re-runs are always clean.

---

## Services

| Service | Image | Purpose |
|---|---|---|
| `ollama` | `ollama/ollama` | Serves LLMs locally on port 11434 |
| `ollama-pull` | `ollama/ollama` | One-shot model downloader (exits after pull) |
| `openhands` | custom build (`openhands/Dockerfile`) | Agentic execution — bash, file tools, Docker sandbox |
| `orchestrator` | `python:3.12-slim` | Clones repos, drives OpenHands, calls Ollama |

---

## Stopping

```bash
docker compose down          # stop everything, keep volumes (models preserved)
docker compose down -v       # stop everything AND delete model cache
```