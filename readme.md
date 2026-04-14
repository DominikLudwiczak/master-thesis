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
(llama3.2:3b)
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

---

## Project structure

```
your-project/
├── docker-compose.yml
├── .env                        # copy from .env.example
├── workspace/                  # repos cloned here (create empty)
├── results/                    # JSON verdicts written here (create empty)
└── orchestrator/
    ├── main.py                 # orchestrator script
    └── repos.txt               # list of repos to reproduce
```

---

## Setup

**1. Create required directories**

```bash
mkdir -p workspace results
```

**2. Configure models** (optional — defaults are fine to start)

```bash
cp .env.example .env
```

Edit `.env` if you want different models:

```env
AGENT_MODEL=qwen2.5-coder:14b    # runs the experiment — needs good tool use
ANALYSIS_MODEL=llama3.2:3b       # analyses the result — lightweight is fine
```

**3. Pull images and models**

```bash
docker compose up ollama ollama-pull openhands -d
```

This pulls the OpenHands and Ollama images and downloads both models. The first run takes a while depending on your connection. Models are cached in a Docker volume so subsequent runs are instant.

**4. Verify OpenHands is up**

Open [http://localhost:3000](http://localhost:3000) — you should see the OpenHands UI.

---

## Running

### Single repo

```bash
GITHUB_REPOS=https://github.com/org/repo docker compose run orchestrator
```

### Multiple repos via env var

```bash
GITHUB_REPOS=https://github.com/org/repo1,https://github.com/org/repo2 docker compose run orchestrator
```

### Multiple repos via file (recommended for large batches)

Edit `orchestrator/repos.txt` — one URL per line, `#` for comments:

```
https://github.com/org/repo1
https://github.com/org/repo2
# https://github.com/org/skipped-for-now
```

Then just run:

```bash
docker compose run orchestrator
```

The orchestrator processes repos sequentially and prints a summary table at the end:

```
============================================================
SUMMARY
============================================================
  ✓  reproduced  https://github.com/org/repo1
  ✗  failed      https://github.com/org/repo2
               error: env
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

**Disk space** — large repos with bundled datasets can fill your disk during cloning. The orchestrator automatically retries with a shallow sparse clone if it runs out of space. You can also free Docker cache manually:

```bash
docker image prune -a
docker builder prune -a
```

**No GPU** — remove the `deploy:` block from the `ollama` service in `docker-compose.yml`. Inference will be slow but functional. Consider using smaller models (`qwen2.5-coder:7b` and `llama3.2:1b`).

**Monitoring a batch run** — tail the orchestrator logs in another terminal:

```bash
docker compose logs -f orchestrator
```

**OpenHands UI** — you can watch the agent work in real time at [http://localhost:3000](http://localhost:3000) while the orchestrator is running.

**Re-running a repo** — the orchestrator always deletes and re-clones the repo directory, so re-runs are always clean.

---

## Services

| Service | Image | Purpose |
|---|---|---|
| `ollama` | `ollama/ollama` | Serves LLMs locally on port 11434 |
| `ollama-pull` | `ollama/ollama` | One-shot model downloader (exits after pull) |
| `openhands` | `openhands:1.4` | Agentic execution — bash, file tools, sandbox |
| `orchestrator` | `python:3.12-slim` | Clones repos, drives OpenHands, calls Ollama |

---

## Stopping

```bash
docker compose down          # stop everything, keep volumes (models preserved)
docker compose down -v       # stop everything AND delete model cache
```