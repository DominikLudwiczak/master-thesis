from pydantic import BaseModel


class ReproductionResult(BaseModel):
    repo_url:            str
    agent_output:        str          # raw OpenHands final response
    steps_completed:     list[str]    # steps the agent reported completing
    conversation_trace:  list[dict]   # full OpenHands message history
    verdict:             str          # "reproduced" | "timeout_running" | "partial" | "failed" | "prereq_missing" | "inconclusive"
    error_type:          str | None   # "env" | "code" | "data" | "timeout" | "agent" | "prereq" | None
    metrics_found:       dict         # any numeric results the agent reported
    code_modifications:  str          # what code changes the agent had to make
    failure_reason:      str | None   # specific error/blocker if not reproduced
    analysis:            str          # Ollama's natural-language verdict
    agent_quality:       dict | None = None  # quality metrics from classify_agent_run
