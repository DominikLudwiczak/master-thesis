from pydantic import BaseModel


class ReproductionResult(BaseModel):
    repo_url:            str
    agent_output:        str          # raw OpenHands final response
    steps_completed:     list[str]    # steps the agent reported completing
    conversation_trace:  list[dict]   # full OpenHands message history
    verdict:             str          # "reproduced" | "failed" | "partial"
    error_type:          str | None   # "env" | "code" | "data" | "timeout" | None
    metrics_found:       dict         # any numeric results the agent reported
    analysis:            str          # Ollama's natural-language verdict
