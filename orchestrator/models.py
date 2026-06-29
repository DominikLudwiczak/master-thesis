from pydantic import BaseModel


class ReproductionResult(BaseModel):
    repo_url:            str
    agent_output:        str          # raw OpenHands final response
    steps_completed:     str          # description of steps the agent completed
    conversation_trace:  list[dict]   # full OpenHands message history
    outcome:             str          # free-form outcome label chosen by the analyzer
    metrics_found:       dict         # any numeric results the agent reported
    code_modifications:  str          # what code changes the agent had to make
    readme_adherence:    str          # how closely the agent followed the README
    failure_reasons:     str | None   # root-cause analysis if the run did not succeed
    success_factors:     str | None   # what contributed to success if it did
    detailed_description: str         # comprehensive narrative of the entire run
    agent_quality:       dict | None = None  # quality metrics from classify_agent_run
