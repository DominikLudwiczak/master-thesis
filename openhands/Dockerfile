FROM docker.openhands.dev/openhands/openhands:1.4

RUN sed -i 's/stop_after_delay(120)/stop_after_delay(600)/' \
    /app/openhands/runtime/impl/docker/docker_runtime.py

# Remove security_risk from required parameters — Ollama models don't generate it
RUN sed -i "s/'required': \['command', 'security_risk'\]/'required': ['command']/" \
    /app/openhands/agenthub/codeact_agent/tools/bash.py && \
    sed -i "s/'required': \['code', 'security_risk'\]/'required': ['code']/" \
    /app/openhands/agenthub/codeact_agent/tools/ipython.py && \
    sed -i "s/'required': \['command', 'path', 'security_risk'\]/'required': ['command', 'path']/" \
    /app/openhands/agenthub/codeact_agent/tools/str_replace_editor.py && \
    sed -i "s/'required': \['path', 'content', 'security_risk'\]/'required': ['path', 'content']/" \
    /app/openhands/agenthub/codeact_agent/tools/llm_based_edit.py && \
    sed -i "s/'required': \['code', 'security_risk'\]/'required': ['code']/" \
    /app/openhands/agenthub/codeact_agent/tools/browser.py

# Remove enum validation for security_risk — Ollama models send wrong values
RUN sed -i "/'enum': RISK_LEVELS/d" \
    /app/openhands/agenthub/codeact_agent/tools/bash.py \
    /app/openhands/agenthub/codeact_agent/tools/ipython.py \
    /app/openhands/agenthub/codeact_agent/tools/str_replace_editor.py \
    /app/openhands/agenthub/codeact_agent/tools/browser.py \
    /app/openhands/agenthub/codeact_agent/tools/llm_based_edit.py

# Remove enum validation for is_input — Ollama models send 'True'/'False' instead of 'true'/'false'
RUN sed -i "/'enum': \['true', 'false'\]/d" \
    /app/openhands/agenthub/codeact_agent/tools/bash.py

# Condenser config — use observation masking to save context window
# Keeps last N event outputs visible, masks older ones with <MASKED>
ARG OPENHANDS_ATTENTION_WINDOW=25
RUN printf "[core]\nenable_default_condenser = false\n\n[condenser]\ntype = \"observation_masking\"\nattention_window = ${OPENHANDS_ATTENTION_WINDOW}\n" > /app/config.toml

# Max chars per single observation message.
# Configurable via build arg; default 30000 matches upstream OpenHands default.
ARG OPENHANDS_MAX_MESSAGE_CHARS=30000
RUN sed -i "s/default=30_000/default=${OPENHANDS_MAX_MESSAGE_CHARS}/" \
    /app/openhands/core/config/llm_config.py

# Replace default 50/50 truncation with asymmetric head+mid+tail split
# (1000 head + 2000 middle + 4000 tail) — preserves errors at the end
COPY patch_truncation.py /tmp/
RUN python3 /tmp/patch_truncation.py