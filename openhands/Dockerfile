FROM docker.openhands.dev/openhands/openhands:1.4

RUN sed -i 's/stop_after_delay(120)/stop_after_delay(600)/' \
    /app/openhands/runtime/impl/docker/docker_runtime.py

# Remove security_risk from required parameters — Ollama models don't generate it
RUN sed -i "s/'required': \['command', 'security_risk'\]/'required': ['command']/" \
    /app/openhands/agenthub/codeact_agent/tools/bash.py && \
    sed -i "s/'required': \['code', 'security_risk'\]/'required': ['code']/" \
    /app/openhands/agenthub/codeact_agent/tools/ipython.py && \
    sed -i "s/'required': \['command', 'path', 'security_risk'\]/'required': ['command', 'path']/" \
    /app/openhands/agenthub/codeact_agent/tools/str_replace_editor.py