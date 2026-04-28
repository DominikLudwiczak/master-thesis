FROM docker.openhands.dev/openhands/openhands:1.4

RUN sed -i 's/stop_after_delay(120)/stop_after_delay(600)/' \
    /app/openhands/runtime/impl/docker/docker_runtime.py