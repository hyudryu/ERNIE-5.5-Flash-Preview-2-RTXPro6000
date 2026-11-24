# vLLM 0.32.0 (unmodified wheel) + Baidu's vLLM-ERNIE plugin, pinned.
FROM vllm/vllm-openai:v0.32.0
ARG VLLM_ERNIE_REF=ba322a5
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/* \
 && git clone https://github.com/baidu/vLLM-ERNIE.git /opt/vLLM-ERNIE \
 && cd /opt/vLLM-ERNIE && git checkout ${VLLM_ERNIE_REF} \
 && (uv pip install --system --no-deps -e . || pip install --no-deps -e .)
