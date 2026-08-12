FROM nvcr.io/nvidia/pytorch:24.01-py3

WORKDIR /app/HydraServe

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Model volume
VOLUME ["/models"]

# Expose API port
EXPOSE 8000

# Default: PD disaggregated mode
ENV HYDRA_MODE=pd_disaggregated
ENV HYDRA_MODEL=/models/Qwen3.5-9B-AWQ
ENV HYDRA_PREFILL_GPU=0
ENV HYDRA_DECODE_GPU=1

CMD ["python", "-m", "hydraserve.serve.serve", \
     "--model", "${HYDRA_MODEL}", \
     "--mode", "${HYDRA_MODE}", \
     "--prefill-gpu", "${HYDRA_PREFILL_GPU}", \
     "--decode-gpu", "${HYDRA_DECODE_GPU}"]
