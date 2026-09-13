FROM python:3.12-slim

WORKDIR /app

# System libraries required by non-headless opencv-python (a transitive
# ultralytics dependency, imported at module load time) — python:3.12-slim
# does not include these by default.
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Install the already-pinned torch/torchvision as CPU-only wheels first —
# PyPI's default wheels bundle CUDA runtime deps, which are unnecessary
# and multi-GB here. pip treats these as satisfied for the exact pinned
# versions below, so it won't re-resolve them from the default index.
RUN pip install --no-cache-dir torch==2.8.0 torchvision==0.23.0 \
      --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ app/
COPY part_b/ part_b/
COPY weights/ weights/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
