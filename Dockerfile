FROM python:3.12-slim

WORKDIR /app

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
