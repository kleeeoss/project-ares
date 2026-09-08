FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-compute.txt .
RUN pip install --no-cache-dir -r requirements-compute.txt

COPY compute_node.py .

EXPOSE 8000
CMD ["uvicorn", "compute_node:app", "--host", "0.0.0.0", "--port", "8000"]
