FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y build-essential curl && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY compute_node.py .
COPY coordinator.py .
COPY rag_pipeline.py .

EXPOSE 8000
EXPOSE 9000

CMD ["uvicorn", "compute_node:app", "--host", "0.0.0.0", "--port", "8000"]