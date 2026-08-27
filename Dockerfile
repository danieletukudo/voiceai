FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl build-essential && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api.py .
COPY business_ingest.py .
COPY ingest_document.py .
COPY maps_profile.py .
COPY website_profile.py .
COPY rag_config.py .
COPY ask_document.py .
COPY voice_agent.py .
COPY static/ ./static/
COPY docs/ ./docs/

RUN mkdir -p docs transcripts

ENV API_HOST=0.0.0.0
ENV PORT=8000
ENV PYTORCH_ENABLE_MPS_FALLBACK=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD curl -f http://127.0.0.1:${PORT}/health || exit 1

CMD ["python", "api.py"]
