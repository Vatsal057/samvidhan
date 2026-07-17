# Streamlit app image. Ingestion of the sample corpus happens at build time so
# the container is demo-ready out of the box (no HF token needed for retrieval).
FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

COPY requirements.txt pyproject.toml ./
RUN pip install -r requirements.txt

COPY . .
RUN pip install -e .

# Build the vector store from the bundled sample so the image runs offline.
RUN python -m samvidhan.ingest --text data/sample

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
