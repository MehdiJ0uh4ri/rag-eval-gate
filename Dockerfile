FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /srv

# Runtime deps only. The eval stack (ragas, sentence-transformers, torch) is
# ~2GB and has no business in the served image -- evaluation runs in CI, not
# in the pod.
COPY requirements.txt .
RUN pip install -q -r requirements.txt

COPY app ./app

RUN useradd --uid 10001 --no-create-home --shell /usr/sbin/nologin rag
USER 10001

EXPOSE 8080
HEALTHCHECK --interval=10s --timeout=3s --retries=5 \
  CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8080/healthz').read()"

CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8080"]
