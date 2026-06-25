FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the model at build time: startup is instant, no cold-pull on first request.
RUN python -c "from transformers import AutoTokenizer, AutoModelForSequenceClassification; \
    AutoTokenizer.from_pretrained('typeform/distilbert-base-uncased-mnli'); \
    AutoModelForSequenceClassification.from_pretrained('typeform/distilbert-base-uncased-mnli')"

COPY server.py .

EXPOSE 9090
HEALTHCHECK --interval=10s --timeout=5s --retries=12 --start-period=30s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:9090/healthz')" 2>/dev/null || exit 1

CMD ["python", "server.py"]
