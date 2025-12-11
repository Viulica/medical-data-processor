# syntax=docker/dockerfile:1
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps: C++ runtime + Tesseract (hrv po potrebi)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libstdc++6 libgcc-s1 \
    tesseract-ocr tesseract-ocr-hrv libtesseract-dev \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --timeout=1000 --retries=5 -r requirements.txt

COPY . .

# Sanity check (opc.)
RUN tesseract --version && python -c "import fitz, pytesseract; print('deps OK')"

# Važno: koristi $PORT (fallback 8000). Shell-form da se varijabla expanda.
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
