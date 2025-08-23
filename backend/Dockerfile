# syntax=docker/dockerfile:1
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System libs: PyMuPDF needs libstdc++, pytesseract needs tesseract-ocr
# Dodaj jezike po potrebi (eng je default; hrv = tesseract-ocr-hrv)
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr tesseract-ocr-hrv \
    libtesseract-dev \
    libstdc++6 \
 && rm -rf /var/lib/apt/lists/*

# Dependencies
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

# App code
COPY . .

# Sanity checks (optional, ali korisno za fail-fast)
RUN tesseract --version && python -c "import fitz, pytesseract; print('deps OK')"

ENV PORT=8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
