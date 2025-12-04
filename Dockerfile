FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_ENABLE_ONEDNN_OPTS=0

WORKDIR /app

# Backend bağımlılıkları
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

# Kod ve model dosyası
COPY backend /app/backend
COPY service /app/service

# Artık backend dizininde çalış
WORKDIR /app/backend

EXPOSE 8000

# serve.py içinde FastAPI instance'ının adı "app" ise:
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]
