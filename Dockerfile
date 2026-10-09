FROM python:3.12-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY docmind ./docmind
COPY ui ./ui
RUN useradd -m app && mkdir -p data && chown -R app /app
USER app
EXPOSE 8000
CMD ["uvicorn", "docmind.api:app", "--host", "0.0.0.0", "--port", "8000"]
