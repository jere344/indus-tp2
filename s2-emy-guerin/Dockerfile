FROM python:3.14-slim
WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY model/modele.joblib ./model/
COPY app.py ./
RUN useradd -u 10001 appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

CMD ["gunicorn", "-b", "0.0.0.0:8000", "app:app"]
