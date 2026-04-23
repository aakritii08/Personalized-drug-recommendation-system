# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# copy project files
COPY . /app

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# create directories
RUN mkdir -p /app/models /app/data

# Expose port
EXPOSE 8501

CMD ["gunicorn", "--bind", "0.0.0.0:8501", "app.app:app", "--workers", "2"]
