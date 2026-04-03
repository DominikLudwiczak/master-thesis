FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y git curl && rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip install -r requirements.txt --no-cache-dir

CMD ["python", "main.py"]