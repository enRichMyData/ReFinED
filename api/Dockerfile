# uses python 3.9 for ReFinED to work
FROM python:3.9-slim

# system-level build tools for heavy libraries (e.g. torch, numpy)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# working directory in container
WORKDIR /app

# requirements for both the API and ReFinED
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# copy model, api, test -logic
COPY ./src ./src
COPY ./api ./api

# replicates local 'export PYTHONPATH' required to run both api and refined
# - /app/api    (so 'from app.main' works)
# - /app/src    (so 'import refined' works)
ENV PYTHONPATH="/app/api:/app/src"

# exposes 8002 as port
EXPOSE 8002

# run from app directory
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${FASTAPI_SERVER_PORT:-8002} --app-dir api"]
