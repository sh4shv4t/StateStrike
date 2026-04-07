FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN touch /app/statestrike.db

COPY scripts/start.sh /start.sh
RUN chmod +x /start.sh

EXPOSE 7860

HEALTHCHECK --interval=10s --timeout=5s --start-period=20s --retries=5 \
  CMD curl -f http://localhost:7860/health || exit 1

CMD ["/start.sh"]
