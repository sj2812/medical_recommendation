# Medical Students Learning Assistant: Streamlit app + Neo4j KG
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY src/ ./src/
COPY app.py ./
COPY data/ ./data/
COPY notebooks/ ./notebooks/
COPY docker-entrypoint.sh ./
RUN chmod +x docker-entrypoint.sh

# Ensure dirs exist (models for ML recommender, data for mounts)
RUN mkdir -p /app/data /app/models /app/notebooks

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Streamlit
EXPOSE 8501

# Entrypoint waits for Neo4j (if configured) and optionally builds KG; then runs CMD
ENTRYPOINT ["./docker-entrypoint.sh"]
# Default: run Streamlit app (bind to 0.0.0.0 for Docker)
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
