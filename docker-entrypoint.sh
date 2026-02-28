#!/usr/bin/env bash
set -e

# Wait for Neo4j to be ready when NEO4J_URI is set
wait_for_neo4j() {
  local uri="${NEO4J_URI:-}"
  [ -z "$uri" ] && return 0
  local user="${NEO4J_USER:-neo4j}"
  local pass="${NEO4J_PASSWORD:-password}"
  local max_attempts=60
  local interval=3
  echo "Waiting for Neo4j at $uri (up to $((max_attempts * interval))s) ..."
  for i in $(seq 1 $max_attempts); do
    if python -c "
from neo4j import GraphDatabase
import os
uri = os.environ.get('NEO4J_URI')
user = os.environ.get('NEO4J_USER', 'neo4j')
password = os.environ.get('NEO4J_PASSWORD', 'password')
try:
    d = GraphDatabase.driver(uri, auth=(user, password))
    d.verify_connectivity()
    d.close()
    exit(0)
except Exception:
    exit(1)
" 2>/dev/null; then
      echo "Neo4j is ready."
      return 0
    fi
    [ $((i % 10)) -eq 0 ] && echo "  still waiting (${i}/${max_attempts}) ..."
    sleep $interval
  done
  echo "Neo4j did not become ready in time. Continuing without KG." >&2
  return 1
}

# Optional: build KG if Neo4j is available and train data exists
if [ -n "${NEO4J_URI}" ]; then
  if wait_for_neo4j; then
    if [ -f /app/data/train_data/train_data_per_user_80_20.csv ] && [ -f /app/data/ds_test_case_qbank_questions.csv ]; then
      echo "Building knowledge graph ..."
      python -m src.kg_build || true
    fi
  fi
fi

# Run the CMD (default: Streamlit)
exec "$@"
