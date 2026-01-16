#!/bin/bash

# --- SELECT MODEL (Uncomment only one) ---
MODEL="nomic-embed-text:v1.5"
#MODEL="jina/jina-embeddings-v2-base-en:latest"
#MODEL="mxbai-embed-large:latest"
#MODEL="bge-m3:latest"
#MODEL="embeddinggemma:latest"

# --- EXECUTION ---
echo "Starting ingestion for model: $MODEL"

python3 ingest.py --model "$MODEL" --source book
python3 ingest.py --model "$MODEL" --source ati --threads 8

echo "Ingestion for $MODEL completed successfully."