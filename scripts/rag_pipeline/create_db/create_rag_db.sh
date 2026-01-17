#!/bin/bash

# Nomic Embed Text v1.5
python3 ingest.py --model nomic-embed-text:v1.5 --source ati
python3 ingest.py --model nomic-embed-text:v1.5 --source book

# Jina Embeddings v2 (Base English)
python3 ingest.py --model jina/jina-embeddings-v2-base-en:latest --source ati
python3 ingest.py --model jina/jina-embeddings-v2-base-en:latest --source book

# MXBAI Embed Large
python3 ingest.py --model mxbai-embed-large:latest --source ati
python3 ingest.py --model mxbai-embed-large:latest --source book

# BGE-M3 (Multilingual)
python3 ingest.py --model bge-m3:latest --source ati
python3 ingest.py --model bge-m3:latest --source book

echo "All valid embedding models ingested successfully."