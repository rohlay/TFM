#!/bin/bash

# Nomic Embed Text v1.5
python3 ingest.py --model nomic-embed-text:v1.5 --source book
python3 ingest.py --model nomic-embed-text:v1.5 --source ati

echo "All valid embedding models ingested successfully."