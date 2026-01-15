@echo off
:: Nomic-Embed-Text-v1.5
python ingest.py --model nomic-embed-text --source ati --threads 4
python ingest.py --model nomic-embed-text --source book

:: MXBAI-Embed-Large-V2
python ingest.py --model mxbai-embed-large --source ati --threads 4
python ingest.py --model mxbai-embed-large --source book

:: BGE-M3
python ingest.py --model bge-m3 --source ati --threads 4
python ingest.py --model bge-m3 --source book

:: Qwen3-Embedding (0.6B)
python ingest.py --model qwen3-embedding --source ati --threads 4
python ingest.py --model qwen3-embedding --source book

:: Jina-Embedding-v4 (4B)
:: Note: This is a heavy 4B model; keeping threads low is safer for VRAM
python ingest.py --model jina-v4 --source ati --threads 2
python ingest.py --model jina-v4 --source book

:: Gemma-2-9B-IT (As an embedding model)
python ingest.py --model gemma2 --source ati --threads 2
python ingest.py --model gemma2 --source book

echo All models ingested successfully.
pause