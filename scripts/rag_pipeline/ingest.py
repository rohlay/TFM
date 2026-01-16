import ollama
import chromadb
import argparse
import pandas as pd
import sys
from bs4 import BeautifulSoup
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from loguru import logger

# Library-based chunking
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- LOGGER CONFIG ---
# Remove default handler and add a clean, colorized one
#logger.remove()
#logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>", level="INFO")
# Optional: Log to a file to keep a permanent record of the ingestion
logger.add("ingestion.log", rotation="10 MB", level="DEBUG")

# --- CONFIG ---
CHUNK_SIZE = 512
CHUNK_OVERLAP = 100 # 20%
SERVER = True
if SERVER:
    ATI_BASE_PATH = Path("/disk1/rlaycock/rag/data/ati/tipitaka")
    ATI_EXCEL_PATH = Path("/disk1/rlaycock/rag/data/ati_index_metadata.xlsx")
    MD_FILE_PATH = Path("/disk1/rlaycock/rag/data/wbt.md")
    DB_ROOT = Path("/disk1/rlaycock/rag/rag_db/sutta_vector_db")
else:
    ATI_BASE_PATH = Path(r"C:\Users\rohan\ws\git\TFM\data\data-tipitaka\src")
    ATI_EXCEL_PATH = r"C:\Users\rohan\ws\git\TFM\data\data-tipitaka\database\ati_index_metadata.xlsx"
    MD_FILE_PATH = Path(r"C:\Users\rohan\ws\LFS_local\TFM-data\what_the_buddha_taught.md")
    DB_ROOT = "./sutta_vector_db"


# Initialize Ollama Client
client = ollama.Client(host='http://localhost:11434')

text_chunker = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""]
)

# --- WORKERS ---

def process_markdown(md_path, model_name, collection):
    if not md_path.exists():
        logger.error(f"Markdown file not found at {md_path}")
        return

    try:
        logger.debug(f"Reading book from {md_path}")
        raw_text = md_path.read_text(encoding='utf-8')
        chunks = text_chunker.split_text(raw_text)
        
        response = client.embed(model=model_name, input=chunks)
        embeddings = response['embeddings']

        collection.add(
            ids=[f"wtbt_c{i}" for i in range(len(chunks))],
            embeddings=embeddings,
            documents=chunks,
            metadatas=[{"source": "What The Buddha Taught", "type": "book", "chunk_index": i} for i in range(len(chunks))]
        )
        logger.success(f"Successfully indexed Book: {len(chunks)} chunks")
    except Exception as e:
        logger.exception(f"Critical error processing book") # Capture full stack trace

def clean_pali_text(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    content = soup.find('div', id='H_content')
    if not content: return None
    for script in content(["script", "style"]): script.extract()
    return content.get_text(separator="\n", strip=True)

def process_ati_row(row, model_name, collection):
    filename = row['Filename']
    nikaya_dir = ATI_BASE_PATH / row['Nikaya'].lower()
    file_path = list(nikaya_dir.rglob(filename))

    if not file_path:
        logger.warning(f"File missing: {filename}")
        return

    try:
        with open(file_path[0], 'r', encoding='utf-8', errors='ignore') as f:
            raw_text = clean_pali_text(f.read())
        
        if not raw_text:
            logger.debug(f"Skipping empty file: {filename}")
            return

        chunks = text_chunker.split_text(raw_text)
        response = client.embed(model=model_name, input=chunks)
        
        collection.add(
            ids=[f"{filename}_c{i}" for i in range(len(chunks))],
            embeddings=response['embeddings'],
            documents=chunks,
            metadatas=[{
                "source": filename, 
                "nikaya": row['Nikaya'], 
                "author": row.get('Author', 'Unknown'),
                "chunk_index": i
            } for i in range(len(chunks))]
        )
        logger.info(f"Indexed {filename}: {len(chunks)} chunks")
    except Exception:
        logger.exception(f"Error indexing {filename}")

# --- ORCHESTRATION ---

def process_ati_source(df, model_name, collection, threads):
    if threads > 1:
        logger.info(f"Parallel threads: {threads}") # TODO: Multithreading not tested
        with ThreadPoolExecutor(max_workers=threads) as executor:
            futures = [executor.submit(process_ati_row, row, model_name, collection) for _, row in df.iterrows()]
            for future in as_completed(futures):
                future.result()
    else:
        logger.info("Single thread")
        for _, row in df.iterrows():
            process_ati_row(row, model_name, collection)

def main():
    parser = argparse.ArgumentParser(description="Pali Canon RAG Ingestion")
    parser.add_argument("--model", required=True, help="Embedding model name in Ollama")
    parser.add_argument("--source", required=True, choices=['ati', 'book'], help="Data source")
    parser.add_argument("--threads", type=int, default=1, help="Parallel workers")
    args = parser.parse_args()

    client = chromadb.PersistentClient(path=DB_ROOT)
    safe_name = args.model.replace('/', '_').replace(':', '_').replace('-', '_').replace('.', '_')
    col_name = f"{args.source}_{safe_name}"
    collection = client.get_or_create_collection(name=col_name, metadata={"hnsw:space": "cosine"})

    logger.info(f"STARTING INGESTION | Source: {args.source} | Model: {args.model}")

    if args.source == 'ati':
        df = pd.read_excel(ATI_EXCEL_PATH)
        process_ati_source(df, args.model, collection, args.threads)
    else:
        process_markdown(MD_FILE_PATH, args.model, collection)

    logger.success(f"INGESTION COMPLETE | Collection: {col_name}")

if __name__ == "__main__":
    main()