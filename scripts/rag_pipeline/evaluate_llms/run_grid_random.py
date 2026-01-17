import os
import argparse
import pandas as pd
import ollama
import chromadb
from pathlib import Path
from loguru import logger
from datetime import datetime

# --- CONFIG & PATHS ---
SERVER = True
if SERVER:
    DB_ROOT = "/disk1/rlaycock/rag/rag_db/sutta_vector_db"
    QA_DATA_DIR = Path("/disk1/rlaycock/rag/data/qa_datasets")
    OUTPUT_DIR = Path("/disk1/rlaycock/rag/experiments/results_grid")
else:
    DB_ROOT = "./sutta_vector_db"
    QA_DATA_DIR = Path("./data/qa_datasets")
    OUTPUT_DIR = Path("./experiments/results_grid")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- THE GRID LISTS (5 Values Each) ---
# Selected for "Continuous" analysis from strict to loose
TEMP_LIST = [0.0, 0.25, 0.5, 0.75, 1.0]
TOP_P_LIST = [0.1, 0.3, 0.5, 0.7, 0.9]
TOP_K_LIST = [1, 10, 20, 40, 60]

# --- RETRIEVAL FUNCTIONS ---

def get_chunk_context(query, collection, n_results=5):
    """Standard RAG: Retrieves top N chunks."""
    try:
        results = collection.query(query_texts=[query], n_results=n_results)
        if not results['documents']: return "No context found."
        return "\n\n".join(results['documents'][0])
    except Exception as e:
        logger.error(f"Chunk retrieval failed: {e}")
        return "ERROR_RETRIEVING_CHUNKS"

def get_full_sutta_context(query, collection):
    """Full Sutta RAG: Retrieves the entire document of the best match."""
    try:
        initial_match = collection.query(query_texts=[query], n_results=1)
        if not initial_match['ids'] or not initial_match['metadatas'][0]:
            return "No context found."

        source_file = initial_match['metadatas'][0][0]['source']
        
        full_sutta_data = collection.get(where={"source": source_file})
        if not full_sutta_data['documents']: return "Error: Source found but no chunks."

        # Sort chunks by index to reconstruct text flow
        combined = zip(full_sutta_data['metadatas'], full_sutta_data['documents'])
        sorted_chunks = sorted(combined, key=lambda x: x[0]['chunk_index'])

        return "\n".join([chunk[1] for chunk in sorted_chunks])
    except Exception as e:
        logger.error(f"Sutta retrieval failed: {e}")
        return "ERROR_RETRIEVING_SUTTA"

# --- INFERENCE FUNCTION (GRID AWARE) ---

def run_inference(llm_model, prompt, temp, top_p, top_k):
    """Calls Ollama with ALL grid parameters."""
    try:
        # Client initialized inside to ensure freshness per call
        client = ollama.Client(host='http://localhost:11434')
        
        response = client.generate(
            model=llm_model,
            prompt=prompt,
            options={
                "temperature": temp,
                "top_p": top_p,
                "top_k": top_k,
                # "num_ctx": 8192 # Uncomment if dealing with huge contexts
            }
        )
        return response['response'], response['total_duration']
    except Exception as e:
        logger.error(f"Ollama inference error: {e}")
        return "ERROR_IN_GENERATION", 0

# --- MAIN GRID LOOP ---

def main():
    parser = argparse.ArgumentParser(description="Pali Canon LLM GRID Experiments")
    parser.add_argument("--llm", required=True, help="Llama model tag (e.g., llama3.3:8b)")
    parser.add_argument("--embed", help="Embedding model tag (if using RAG)")
    parser.add_argument("--source", required=True, choices=['ati', 'book'], help="Data source")
    parser.add_argument("--samples", type=int, default=1, help="N samples per grid configuration")
    parser.add_argument("--rag", action="store_true", help="Enable RAG mode")
    parser.add_argument("--context_mode", choices=['chunk', 'sutta'], default='chunk', 
                        help="RAG Mode: 'chunk' (top 5) or 'sutta' (full document)")
    args = parser.parse_args()

    # 1. Load Data
    qa_path = QA_DATA_DIR / f"{args.source}_qa.xlsx"
    if not qa_path.exists():
        logger.critical(f"QA Dataset not found at {qa_path}")
        return
    df_qa = pd.read_excel(qa_path)
    logger.info(f"Loaded {len(df_qa)} questions. Starting GRID Search.")

    # 2. Setup ChromaDB (only if RAG)
    collection = None
    if args.rag:
        if not args.embed:
            logger.critical("RAG mode requires --embed model name!")
            return
        client = chromadb.PersistentClient(path=str(DB_ROOT))
        safe_embed = args.embed.replace('/', '_').replace(':', '_').replace('-', '_').replace('.', '_')
        col_name = f"{args.source}_{safe_embed}"
        try:
            collection = client.get_collection(name=col_name)
            logger.info(f"RAG Connected: {col_name} | Mode: {args.context_mode}")
        except Exception as e:
            logger.critical(f"Collection '{col_name}' not found! Error: {e}")
            return

    # 3. THE GRID LOOPS
    # Total configs = 5 (K) * 5 (P) * 5 (T) = 125 * Samples
    total_combinations = len(TOP_K_LIST) * len(TOP_P_LIST) * len(TEMP_LIST)
    current_config = 0

    for k in TOP_K_LIST:
        for p in TOP_P_LIST:
            for t in TEMP_LIST:
                current_config += 1
                
                # REPETITIONS (N Samples)
                for n in range(1, args.samples + 1):
                    results = []
                    logger.info(f"GRID [{current_config}/{total_combinations}] | K={k} P={p} T={t} | Round={n}")

                    for idx, row in df_qa.iterrows():
                        question = row['question']
                        final_prompt = question
                        retrieved_text = "N/A"

                        # RAG Context Retrieval
                        if args.rag and collection:
                            if args.context_mode == 'sutta':
                                retrieved_text = get_full_sutta_context(question, collection)
                            else:
                                retrieved_text = get_chunk_context(question, collection)
                            
                            final_prompt = f"Context:\n{retrieved_text}\n\nQuestion: {question}\nAnswer based on context:"

                        # Run Inference with current Grid Params
                        answer, duration = run_inference(args.llm, final_prompt, t, p, k)

                        results.append({
                            "question_id": idx,
                            "question": question,
                            "generated_answer": answer,
                            "temp": t,
                            "top_p": p,
                            "top_k": k,
                            "round_n": n,
                            "duration_ns": duration,
                            "rag_mode": args.context_mode if args.rag else "None",
                            "retrieved_context": retrieved_text
                        })

                    # SAVE RESULT IMMEDIATELY
                    # Filename: llm_source_T_P_K_Round.xlsx
                    safe_llm = args.llm.replace(':', '_')
                    fname = f"{safe_llm}_{args.source}_T{t}_P{p}_K{k}_Round{n}.xlsx"
                    
                    # Create subfolder for cleanliness? Optional, keeping flat for now.
                    out_path = OUTPUT_DIR / fname
                    pd.DataFrame(results).to_excel(out_path, index=False)
                    logger.success(f"Saved: {fname}")

if __name__ == "__main__":
    main()