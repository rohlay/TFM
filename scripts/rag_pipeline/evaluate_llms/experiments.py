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
    OUTPUT_DIR = Path("/disk1/rlaycock/rag/experiments/results")
else:
    DB_ROOT = "./sutta_vector_db"
    QA_DATA_DIR = Path("./data/qa_datasets")
    OUTPUT_DIR = Path("./experiments/results")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Hardcoded Temperature List for Experiments
TEMP_LIST = [0, 0.25, 0.5, 0.75, 1.0]

# --- RETRIEVAL FUNCTIONS ---

def get_chunk_context(query, collection, n_results=5):
    """
    Standard RAG: Retrieves the top N most relevant chunks.
    Good for specific fact retrieval but might lose narrative flow.
    """
    try:
        results = collection.query(
            query_texts=[query],
            n_results=n_results
        )
        if not results['documents']:
            return "No context found."
        
        # Combine retrieved documents into a single context string
        context = "\n\n".join(results['documents'][0])
        return context
    except Exception as e:
        logger.error(f"Chunk retrieval failed: {e}")
        return "ERROR_RETRIEVING_CHUNKS"

def get_full_sutta_context(query, collection):
    """
    Full Sutta RAG: Finds the single best matching chunk, identifies the 
    source file (e.g., 'dn01.html'), and retrieves the ENTIRE text of that sutta.
    """
    try:
        # 1. Get the single best matching chunk to identify the source
        initial_match = collection.query(query_texts=[query], n_results=1)
        
        if not initial_match['ids'] or not initial_match['metadatas'][0]:
            return "No context found."

        # 2. Extract the source filename from metadata
        source_file = initial_match['metadatas'][0][0]['source']
        
        # 3. Retrieve EVERY chunk belonging to that source file
        # We use collection.get() to fetch by metadata filter
        full_sutta_data = collection.get(
            where={"source": source_file}
        )

        if not full_sutta_data['documents']:
            return f"Error: Source file {source_file} found but no chunks retrieved."

        # 4. Sort by chunk_index to reconstruct the text in order
        # Zip metadata (contains index) with documents (text)
        combined = zip(full_sutta_data['metadatas'], full_sutta_data['documents'])
        # Sort based on 'chunk_index' key in metadata
        sorted_chunks = sorted(combined, key=lambda x: x[0]['chunk_index'])

        # 5. Join them into one giant string
        full_text = "\n".join([chunk[1] for chunk in sorted_chunks])
        
        logger.debug(f"Retrieved full sutta: {source_file} ({len(sorted_chunks)} chunks)")
        return full_text

    except Exception as e:
        logger.error(f"Sutta retrieval failed: {e}")
        return "ERROR_RETRIEVING_SUTTA"

# --- INFERENCE FUNCTION ---

def run_inference(llm_model, prompt, temp):
    """Calls Ollama API with specific temperature control."""
    try:
        # Ensure we are using the correct local host
        client = ollama.Client(host='http://localhost:11434')
        
        response = client.generate(
            model=llm_model,
            prompt=prompt,
            options={
                "temperature": temp,
                "top_k": 40,  # Standard baseline
                "top_p": 0.9,
                # "num_ctx": 8192 # Uncomment if you need to force larger context window
            }
        )
        return response['response'], response['total_duration']
    except Exception as e:
        logger.error(f"Ollama inference error: {e}")
        return "ERROR_IN_GENERATION", 0

# --- MAIN EXPERIMENT LOOP ---

def main():
    parser = argparse.ArgumentParser(description="Pali Canon LLM Experiments")
    parser.add_argument("--llm", required=True, help="Llama model tag (e.g., llama3.3:8b)")
    parser.add_argument("--embed", help="Embedding model tag (if using RAG)")
    parser.add_argument("--source", required=True, choices=['ati', 'book'], help="Data source")
    parser.add_argument("--temp", type=float, default=0.0, help="Single temp (ignored if --use_temp_list is True)")
    parser.add_argument("--use_temp_list", action="store_true", help="Iterate through [0, 0.25, 0.5, 0.75, 1.0]")
    parser.add_argument("--samples", type=int, default=1, help="N samples per question per temp")
    parser.add_argument("--rag", action="store_true", help="Enable RAG mode")
    parser.add_argument("--context_mode", choices=['chunk', 'sutta'], default='chunk', 
                        help="RAG Mode: 'chunk' (top 5) or 'sutta' (full document)")
    args = parser.parse_args()

    # 1. Load QA Dataset
    qa_filename = f"{args.source}_qa.xlsx"
    qa_path = QA_DATA_DIR / qa_filename
    if not qa_path.exists():
        logger.critical(f"QA Dataset not found at {qa_path}")
        return
    
    df_qa = pd.read_excel(qa_path)
    logger.info(f"Loaded {len(df_qa)} questions from {qa_filename}")

    # 2. Setup ChromaDB if RAG is enabled
    collection = None
    if args.rag:
        if not args.embed:
            logger.critical("RAG mode requires --embed model name!")
            return
        
        client = chromadb.PersistentClient(path=str(DB_ROOT))
        
        # Exact sanitization logic from ingest.py to find the collection
        safe_embed = args.embed.replace('/', '_').replace(':', '_').replace('-', '_')
        col_name = f"{args.source}_{safe_embed}"
        
        try:
            collection = client.get_collection(name=col_name)
            logger.info(f"RAG Connected: {col_name} | Mode: {args.context_mode.upper()}")
        except Exception as e:
            logger.critical(f"Could not find collection '{col_name}'. Check your ingestion names! Error: {e}")
            return

    # 3. Determine Temperatures
    temps_to_run = TEMP_LIST if args.use_temp_list else [args.temp]

    # 4. Execution Loop
    for t in temps_to_run:
        for n in range(1, args.samples + 1):
            results = []
            logger.info(f"RUNNING: Model={args.llm} | Temp={t} | Round={n}/{args.samples} | RAG={args.rag}")

            for idx, row in df_qa.iterrows():
                question = row['question']
                final_prompt = question
                retrieved_text = "N/A"
                
                # RAG Logic
                if args.rag and collection:
                    if args.context_mode == 'sutta':
                        retrieved_text = get_full_sutta_context(question, collection)
                    else:
                        retrieved_text = get_chunk_context(question, collection)
                    
                    final_prompt = f"Context:\n{retrieved_text}\n\nQuestion: {question}\nAnswer based on the context provided:"

                # Inference
                answer, duration = run_inference(args.llm, final_prompt, t)

                print("\n" + "="*80)
                print(f"ROUND: {n}/{args.samples} | TEMP: {t} | MODEL: {args.llm}")
                print(f"QUESTION: {question}")
                print("-" * 40)
                print(f"RETRIEVED CONTEXT (First 300 chars):\n{retrieved_text[:300]}...")
                print("-" * 40)
                print(f"LLM RESPONSE:\n{answer}")
                print("="*80 + "\n")
                
                results.append({
                    "question_id": idx,
                    "question": question,
                    "generated_answer": answer,
                    "duration_ns": duration,
                    "temperature": t,
                    "sample_round": n,
                    "llm_model": args.llm,
                    "rag_mode": args.context_mode if args.rag else "None",
                    "retrieved_context": retrieved_text
                })

            # 5. Save Results (One file per round to prevent data loss)
            rag_tag = f"RAG_{args.context_mode.upper()}" if args.rag else "NO_RAG"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            # Filename: model_source_mode_temp_round_time.xlsx
            safe_llm = args.llm.replace(':', '_')
            filename = f"{safe_llm}_{args.source}_{rag_tag}_T{t}_R{n}_{timestamp}.xlsx"
            
            out_path = OUTPUT_DIR / filename
            pd.DataFrame(results).to_excel(out_path, index=False)
            logger.success(f"SAVED: {out_path}")

if __name__ == "__main__":
    main()