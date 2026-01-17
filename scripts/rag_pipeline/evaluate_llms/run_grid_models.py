import os
import argparse
import pandas as pd
import ollama
import chromadb
from pathlib import Path
from loguru import logger
from datetime import datetime

# --- CONFIGURATION ---
SERVER = True
if SERVER:
    DB_ROOT = Path("/disk1/rlaycock/rag/rag_db/sutta_vector_db")
    QA_DATA_DIR = Path("/disk1/rlaycock/rag/data/qa_datasets")
    OUTPUT_DIR = Path("/disk1/rlaycock/rag/experiments/final_grid_results")
else:
    DB_ROOT = Path("./sutta_vector_db")
    QA_DATA_DIR = Path("./data/qa_datasets")
    OUTPUT_DIR = Path("./experiments/final_grid_results")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- THE GRID ---
LLM_MODELS = ["llama3.2:3b", "llama3.1:8b", "llama3.3:70b"]
RAG_MODELS = ["nomic-embed-text:v1.5", "bge-m3:latest", "None"] # 'None' = No RAG
TEMP_LIST = [0.0, 0.5, 1.0]

# --- DATASET CONFIGURATION ---
# Now simplified because you pre-processed the Excel files
DATASET_CONFIG = {
    "buddha_taught_qa.xlsx": {  # Ensure this matches your dummy filename
        "db_source": "book",          
        "prompt_instr": "Answer in exactly 1-2 sentences. Be concise.",
        "max_tokens": 80,
        "retrieval_mode": "chunk"
    },
    "PaliCanon_QA_Cited_Only.xlsx": { # Ensure this matches your dummy filename
        "db_source": "ati",           
        "prompt_instr": "Provide a detailed explanation (1-2 paragraphs).",
        "max_tokens": 512,
        "retrieval_mode": "sutta"
    }
}

# --- DATA LOADER HELPER ---

def load_qa_dataset(filepath):
    """
    Simple loader. Expects 'question' and optional 'answer' columns.
    """
    try:
        # 1. Read File
        if str(filepath).endswith('.csv'):
            df = pd.read_csv(filepath)
        else:
            df = pd.read_excel(filepath)
        
        # 2. Check for required column
        if 'question' not in df.columns:
            logger.error(f"Column 'question' missing in {filepath.name}")
            return pd.DataFrame()
            
        # 3. Clean and return
        # Keep 'answer' (Ground Truth) if it exists, otherwise just question
        cols_to_keep = ['question']
        if 'answer' in df.columns:
            cols_to_keep.append('answer')
            
        return df[cols_to_keep].dropna(subset=['question'])

    except Exception as e:
        logger.error(f"Failed to load dataset {filepath.name}: {e}")
        return pd.DataFrame()

# --- RETRIEVAL LOGIC ---

def get_rag_context(query, collection, mode):
    """Retrieves context based on the dataset mode (Chunk vs Full Sutta)."""
    try:
        # 1. Standard Chunk Retrieval (For Book)
        if mode == "chunk":
            results = collection.query(query_texts=[query], n_results=5)
            if not results['documents']: return ""
            return "\n\n".join(results['documents'][0])

        # 2. Full Sutta Retrieval (For ATI)
        elif mode == "sutta":
            initial_match = collection.query(query_texts=[query], n_results=1)
            if not initial_match['ids'] or not initial_match['metadatas'][0]: return ""
            
            source_file = initial_match['metadatas'][0][0]['source']
            
            full_data = collection.get(where={"source": source_file})
            if not full_data['documents']: return ""

            combined = zip(full_data['metadatas'], full_data['documents'])
            sorted_chunks = sorted(combined, key=lambda x: x[0]['chunk_index'])
            return "\n".join([chunk[1] for chunk in sorted_chunks])
            
    except Exception as e:
        logger.error(f"Retrieval Error ({mode}): {e}")
        return ""
    return ""

# --- INFERENCE ---

def run_inference(llm, prompt, temp, max_tokens):
    try:
        client = ollama.Client(host='http://localhost:11434')
        response = client.generate(
            model=llm,
            prompt=prompt,
            options={
                "temperature": temp,
                "num_predict": max_tokens, 
            }
        )
        return response['response'], response['total_duration']
    except Exception as e:
        logger.error(f"Inference Error: {e}")
        return "ERROR", 0

# --- MAIN EXPERIMENT LOOP ---

def main():
    logger.info("STARTING DUMMY GRID EXPERIMENT")
    chroma_client = chromadb.PersistentClient(path=str(DB_ROOT))

    # 1. Loop through Configured Datasets
    for filename, config in DATASET_CONFIG.items():
        qa_path = QA_DATA_DIR / filename
        
        # Load using the simplified function
        df_qa = load_qa_dataset(qa_path)
        
        if df_qa.empty:
            logger.warning(f"Skipping {filename}: Dataset empty or not found.")
            continue

        logger.info(f"Loaded {len(df_qa)} questions from {filename}")

        # 2. Loop through RAG Models
        for rag_model in RAG_MODELS:
            
            collection = None
            if rag_model != "None":
                safe_embed = rag_model.replace('/', '_').replace(':', '_').replace('-', '_').replace('.', '_')
                col_name = f"{config['db_source']}_{safe_embed}"
                try:
                    collection = chroma_client.get_collection(name=col_name)
                    logger.info(f"Connected to DB: {col_name}")
                except Exception:
                    logger.error(f"DB Collection {col_name} not found! Skipping this RAG config.")
                    continue

            # 3. Loop through LLM Models
            for llm_model in LLM_MODELS:
                
                # Naming for output file
                rag_label = "NO_RAG" if rag_model == "None" else rag_model.replace(':', '_')
                llm_label = llm_model.replace(':', '_')
                # Use a shorter clean name for the dataset in the filename
                dataset_clean_name = "se_qa" if "PaliCanon" in filename else "wbt_qa"
                
                output_filename = f"{dataset_clean_name}_{llm_label}_{rag_label}.xlsx"
                output_path = OUTPUT_DIR / output_filename
                
                if output_path.exists():
                    logger.warning(f"Skipping {output_filename}: Already exists.")
                    continue

                logger.info(f"RUNNING: Dataset={dataset_clean_name} | LLM={llm_model} | RAG={rag_model}")
                all_results = []

                # 4. Loop through Temperatures
                for temp in TEMP_LIST:
                    for idx, row in df_qa.iterrows():
                        question = row['question']
                        # Grab ground truth if it exists, else N/A
                        ground_truth = row['answer'] if 'answer' in row else "N/A"
                        
                        # Retrieval
                        context = ""
                        if collection:
                            context = get_rag_context(question, collection, config['retrieval_mode'])
                        
                        # Prompt
                        full_prompt = f"""Instruction: {config['prompt_instr']}
                        
Context:
{context}

Question: {question}
Answer:"""

                        # Inference
                        answer, duration = run_inference(llm_model, full_prompt, temp, config['max_tokens'])

                        # Store
                        all_results.append({
                            "question_id": idx,
                            "original_question": question,
                            "ground_truth": ground_truth,
                            "generated_answer": answer,
                            "temperature": temp,
                            "llm_model": llm_model,
                            "rag_model": rag_model,
                            "duration_ns": duration,
                            "dataset": dataset_clean_name,
                            "context_used": context if context else "N/A"
                        })
                
                # 5. Save Excel
                pd.DataFrame(all_results).to_excel(output_path, index=False)
                logger.success(f"SAVED: {output_filename}")

if __name__ == "__main__":
    main()