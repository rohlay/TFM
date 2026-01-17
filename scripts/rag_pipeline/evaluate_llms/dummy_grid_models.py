import os
import pandas as pd
import ollama
import chromadb
from pathlib import Path
from loguru import logger
from chromadb.utils import embedding_functions

# Metrics Imports
from rouge_score import rouge_scorer
from bert_score import score as bert_score
try:
    from bart_score import BARTScorer
except ImportError:
    logger.error("bart_score.py not found in directory. BARTScore will be skipped.")

# --- CONFIGURATION ---
SERVER = False
if SERVER:
    DB_ROOT = Path("/disk1/rlaycock/rag/rag_db/sutta_vector_db")
    QA_DATA_DIR = Path("/disk1/rlaycock/rag/data/qa_datasets")
    OUTPUT_DIR = Path("/disk1/rlaycock/rag/experiments/final_grid_results")
else:
    DB_ROOT = Path(r"C:\Users\rohan\ws\LFS_local\data_server\sutta_vector_db")
    DB_ROOT = r"C:\Users\rohan\ws\LFS_local\data_server\sutta_vector_db"
    QA_DATA_DIR = Path(r"C:\Users\rohan\ws\LFS_local\TFM-data\QA-data_dummy")
    OUTPUT_DIR = Path("./experiments/final_grid_results")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- THE GRID ---
LLM_MODELS = ["llama3.2:latest"]

# SIMPLIFIED DICT: Key is the suffix for col_name, value is the Ollama embed model name
RAG_MODELS = {
    "nomic": "nomic-embed-text:v1.5",
    "bge": "bge-m3:latest",
    "None": None,
}

TEMP_LIST = [0.0, 0.5, 1.0]

DATASET_CONFIG = {
    "buddha_taught_qa_2.xlsx": {  
        "db_source": "book",          
        "prompt_instr": "Answer in exactly 1-2 sentences. Be concise.",
        "max_tokens": 80,
        "retrieval_mode": "chunk"
    },
    "se_pali_canon_qa_2.xlsx": { 
        "db_source": "ati",           
        "prompt_instr": "Provide a detailed explanation (1-2 paragraphs).",
        "max_tokens": 512,
        "retrieval_mode": "chunk"
    }
}

# --- HELPERS ---

def load_qa_dataset(filepath):
    try:
        df = pd.read_excel(filepath) if str(filepath).endswith('.xlsx') else pd.read_csv(filepath)
        if 'question' not in df.columns: return pd.DataFrame()
        return df[['question', 'answer']].dropna(subset=['question'])
    except Exception as e:
        logger.error(f"Load error: {e}")
        return pd.DataFrame()

def get_rag_context(query, collection, mode):
    try:
        if mode == "chunk":
            results = collection.query(query_texts=[query], n_results=5)
            context = "\n\n".join(results['documents'][0]) if results['documents'] else ""
            logger.debug(f"RETRIEVED CONTEXT (Top 200 chars): {context[:200]}...")
            return context
        elif mode == "sutta": # TODO: Test full sutta retrieval
            initial = collection.query(query_texts=[query], n_results=1)
            if not initial['metadatas'][0]: return ""
            source = initial['metadatas'][0][0]['source']
            full_data = collection.get(where={"source": source})
            combined = sorted(zip(full_data['metadatas'], full_data['documents']), key=lambda x: x[0]['chunk_index'])
            return "\n".join([chunk[1] for chunk in combined])
    except Exception as e:
        logger.error(f"Retrieval Error: {e}")
        return ""

def run_inference(llm, prompt, temp, max_tokens):
    try:
        client = ollama.Client(host='http://localhost:11434')
        response = client.generate(model=llm, prompt=prompt, options={"temperature": temp, "num_predict": max_tokens})
        return response['response'], response['total_duration']
    except Exception as e:
        return f"ERROR: {e}", 0

# --- MAIN ---

def main():
    logger.info("STARTING DUMMY GRID EXPERIMENT")
    chroma_client = chromadb.PersistentClient(path=str(DB_ROOT))
    
    # Initialize Scorers
    r_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    # Set to 'cuda' if GPU available
    b_scorer = BARTScorer(device='cpu', checkpoint='facebook/bart-large-cnn') 
    
    mega_metrics = []

    for d_idx, (filename, config) in enumerate(DATASET_CONFIG.items(), 1):
        df_qa = load_qa_dataset(QA_DATA_DIR / filename)
        if df_qa.empty: continue

        # Iterate through the simplified dictionary
        for r_idx, (rag_key, embed_model_name) in enumerate(RAG_MODELS.items(), 1):
            collection = None
            if rag_key != "None":
                # Construct col_name using the key: e.g., 'book_nomic' or 'ati_bge'
                col_name = f"{config['db_source']}_{rag_key}"

                # Setup embedding function using the dict value
                ef = embedding_functions.OllamaEmbeddingFunction(
                    url="http://localhost:11434/api/embeddings",
                    model_name=embed_model_name
                )
                logger.debug(f"Using embedding model: {embed_model_name} for collection: {col_name}")
                try:
                    #collection = chroma_client.get_collection(name=col_name, embedding_function=ef)
                    # MANUALLY override the internal function so your queries use Ollama (768-dim)
                    #logger.info(f"Connected to DB: {col_name}")
                    collection = chroma_client.get_collection(name=col_name)
                    collection._embedding_function = ef
                    logger.info(f"Connected to DB: {col_name} (Manual EF Override)")
                except Exception as e:
                    logger.error(f"Collection Error ({col_name}): {e}")
                    collection = None

            for l_idx, llm_model in enumerate(LLM_MODELS, 1):
                dataset_label = "se_qa" if "pali" in filename.lower() else "wbt_qa"
                out_path = OUTPUT_DIR / f"{dataset_label}_{llm_model.replace(':','_')}_{rag_key}.xlsx"
                
                all_results = []
                for t_idx, temp in enumerate(TEMP_LIST, 1):
                    for idx, row in df_qa.iterrows():
                        question, ref = row['question'], str(row['answer'])
                        context = get_rag_context(question, collection, config['retrieval_mode']) if collection else ""
                        
                        prompt = f"Instruction: {config['prompt_instr']}\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
                        #logger.debug(f"FULL PROMPT SENT TO LLM:\n{prompt}")
                        logger.info(f"RUNNING -> LLM: {llm_model} | RAG: {rag_key} | Temp: {temp} | Dataset: {dataset_label}")
                        logger.info(
                            f"PROGRESS: Dataset ({d_idx}/{len(DATASET_CONFIG)}) | "
                            f"RAG ({r_idx}/{len(RAG_MODELS)}) | "
                            f"LLM ({l_idx}/{len(LLM_MODELS)}) | "
                            f"Temp ({t_idx}/{len(TEMP_LIST)})"
                        )
                        logger.info(f"RUNNING -> LLM: {llm_model} | RAG: {rag_key} | Temp: {temp}")
                        ans, dur = run_inference(llm_model, prompt, temp, config['max_tokens'])

                        # Metrics Computation
                        rl = r_scorer.score(ref, ans)['rougeL'].fmeasure
                        P, R, F1 = bert_score([ans], [ref], lang="en")
                        bs = F1.item()
                        bars = b_scorer.score([ans], [ref])[0]

                        # Terminal Log
                        print(f"\n{'='*60}\nQ: {question}\nA: {ans}\nMETRICS -> R-L: {rl:.3f} | BS: {bs:.3f} | BART: {bars:.3f}\n{'='*60}")

                        res_obj = {
                            "question": question, "generated_answer": ans, "ground_truth": ref,
                            "temp": temp, "llm": llm_model, "rag": rag_key,
                            "ROUGE-L": rl, "BERTScore": bs, "BARTScore": bars, "context": context[:200]
                        }
                        all_results.append(res_obj)
                        mega_metrics.append(res_obj)

                pd.DataFrame(all_results).to_excel(out_path, index=False)

    pd.DataFrame(mega_metrics).to_excel(OUTPUT_DIR / "MEGA_METRICS_SUMMARY.xlsx", index=False)
    logger.success("Experiment Complete.")

if __name__ == "__main__":
    main()