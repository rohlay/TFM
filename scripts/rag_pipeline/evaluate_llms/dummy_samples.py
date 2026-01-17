import os
import pandas as pd
import ollama
from pathlib import Path
from loguru import logger
import torch

# --- STANDARD METRICS ---
from rouge_score import rouge_scorer
from bert_score import score as bert_score
try:
    from bart_score import BARTScorer
except ImportError:
    logger.error("bart_score not found. Skipping.")

# --- ADVANCED QA METRICS ---
# 1. SummaC (NLI based)
try:
    from summac.model_summac import SummaCZS
    SUMMAC_AVAILABLE = True
except ImportError:
    logger.warning("SummaC not installed. Run 'pip install summac'")
    SUMMAC_AVAILABLE = False

# 2. AlignScore (SOTA Alignment)
# Note: You typically need to download a checkpoint for this.
try:
    from alignscore import AlignScore
    ALIGNSCORE_AVAILABLE = True
except ImportError:
    logger.warning("AlignScore not installed or setup.")
    ALIGNSCORE_AVAILABLE = False

# 3. QAFactEval (The heavy hitter)
try:
    from qafacteval import QAFactEval
    QAFACT_AVAILABLE = True
except ImportError:
    logger.warning("QAFactEval not installed. Run 'pip install qafacteval'")
    QAFACT_AVAILABLE = False


# --- CONFIGURATION ---
QA_DATA_DIR = Path(r"C:\Users\rohan\ws\LFS_local\TFM-data\QA-data_dummy")
OUTPUT_DIR = Path("./experiments/n30_test")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ALIGNSCORE CHECKPOINT (You must download this from their repo if using AlignScore)
# Usually 'AlignScore-base.ckpt' or similar
ALIGNSCORE_CKPT = r"C:\Users\rohan\ws\models\AlignScore-base.ckpt" 

# --- TEST SETTINGS ---
LLM_MODEL = "llama3.2:latest"
N_SAMPLES = 30
TEMPERATURE = 0.7 
WBT_FILE = "buddha_taught_qa_2.xlsx"
PROMPT_INSTR = "Answer in exactly 1 sentence. Be concise."

# --- HELPERS ---

def load_qa_dataset(filepath):
    try:
        df = pd.read_excel(filepath) if str(filepath).endswith('.xlsx') else pd.read_csv(filepath)
        if 'question' not in df.columns: return pd.DataFrame()
        return df[['question', 'answer']].dropna(subset=['question'])
    except Exception as e:
        logger.error(f"Load error: {e}")
        return pd.DataFrame()

def run_inference(llm, prompt, temp, max_tokens):
    try:
        client = ollama.Client(host='http://localhost:11434')
        response = client.generate(model=llm, prompt=prompt, options={"temperature": temp, "num_predict": max_tokens})
        return response['response']
    except Exception as e:
        return f"ERROR: {e}"

# --- MAIN ---

def main():
    logger.info(f"STARTING N={N_SAMPLES} SAMPLES TEST (ALL METRICS)")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Inference Device: {device}")

    # ---------------------------
    # 1. INITIALIZE SCORERS
    # ---------------------------
    logger.info("Initializing Metrics...")
    
    # Standard
    r_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    b_scorer = BARTScorer(device=device, checkpoint='facebook/bart-large-cnn') 

    # SummaC (Zero Shot is standard)
    summac_model = None
    if SUMMAC_AVAILABLE:
        try:
            # imn = "MNLI" (Multi-Genre NLI) is standard for contradiction detection
            summac_model = SummaCZS(granularity="sentence", model_name="vitaminc", device=device) 
            logger.success("SummaC Loaded")
        except Exception as e:
            logger.error(f"Failed to load SummaC: {e}")

    # AlignScore
    align_model = None
    if ALIGNSCORE_AVAILABLE and os.path.exists(ALIGNSCORE_CKPT):
        try:
            align_model = AlignScore(model='roberta-base', batch_size=32, device=device, ckpt_path=ALIGNSCORE_CKPT, evaluation_mode='nli_sp')
            logger.success("AlignScore Loaded")
        except Exception as e:
            logger.error(f"Failed to load AlignScore: {e}")

    # QAFactEval
    qafact_model = None
    if QAFACT_AVAILABLE:
        try:
            # This loads a localized QG/QA pipeline
            qafact_model = QAFactEval(protocol="qa_race", device=device)
            logger.success("QAFactEval Loaded")
        except Exception as e:
            logger.error(f"Failed to load QAFactEval: {e}")


    # ---------------------------
    # 2. LOAD DATA
    # ---------------------------
    df_qa = load_qa_dataset(QA_DATA_DIR / WBT_FILE)
    if df_qa.empty:
        logger.error("Dataset empty!")
        return
    
    first_row = df_qa.iloc[0] 
    question = first_row['question']
    reference = str(first_row['answer'])
    
    results = []

    # ---------------------------
    # 3. GENERATION LOOP
    # ---------------------------
    for n in range(1, N_SAMPLES + 1):
        logger.info(f"Processing Sample {n}/{N_SAMPLES}...")
        
        prompt = f"Instruction: {PROMPT_INSTR}\n\nQuestion: {question}\nAnswer:"
        answer = run_inference(LLM_MODEL, prompt, TEMPERATURE, 80)

        # --- CALCULATE METRICS ---
        
        # A. ROUGE
        rl = r_scorer.score(reference, answer)['rougeL'].fmeasure
        
        # B. BERTScore
        # (Returns Precision, Recall, F1)
        _, _, F1 = bert_score([answer], [reference], lang="en", verbose=False)
        bs = F1.item()
        
        # C. BARTScore
        bars = b_scorer.score([answer], [reference])[0]

        # D. SummaC
        sc_score = 0.0
        if summac_model:
            # SummaC expects (document, summary) -> (reference, generated)
            sc_score = summac_model.score([reference], [answer])['scores'][0]

        # E. AlignScore
        as_score = 0.0
        if align_model:
            as_score = align_model.score(contexts=[reference], claims=[answer])[0]

        # F. QAFactEval
        qaf_score = 0.0
        if qafact_model:
            # Returns a complex object, we want the composite score
            # evaluate(src, pred)
            qaf_res = qafact_model.evaluate_batch([reference], [[answer]])
            qaf_score = qaf_res[0][0]['qa-eval']['f1'] # Extracting the QA F1 consistency

        # Live Print
        print(f"\n{'='*20} SAMPLE {n} {'='*20}")
        print(f"Gen: {answer}")
        print(f"Ref: {reference}")
        print(f"--- STANDARD ---")
        print(f"ROUGE-L:   {rl:.3f}")
        print(f"BERTScore: {bs:.3f}")
        print(f"BARTScore: {bars:.3f}")
        print(f"--- ADVANCED ---")
        print(f"SummaC:    {sc_score:.3f}")
        print(f"AlignSc:   {as_score:.3f}")
        print(f"QAFact:    {qaf_score:.3f}")

        results.append({
            "sample_id": n,
            "question": question,
            "generated_answer": answer,
            "ground_truth": reference,
            "ROUGE-L": rl,
            "BERTScore": bs,
            "BARTScore": bars,
            "SummaC": sc_score,
            "AlignScore": as_score,
            "QAFactEval": qaf_score
        })

    # 4. Save Results
    out_path = OUTPUT_DIR / f"n30_ADVANCED_{LLM_MODEL.replace(':','_')}.xlsx"
    pd.DataFrame(results).to_excel(out_path, index=False)
    logger.success(f"Advanced Test complete. Results saved to {out_path}")

if __name__ == "__main__":
    main()