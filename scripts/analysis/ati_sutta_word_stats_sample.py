import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from pathlib import Path
import argparse

# --- CONFIG ---
BASE_PATH = Path(r"C:\Users\rohan\ws\git\TFM\data\data-tipitaka\src")
EXCEL_PATH = r"C:\Users\rohan\ws\git\TFM\data\data-tipitaka\database\ati_index_metadata.xlsx"

def get_sutta_text(row):
    """Extracts raw text from the HTML file based on the Excel index."""
    file_path = BASE_PATH / row['Nikaya'].lower() / row['Filename']
    if not file_path.exists():
        return None
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            soup = BeautifulSoup(f.read(), 'html.parser')
            content = soup.find('div', id='H_content')
            if content:
                return content.get_text(separator=" ", strip=True)
    except Exception as e:
        return None
    return None

def main():
    print(f"Loading index from: {EXCEL_PATH}")
    df = pd.read_excel(EXCEL_PATH)
    
    stats_data = []

    print("Analyzing Sutta lengths (this may take a minute)...")
    for _, row in df.iterrows():
        text = get_sutta_text(row)
        if text:
            word_count = len(text.split())
            char_count = len(text)
            # Rough token estimate (OpenAI/Nomic rule of thumb: ~4 chars per token or 0.75 words/token)
            # We will use word count as the primary metric for chunk_size decisions
            stats_data.append(word_count)

    if not stats_data:
        print("No data found. Check your file paths.")
        return

    # Calculate Metrics
    mean_val = np.mean(stats_data)
    median_val = np.median(stats_data)
    std_dev = np.std(stats_data)
    min_val = np.min(stats_data)
    max_val = np.max(stats_data)
    p90 = np.percentile(stats_data, 90)
    p75 = np.percentile(stats_data, 75)

    print("\n" + "="*40)
    print("      SUTTA TOKEN/WORD STATISTICS")
    print("="*40)
    print(f"Total Suttas Analyzed: {len(stats_data)}")
    print(f"Minimum Length:       {min_val} words")
    print(f"Maximum Length:       {max_val} words")
    print(f"Mean (Average):       {mean_val:.2f} words")
    print(f"Median:               {median_val} words")
    print(f"Standard Deviation:   {std_dev:.2f}")
    print(f"75th Percentile:      {p75:.2f} words")
    print(f"90th Percentile:      {p90:.2f} words")
    print("-" * 40)
    print("RECOMMENDATION FOR FIXED CHUNK SIZE:")
    if median_val < 512:
        print(" -> Your median Sutta is small. A chunk size of 256 or 512 is ideal.")
    else:
        print(" -> You have many long Suttas. Consider testing 512 and 1024.")
    print("="*40)

if __name__ == "__main__":
    main()



# Loading index from: C:\Users\rohan\ws\git\TFM\data\data-tipitaka\database\ati_index_metadata.xlsx
# Analyzing Sutta lengths (this may take a minute)...

# ========================================
#       SUTTA TOKEN/WORD STATISTICS
# ========================================
# Total Suttas Analyzed: 121
# Minimum Length:       316 words
# Maximum Length:       27329 words
# Mean (Average):       3656.88 words
# Median:               2723.0 words
# Standard Deviation:   3287.19
# 75th Percentile:      4101.00 words
# 90th Percentile:      7149.00 words
# ----------------------------------------
# RECOMMENDATION FOR FIXED CHUNK SIZE:
#  -> You have many long Suttas. Consider testing 512 and 1024.
# ========================================
