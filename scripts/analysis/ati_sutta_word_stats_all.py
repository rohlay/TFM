import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from pathlib import Path

# --- CONFIG ---
BASE_PATH = Path(r"C:\Users\rohan\ws\git\TFM\data\data-tipitaka\src")
EXCEL_PATH = r"C:\Users\rohan\ws\git\TFM\data\data-tipitaka\database\ati_index_metadata.xlsx"

def get_sutta_text(row):
    """Modified to search subfolders recursively to find all 1200 suttas."""
    filename = row['Filename']
    nikaya_dir = BASE_PATH / row['Nikaya'].lower()
    
    # Use rglob to find the file even if it is in a subfolder like sn/sn01/
    actual_paths = list(nikaya_dir.rglob(filename))
    
    if not actual_paths:
        return None
        
    file_path = actual_paths[0] # Take the first match found
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            soup = BeautifulSoup(f.read(), 'html.parser')
            content = soup.find('div', id='H_content')
            if content:
                return content.get_text(separator=" ", strip=True)
    except Exception:
        return None
    return None

def main():
    print(f"Loading index from: {EXCEL_PATH}")
    df = pd.read_excel(EXCEL_PATH)
    
    stats_data = []

    print(f"Analyzing {len(df)} Suttas listed in Excel...")
    for _, row in df.iterrows():
        text = get_sutta_text(row)
        if text:
            word_count = len(text.split())
            stats_data.append(word_count)

    if not stats_data:
        print("No data found. Check your BASE_PATH or Excel Filenames.")
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
    print("      SUTTA WORD COUNT STATISTICS")
    print("="*40)
    print(f"Total Suttas Found:   {len(stats_data)} / {len(df)}")
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



# (pec1_rdf) PS C:\Users\rohan\ws\git\TFM\scripts\analysis> python .\ati_sutta_word_stats_all.py
# Loading index from: C:\Users\rohan\ws\git\TFM\data\data-tipitaka\database\ati_index_metadata.xlsx
# Analyzing 1223 Suttas listed in Excel...

# ========================================
#       SUTTA WORD COUNT STATISTICS
# ========================================
# Total Suttas Found:   1220 / 1223
# Minimum Length:       26 words
# Maximum Length:       27329 words
# Mean (Average):       907.89 words
# Median:               469.5 words
# Standard Deviation:   1571.92
# 75th Percentile:      938.25 words
# 90th Percentile:      2017.30 words
# ----------------------------------------
# RECOMMENDATION FOR FIXED CHUNK SIZE:
#  -> Your median Sutta is small. A chunk size of 256 or 512 is ideal.
# ========================================