import os
import pandas as pd
from pathlib import Path
from bs4 import BeautifulSoup
import re

def extract_sutta_metadata(file_path):
    """Parses the ATI metadata dump inside the HTML comment."""
    metadata = {
        "Title": "Unknown",
        "Subtitle": "",
        "Author": "Unknown",
        "Summary": ""
    }
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # 1. Extract Title from the <title> tag as a backup
            soup = BeautifulSoup(content, 'html.parser')
            if soup.title:
                metadata["Title"] = soup.title.string.split(":")[0].strip()

            # 2. Extract from the 'ATIDoc metadata dump' using Regex
            # This looks for patterns like [MY_TITLE]={Ogha-tarana Sutta}
            fields = {
                "Title": r"\[MY_TITLE\]=\{(.*?)\}",
                "Subtitle": r"\[SUBTITLE\]=\{(.*?)\}",
                "Author": r"\[AUTHOR\]=\{(.*?)\}",
                "Summary": r"\[SUMMARY\]=\{(.*?)\}"
            }
            
            for key, pattern in fields.items():
                match = re.search(pattern, content)
                if match:
                    metadata[key] = match.group(1)
                    
    except Exception as e:
        print(f"Error reading {file_path.name}: {e}")
        
    return metadata

def create_rich_index(root_dir, output_dir):
    nikayas = ['dn', 'mn', 'sn', 'an', 'kn']
    root = Path(root_dir)
    all_data = []

    print(f"--- Extracting Metadata from {len(nikayas)} Nikayas ---")

    for nikaya in nikayas:
        nikaya_path = root / nikaya
        if not nikaya_path.exists(): continue
        
        for file_path in nikaya_path.rglob('*.html'):
            # Skip index/utility files
            if file_path.name in ['index.html', 'sutta.html', 'translators.html']: continue
            
            print(f"Processing: {file_path.name}")
            meta = extract_sutta_metadata(file_path)
            
            all_data.append({
                "Nikaya": nikaya.upper(),
                "Folder": file_path.parent.name,
                "Filename": file_path.name,
                "Title": meta["Title"],
                "Subtitle": meta["Subtitle"],
                "Author": meta["Author"],
                "Summary": meta["Summary"]
            })

    # Save to Mega Excel
    df = pd.DataFrame(all_data)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "Sutta_Knowledge_Base.xlsx"
    df.to_excel(output_file, index=False)
    print(f"\nSaved Rich Index to: {output_file}")

# Usage
input_path = r"C:\Users\rohan\ws\LFS_local\TFM-data\ati\tipitaka"
output_path = r"C:\Users\rohan\ws\git\TFM\dataset_ati\suttas_html_metadata"

create_rich_index(input_path, output_path)