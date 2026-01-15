import os
import pandas as pd
from pathlib import Path
from collections import defaultdict

def create_excel_indexes(root_dir, output_dir, save_mega=True, save_individual=True):
    nikayas = ['dn', 'mn', 'sn', 'an', 'kn']
    exclude_files = {'index.html', 'renumber.html', 'renumber2.html', 'sutta.html', 'translators.html'}
    
    root = Path(root_dir)
    out_base = Path(output_dir)
    out_base.mkdir(parents=True, exist_ok=True)

    folder_groups = defaultdict(list)
    mega_list = []
    
    # Counter for summary
    stats = {n.upper(): 0 for n in nikayas}

    print(f"--- Starting Scan in {root} ---")

    for nikaya in nikayas:
        nikaya_path = root / nikaya
        if not nikaya_path.exists():
            print(f"Skipping: {nikaya} (not found)")
            continue
        
        print(f"Processing Nikaya: {nikaya.upper()}...")

        for file_path in nikaya_path.rglob('*.html'):
            if file_path.name not in exclude_files:
                parent_dir = file_path.parent
                
                sutta_data = {
                    "Nikaya": nikaya.upper(),
                    "Folder": parent_dir.name,
                    "Filename": file_path.name
                }
                
                folder_groups[parent_dir].append(sutta_data)
                mega_list.append(sutta_data)
                stats[nikaya.upper()] += 1
                
                print(f"   [Found] {nikaya.upper()} -> {parent_dir.name} -> {file_path.name}")

    # --- SAVE INDIVIDUAL ---
    if save_individual:
        print(f"\n--- Generating Individual Excel Files ---")
        for folder_path, suttas in folder_groups.items():
            nikaya_name = suttas[0]["Nikaya"].lower()
            folder_name = folder_path.name
            nikaya_out_path = out_base / nikaya_name
            nikaya_out_path.mkdir(exist_ok=True)

            pd.DataFrame(suttas).to_excel(nikaya_out_path / f"{nikaya_name}_{folder_name}_index.xlsx", index=False)

    # --- SAVE MEGA ---
    if save_mega and mega_list:
        print(f"\n--- Generating Mega Excel File ---")
        mega_path = out_base / "ALL_SUTTAS_MEGA_INDEX.xlsx"
        pd.DataFrame(mega_list).to_excel(mega_path, index=False)
        print(f"SUCCESS: Mega Excel saved at {mega_path}")

    # --- FINAL SUMMARY ---
    print("\n" + "="*30)
    print("      FINAL SUTTA COUNT")
    print("="*30)
    for n, count in stats.items():
        print(f"{n}: {count}")
    print("-" * 30)
    print(f"TOTAL SUTTAS INDEXED: {sum(stats.values())}")
    print("="*30)

# --- CONFIGURATION ---
input_path = r"C:\Users\rohan\ws\LFS_local\TFM-data\ati\tipitaka"
output_path = r"C:\Users\rohan\ws\git\TFM\dataset_ati\sutta_indexes"

create_excel_indexes(input_path, output_path, save_mega=True, save_individual=True)