import pymupdf4llm
import pathlib

# --- CONFIG ---
INPUT_FILE = r"C:\Users\rohan\ws\LFS_local\TFM-data\What the Buddha Taught.pdf"
OUTPUT_FILE = "what_the_buddha_taught.md"

def convert_pdf_to_md(pdf_path, output_path):
    print(f"Starting conversion: {pdf_path}...")
    
    # pymupdf4llm analyzes layout and produces structured markdown
    md_text = pymupdf4llm.to_markdown(pdf_path)
    
    # Save the markdown content to a file
    pathlib.Path(output_path).write_bytes(md_text.encode("utf-8"))
    
    print(f"Success! Markdown saved to: {output_path}")

if __name__ == "__main__":
    convert_pdf_to_md(INPUT_FILE, OUTPUT_FILE)