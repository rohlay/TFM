## Sutta Metadata & Indexing Pipeline
This repository contains tools to transform raw **Access to Insight (ATI)** HTML files into a structured **Sutta Knowledge Base**.

# 1. Data Source Structure
The scripts expect a raw Tipitaka directory structured as follows:

```plaintext
tipitaka/
├── an/ (Anguttara Nikaya)
│   ├── an01/an01.021-040.than.html
│   └── ...
├── dn/ (Digha Nikaya)
│   └── dn.01.0.bodh.html
├── kn/ (Khuddaka Nikaya)
│   ├── dhp/dhp.01.budd.html
│   ├── iti/iti.1.001-027.than.html
│   └── ...
├── mn/ (Majjhima Nikaya)
│   └── mn.001.than.html
└── sn/ (Samyutta Nikaya)
    ├── sn01/sn01.001.than.html
    └── ...
```

### 2. Processing Pipeline

The pipeline consists of two scripts designed to scrape and organize metadata from the Tipitaka HTML structure:

#### Step 1: `build_index_with_metadata.py` (Primary Script)
* **Function**: Scrapes the hidden `ATIDoc metadata dump` inside HTML comments.
* **Extracted Info**: Title, Subtitle, Author, and Summary.
* **Primary Output**: `index_ati.xlsx` — The definitive index for RAG pipelines.

#### Step 2: `build_index_indiv_suttas.py` (Supporting Script)
* **Function**: Performs a recursive file system scan to ensure physical file accountability.
* **Outputs**: Generates individual Excel indexes for every subfolder (e.g., `sn01_index.xlsx`) and a secondary global index `index_ati_suttas.xlsx`.
* **Role**: Provided for completeness; used if folder-specific tracking is required later.

---

### 2. Summary of Outputs
* **`index_ati.xlsx`**: Main data source containing doctrinal summaries and author metadata.
* **Individual Folder Indexes**: Supplementary files for targeted processing of specific Sutta groups.

