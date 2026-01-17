import chromadb
from pathlib import Path

# Matches your server DB root
DB_ROOT = r"C:\Users\rohan\ws\LFS_local\data_server\sutta_vector_db"
client = chromadb.PersistentClient(path=DB_ROOT)

# List all collections
collections = client.list_collections()
print(f"Found {len(collections)} collections:")

for col in collections:
    # Print the name and count of items
    print(f" - {col.name} ({col.count()} items)")


#---------------------------------------------------------

# List of collections to PERMANENTLY DELETE
# These are the ones with slashes/wrong names or 0 items
collections_to_delete = [
    "ati_jina_v2",
    "book_jina_v2",
]

print(f"Starting cleanup in: {DB_ROOT}")

for col_name in collections_to_delete:
    try:
        client.delete_collection(name=col_name)
        print(f"Successfully deleted: {col_name}")
    except Exception as e:
        print(f"Skipping {col_name}: Not found or already deleted.")

print("\nRemaining collections:")
for col in client.list_collections():
    print(f" - {col.name} ({col.count()} items)")


# Remaining collections:
#  - book_mxbai_embed_large_latest (1033 items)
#  - book_nomic_embed_text_v1.5 (1033 items)
#  - ati_mxbai_embed_large_latest (19248 items)
#  - ati_bge_m3_latest (19248 items)
#  - ati_nomic_embed_text_v1.5 (19248 items)
#  - book_bge_m3_latest (1033 items)
#  - book_jina_jina_embeddings_v2_base_en_latest (1033 items)
#  - ati_jina_jina_embeddings_v2_base_en_latest (19248 items)