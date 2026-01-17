import chromadb
# Update this to your local path
#DB_ROOT = r"C:\Users\rohan\ws\LFS_local\data_server\sutta_vector_db"
DB_ROOT = "/disk1/rlaycock/rag/rag_db/sutta_vector_db"
client = chromadb.PersistentClient(path=DB_ROOT)

# Rename them one by one
# Format: client.get_collection("old_name").modify(name="new_name")
client.get_collection("book_nomic_embed_text_v1.5").modify(name="book_nomic")
client.get_collection("book_bge_m3_latest").modify(name="book_bge")
client.get_collection("ati_nomic_embed_text_v1.5").modify(name="ati_nomic")
client.get_collection("ati_bge_m3_latest").modify(name="ati_bge")

print("Collections renamed successfully.")