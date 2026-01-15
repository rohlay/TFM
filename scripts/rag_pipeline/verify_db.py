import chromadb

# Point this to your DB_ROOT
DB_ROOT = "./sutta_vector_db"

def verify():
    client = chromadb.PersistentClient(path=DB_ROOT)
    collections = client.list_collections()
    
    print(f"\n--- Database Verification: {DB_ROOT} ---")
    if not collections:
        print("No collections found. Check your ingestion logs.")
        return

    for col_info in collections:
        col = client.get_collection(col_info.name)
        count = col.count()
        print(f"Collection: {col_info.name:<30} | Vectors: {count}")

if __name__ == "__main__":
    verify()