import chromadb

# Initialize the client at your DB path
client = chromadb.PersistentClient(path="./sutta_vector_db")

# Delete the specific collection
try:
    client.delete_collection(name="ati_nomic_embed_text")
    print("Collection 'ati_nomic_embed_text' deleted successfully.")
except ValueError:
    print("Collection not found.")