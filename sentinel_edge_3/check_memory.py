"""
Shows what is stored in your long-term memory (ChromaDB) and which memories
would be injected into the prompt for some test questions.

How to run:
  1. Stop the Streamlit app (Ctrl+C in its terminal).
  2. Open a terminal in your MAIN project folder (the one that contains memory_db).
  3. Activate your main environment, then run:   python memory_check.py
"""
import chromadb
from sentence_transformers import SentenceTransformer

MEMORY_PATH = "./memory_db"
THRESHOLD = 0.65   # same cutoff your recall() function currently uses

QUERIES = [
    "Hello.",
    "What is the capital of France?",
    "explain how photosynthesis works.",
    "My name is Sen.",
    "What's my name?",
]

client = chromadb.PersistentClient(path=MEMORY_PATH)
col = client.get_or_create_collection("memory", metadata={"hnsw:space": "cosine"})

docs = col.get()["documents"]
print(f"\n{len(docs)} memories stored:\n")
for i, d in enumerate(docs, 1):
    print(f"  {i}. {d}")

if not docs:
    raise SystemExit("\nNo memories stored, so memory is not the cause.")

print("\nLoading embedding model...")
embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")

for q in QUERIES:
    emb = embedder.encode(f"Represent this sentence for searching relevant passages: {q}").tolist()
    res = col.query(query_embeddings=[emb], n_results=min(4, len(docs)))
    print(f"\nQuestion: {q}")
    for d, dist in zip(res["documents"][0], res["distances"][0]):
        status = "INJECTED into prompt" if dist < THRESHOLD else "ignored"
        print(f"  distance {dist:.2f}  [{status}]  {d[:100]}")