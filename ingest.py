from sentence_transformers import SentenceTransformer
import sqlite3
import numpy as np
import os

#load embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

#get all the text files in the documents folder
documents_folder = "documents"
all_chunks = []

for filename in os.listdir(documents_folder):
    if filename.endswith(".txt"):
        filepath = os.path.join(documents_folder, filename)
        with open(filepath, "r") as f:
            document = f.read()

        #split intro chunks
        chunks = document.split("\n")
        chunks = [chunk.strip() for chunk in chunks if chunk.strip()]

        all_chunks.extend(chunks)
        print(f"Loaded '{len(chunks)}' chunks from '{filename}'")

print(f"Total chunks: {len(all_chunks)}")

#connect to database
conn = sqlite3.connect("knowledge.db")
cursor = conn.cursor()

#clear old data
cursor.execute("DELETE FROM chunks")

#generate embeddings and save each chunk
for chunk in all_chunks:
    embedding = embedder.encode(chunk)
    embedding_bytes = embedding.astype(np.float32).tobytes()  # Convert embedding to bytes
    cursor.execute("INSERT INTO chunks (text, embedding) VALUES (?, ?)", (chunk, embedding_bytes))

conn.commit()
conn.close()

print(f"Saved {len(all_chunks)} chunks to the database!")