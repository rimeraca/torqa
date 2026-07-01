from sentence_transformers import SentenceTransformer
from openai import OpenAI
import sqlite3
import numpy as np

# Connect to Foundry Local
client = OpenAI(
    base_url="http://127.0.0.1:63429/v1",
    api_key="local"
)

model = "phi-3.5-mini-instruct-trtrtx-gpu:2"

# Load embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

#load chunks and embeddings from the database
conn = sqlite3.connect("knowledge.db")
cursor = conn.cursor()
cursor.execute("SELECT text, embedding FROM chunks")
rows = cursor.fetchall()
conn.close()

chunks = [row[0] for row in rows]
chunk_embeddings = np.array([np.frombuffer(row[1], dtype=np.float32) for row in rows])

print(f"Loaded {len(chunks)} chunks from file:")

question = input("Ask a question about the document: ")
question_embedding = embedder.encode(question)

#find the most similar chunk to the question
similarities = np.dot(chunk_embeddings, question_embedding.T)
top_indices = np.argsort(similarities)[-2:][::-1]  # Get the index of the most similar chunk
top_chunks = [chunks[i] for i in top_indices]

print("Most relevant chunks:")
for chunk in top_chunks:
    print("-:", chunk)

context = "\n".join(top_chunks)

# Ask AI to answer using the top chunks
response = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": f"Answer using ONLY this information:\n{context}"},
        {"role": "user", "content": question}
    ]
)
print("Question:", question)
print("Answer:", response.choices[0].message.content)