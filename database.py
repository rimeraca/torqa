import sqlite3

#connect to (or create) the database file
conn = sqlite3.connect("knowledge.db")
cursor = conn.cursor()

# Create a table to store chunks and their embeddings
cursor.execute("""
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT NOT NULL,
    embedding BLOB NOT NULL
)
""")

conn.commit()
conn.close()

print("Database created successfully")