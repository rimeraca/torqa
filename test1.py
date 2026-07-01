from openai import OpenAI

# Connect to Foundry Local
client = OpenAI(
    base_url="http://127.0.0.1:63429/v1",
    api_key="local"
)

model = "phi-3.5-mini-instruct-trtrtx-gpu:2"

# Send a message
response = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": "You are a helpful assistant that only answers based on provided documents."},
        {"role": "user", "content": "Hello! Who are you?"}
    ]
)

print(response.choices[0].message.content)

# Read the document
with open("documents/sample.txt", "r") as f:
    document = f.read()

#ask a a question about the document
question = "Where is Bahcesehir University located?"

responser = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": f"Answer the user's question using ONLY the information in the document below. Do not introduce yourself. Do not say anything except the answer.\n\nDocument:\n{document}"},
        {"role": "user", "content": question}
    ]
)

print("Question:", question)
print("Answer:", responser.choices[0].message.content)