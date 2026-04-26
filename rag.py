# Step 1: Load the document
with open("resume.txt", "r") as f:
    text = f.read()

# Step 2: Chunk the document
chunks = [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]

print(f"Total chunks: {len(chunks)}")
print("---")
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}:")
    print(chunk)
    print("---")

# Step 3: Create embeddings
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(chunks)

print(f"Embedding shape: {embeddings.shape}")

# Step 4: Retrieve relevant chunks
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def retrieve(question, chunks, embeddings, top_k=2):
    question_embedding = model.encode([question])
    similarities = cosine_similarity(question_embedding, embeddings)[0]
    top_indices = np.argsort(similarities)[::-1][:top_k]
    return [chunks[i] for i in top_indices]

# Test it
question = "what are the skills?"
relevant_chunks = retrieve(question, chunks, embeddings)

print(f"Question: {question}")
print("Relevant chunks found:")
for chunk in relevant_chunks:
    print("---")
    print(chunk)

# Step 5: Generate answer using retrieved chunks
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def ask(question):
    relevant_chunks = retrieve(question, chunks, embeddings)
    context = "\n\n".join(relevant_chunks)
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": f"""You are a helpful assistant. 
Answer the question using ONLY the context provided below.
If the answer is not in the context, say "I don't know".

Context:
{context}"""},
            {"role": "user", "content": question}
        ]
    )
    return response.choices[0].message.content

# Test it
print(ask("what programming languages does Priya know?"))
print("---")
print(ask("what is Priya's CGPA?"))
print("---")
print(ask("does Priya know React?"))