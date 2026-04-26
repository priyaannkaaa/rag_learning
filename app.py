import chainlit as cl
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq
from dotenv import load_dotenv
from pypdf import PdfReader
import numpy as np
import os

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

def retrieve(question, chunks, embeddings, top_k=5):
    q_emb = embed_model.encode([question])
    scores = cosine_similarity(q_emb, embeddings)[0]
    top = np.argsort(scores)[::-1][:top_k]
    return [chunks[i] for i in top]

@cl.on_chat_start
async def start():
    files = await cl.AskFileMessage(
        content="Upload a PDF to get started!",
        accept=["application/pdf"],
        max_size_mb=10
    ).send()

    file = files[0]
    text = extract_text_from_pdf(file.path)
    chunks = chunk_text(text)
    embeddings = embed_model.encode(chunks)

    cl.user_session.set("chunks", chunks)
    cl.user_session.set("embeddings", embeddings)
    cl.user_session.set("history", [])

    await cl.Message(content=f"PDF loaded! {len(chunks)} chunks created. Ask me anything!").send()

@cl.on_message
async def main(message: cl.Message):
    chunks = cl.user_session.get("chunks")
    embeddings = cl.user_session.get("embeddings")
    history = cl.user_session.get("history")

    question = message.content
    relevant_chunks = retrieve(question, chunks, embeddings)
    context = "\n\n".join(relevant_chunks)

    history.append({"role": "user", "content": question})

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": f"Answer ONLY from context. Say 'I don't know' if unsure.\n\nContext:\n{context}"},
        ] + history
    )

    reply = response.choices[0].message.content
    history.append({"role": "assistant", "content": reply})
    cl.user_session.set("history", history)

    await cl.Message(content=reply).send()