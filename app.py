import streamlit as st
import fitz
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
from dotenv import load_dotenv
import requests
import os
from openai import OpenAI
from ebooklib import epub
from docx import Document

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
OPEAI_API_KEY = os.getenv("OPENAI_API_KEY")
model="gpt-4o-mini"
openai = OpenAI()

@st.cache_resource
def load_embedding_model(): 
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model

embedding_model = load_embedding_model()

def embed_chunks(chunks):
    embeddings = embedding_model.encode(chunks)
    return embeddings

def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

def chunk_text(text, max_length=1000, overlap=200):
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + max_length
        chunk = text[start:end]
        chunks.append(chunk)
        start += max_length - overlap

    return chunks

def ask_openai(question, context,model="gpt-4o-mini"):
    promt = f"Context:\n{context}\question:\n{question}\nAnswer:"
    response = openai.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": promt}
        ],
        max_tokens=200,
        temperature=0.2
    )    
    return response.choices[0].message.content.strip()

def extract_text_from_file(file):
    if file.type == "application/pdf":
        with fitz.open(stream=file.read(), filetype="pdf") as doc:
            text = ""
            for page in doc:
                text += page.get_text()
            return text
    elif file.type == "text/plain":
        return file.read().decode("utf-8")
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = Document(file)
        return "\n".join([para.text for para in doc.paragraphs])
    elif file.type == "application/epub+zip":
        book = epub.read_epub(file)
        return "\n".join([item.get_body_content_str() for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT)])
    else:
        raise ValueError("Unsupported file type")
    
st.title("Chat with your Documents")

uploaded_file = st.file_uploader("Upload a document", type=["pdf","txt","docx","epub"])

if uploaded_file is not None:

    text = extract_text_from_file(uploaded_file)

    chunks = chunk_text(text)

    embeddings = embed_chunks(chunks)
    faiss_index = create_faiss_index(embeddings)

    question = st.text_input("Ask a question about the document:")

    if question:
        with st.spinner("Searching for relevant context..."):
        # Embed the question
            question_embedding = embedding_model.encode([question])
            # Search the FAISS index for similar chunks
            top_k = 3
            distances, indices = faiss_index.search(question_embedding, top_k)
            # Collect the relevant chunks
            relevant_chunks = [chunks[i] for i in indices[0]]
            context = "\n".join(relevant_chunks)

            answer = ask_openai(question, context)
            st.subheader("Answer:")
            st.write(answer)



