from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import Dict
import os

from openai import AzureOpenAI
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from pypdf import PdfReader
from docx import Document as DocxDocument

# =========================
# Azure OpenAI
# =========================
client = AzureOpenAI(
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
)
DEPLOYMENT_NAME = os.environ["AZURE_OPENAI_DEPLOYMENT"]

# =========================
# FastAPI
# =========================
app = FastAPI(title="Nyaya AI API")

# =========================
# In-memory document index
# =========================
DOCUMENT_INDEX: Dict[str, FAISS] = {}

# =========================
# Embeddings
# =========================
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# =========================
# Utils
# =========================
def extract_text(file, filename: str) -> str:
    if filename.endswith(".pdf"):
        reader = PdfReader(file)
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    if filename.endswith(".docx"):
        doc = DocxDocument(file)
        return "\n".join(p.text for p in doc.paragraphs)

    if filename.endswith(".txt"):
        return file.read().decode("utf-8")

    raise ValueError("Unsupported file type")

# =========================
# Upload + Summarize
# =========================
@app.post("/api/upload")
async def upload(session_id: str, file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file received")

    print("📄 Received file:", file.filename)

    try:
        text = extract_text(file.file, file.filename)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not text.strip():
        raise HTTPException(status_code=400, detail="Empty document")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    docs = splitter.split_documents([Document(page_content=text)])
    vectorstore = FAISS.from_documents(docs, embeddings)
    DOCUMENT_INDEX[session_id] = vectorstore

    context = "\n\n".join(
        d.page_content
        for d in vectorstore.similarity_search("summarize judgment", k=6)
    )

    resp = client.chat.completions.create(
        model=DEPLOYMENT_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "Summarize the legal document.\n"
                    "Use headings: Facts, Issues, Decision, Relevant Provisions.\n"
                    "Explicitly mention Sections & Acts."
                ),
            },
            {"role": "user", "content": context},
        ],
    )

    return {
        "summary": resp.choices[0].message.content,
        "filename": file.filename,
    }

# =========================
# Ask on uploaded document
# =========================
class DocQuery(BaseModel):
    session_id: str
    question: str

@app.post("/api/ask-document")
def ask_document(q: DocQuery):
    if q.session_id not in DOCUMENT_INDEX:
        return {"answer": "No document uploaded for this session."}

    vectorstore = DOCUMENT_INDEX[q.session_id]
    docs = vectorstore.similarity_search(q.question, k=5)
    context = "\n\n".join(d.page_content for d in docs)

    resp = client.chat.completions.create(
        model=DEPLOYMENT_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer ONLY from the document.\n"
                    "Mention Sections & Acts if present.\n"
                    "If missing, say 'Not mentioned in the document'."
                ),
            },
            {"role": "user", "content": context + "\n\nQ: " + q.question},
        ],
    )

    return {"answer": resp.choices[0].message.content}

# =========================
# Normal Chat
# =========================
class AskQuery(BaseModel):
    session_id: str
    question: str

@app.post("/api/ask")
def ask(q: AskQuery):
    resp = client.chat.completions.create(
        model=DEPLOYMENT_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are Nyaya AI, an Indian legal assistant. "
                    "Answer clearly and concisely."
                ),
            },
            {"role": "user", "content": q.question},
        ],
    )

    return {"answer": resp.choices[0].message.content}
