from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import Dict, List
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
app = FastAPI()

# =========================
# Memory Stores
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
    elif filename.endswith(".docx"):
        doc = DocxDocument(file)
        return "\n".join(p.text for p in doc.paragraphs)
    elif filename.endswith(".txt"):
        return file.read().decode("utf-8")
    else:
        raise ValueError("Unsupported file")

# =========================
# Upload + Summarize
# =========================
@app.post("/upload")
async def upload(session_id: str, file: UploadFile = File(...)):
    text = extract_text(file.file, file.filename)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    docs = splitter.split_documents(
        [Document(page_content=text)]
    )

    vs = FAISS.from_documents(docs, embeddings)
    DOCUMENT_INDEX[session_id] = vs

    context = "\n\n".join(
        d.page_content for d in vs.similarity_search("summarize judgment", k=6)
    )

    resp = client.chat.completions.create(
        model=DEPLOYMENT_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "Summarize the judgment.\n"
                    "Include Facts, Issues, Decision.\n"
                    "Mention Sections & Acts explicitly."
                )
            },
            {"role": "user", "content": context}
        ],
    )

    return {"summary": resp.choices[0].message.content}

# =========================
# Ask on Uploaded Document
# =========================
class DocQuery(BaseModel):
    session_id: str
    question: str

@app.post("/ask-document")
def ask_document(q: DocQuery):
    if q.session_id not in DOCUMENT_INDEX:
        return {"answer": "No document uploaded for this session."}

    vs = DOCUMENT_INDEX[q.session_id]
    docs = vs.similarity_search(q.question, k=5)
    context = "\n\n".join(d.page_content for d in docs)

    resp = client.chat.completions.create(
        model=DEPLOYMENT_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer ONLY from the document.\n"
                    "Mention Sections & Acts if present.\n"
                    "If missing, say not mentioned."
                )
            },
            {"role": "user", "content": context + "\n\nQ: " + q.question}
        ],
    )

    return {"answer": resp.choices[0].message.content}
