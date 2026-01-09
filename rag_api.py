from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import Dict
import os, json
from pathlib import Path

from openai import AzureOpenAI
from supabase import create_client

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from pypdf import PdfReader
from docx import Document as DocxDocument

# =========================
# ENV
# =========================
client = AzureOpenAI(
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
)
DEPLOYMENT_NAME = os.environ["AZURE_OPENAI_DEPLOYMENT"]

supabase = create_client(
    os.environ["SUPABASE_URL"],
    os.environ["SUPABASE_SERVICE_KEY"]
)

# =========================
# APP
# =========================
app = FastAPI(title="Nyaya AI API")

# =========================
# STORAGE
# =========================
DOCUMENT_INDEX: Dict[str, FAISS] = {}
CHAT_DIR = Path("chats")
CHAT_DIR.mkdir(exist_ok=True)

# =========================
# EMBEDDINGS
# =========================
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# =========================
# UTILS
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


def save_chat(session_id, role, content):
    path = CHAT_DIR / f"{session_id}.json"
    chats = json.loads(path.read_text()) if path.exists() else []
    chats.append({"role": role, "content": content})
    path.write_text(json.dumps(chats, indent=2))


def is_legal_document(text: str):
    resp = client.chat.completions.create(
        model=DEPLOYMENT_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "Classify document relevance to Indian law.\n"
                    "Legal documents include judgments, FIRs, contracts, petitions, acts.\n"
                    "Reply strictly as JSON:\n"
                    "{ \"legal\": true/false, \"reason\": \"short reason\" }"
                )
            },
            {"role": "user", "content": text[:3000]}
        ]
    )
    return json.loads(resp.choices[0].message.content)

# =========================
# UPLOAD
# =========================
@app.post("/api/upload")
async def upload(session_id: str, file: UploadFile = File(...)):
    text = extract_text(file.file, file.filename)

    verdict = is_legal_document(text)

    supabase.table("documents").insert({
        "session_id": session_id,
        "filename": file.filename,
        "is_legal": verdict["legal"],
        "reason": verdict["reason"]
    }).execute()

    if not verdict["legal"]:
        raise HTTPException(400, verdict["reason"])

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
                    "Summarize legal document.\n"
                    "Headings: Facts, Issues, Decision, Relevant Provisions."
                )
            },
            {"role": "user", "content": context}
        ]
    )

    summary = resp.choices[0].message.content
    save_chat(session_id, "assistant", summary)

    return {"summary": summary}

# =========================
# ASK DOCUMENT
# =========================
class DocQuery(BaseModel):
    session_id: str
    question: str

@app.post("/api/ask-document")
def ask_document(q: DocQuery):
    if q.session_id not in DOCUMENT_INDEX:
        return {"answer": "No document uploaded."}

    save_chat(q.session_id, "user", q.question)

    docs = DOCUMENT_INDEX[q.session_id].similarity_search(q.question, k=5)
    context = "\n\n".join(d.page_content for d in docs)

    resp = client.chat.completions.create(
        model=DEPLOYMENT_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer ONLY from document.\n"
                    "If not found say 'Not mentioned in the document'."
                )
            },
            {"role": "user", "content": context + "\n\nQ: " + q.question}
        ]
    )

    answer = resp.choices[0].message.content
    save_chat(q.session_id, "assistant", answer)

    return {"answer": answer}

# =========================
# NORMAL CHAT
# =========================
class AskQuery(BaseModel):
    session_id: str
    question: str

@app.post("/api/ask")
def ask(q: AskQuery):
    save_chat(q.session_id, "user", q.question)

    resp = client.chat.completions.create(
        model=DEPLOYMENT_NAME,
        messages=[
            {"role": "system", "content": "You are Nyaya AI, Indian legal assistant."},
            {"role": "user", "content": q.question}
        ]
    )

    answer = resp.choices[0].message.content
    save_chat(q.session_id, "assistant", answer)

    return {"answer": answer}
