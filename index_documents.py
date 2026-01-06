from datasets import load_dataset
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

print("Loading dataset...")
dataset = load_dataset(
    "ShoaibSSM/IndianLawUnified",
    split="train[:20000]"
)


docs = []
for row in dataset:
    text = f"""
Question:
{row['instruction']}

Answer:
{row['response']}
"""
    docs.append(Document(page_content=text.strip()))

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)

chunks = splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local("indian_law_index")

print("FAISS index saved ✅")
