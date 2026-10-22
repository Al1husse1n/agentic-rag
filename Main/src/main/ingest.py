import os
import time
import shutil
from pathlib import Path
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

load_dotenv()

# Resolve paths
BASE_DIR = Path(__file__).resolve().parent
pdf_path = BASE_DIR / "ethiopia_rag_practice.pdf"
persist_dir = r"C:\Users\Lenovo\Desktop\Ali\RAG Template\agentic-rag\Main\database"

print(">>> Starting document ingestion pipeline...")

# 1. Cleanly wipe the old directory to prevent duplicate embedding accumulation
if Path(persist_dir).exists():
    print(f">>> Clearing existing ChromaDB directory at: {persist_dir}")
    shutil.rmtree(persist_dir)

# 2. Load the PDF document
print(f">>> Loading PDF from: {pdf_path}")
pdf_loader = PyPDFLoader(str(pdf_path))
pages = pdf_loader.load()
print(f"✓ Loaded {len(pages)} pages successfully.")

# 3. Split documents into optimal chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)
pages_split = text_splitter.split_documents(pages)
print(f"✓ Split document into {len(pages_split)} chunks.")

# 4. Initialize Google embedding model
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
)

# 5. Embed and persist fresh vector store
print(">>> Embedding chunks and writing to ChromaDB (this may take a few seconds)...")
start_time = time.time()

vector_store = Chroma.from_documents(
    documents=pages_split,
    embedding=embeddings,
    persist_directory=persist_dir,
    collection_name="ethiopia"
)

print(f"✓ ChromaDB successfully indexed and persisted in {time.time() - start_time:.4f}s")