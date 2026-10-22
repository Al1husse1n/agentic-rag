from dotenv import load_dotenv         
import os
import time
from langchain_google_genai import GoogleGenerativeAIEmbeddings    
from langchain_community.document_loaders import PyPDFLoader           
from langchain_text_splitters import RecursiveCharacterTextSplitter     
from langchain_chroma import Chroma                     
from langchain_core.tools import tool
import cohere
from langchain_core.documents import Document
from pathlib import Path
import pypdf
load_dotenv()


embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
)

try:
    chroma_start = time.time()
    vector_store = Chroma(
        persist_directory=r"C:\Users\Lenovo\Desktop\Ali\RAG Template\agentic-rag\Main\database",
        collection_name="ethiopia",
        embedding_function=embeddings
    )
    print(">>> Loaded existing ChromaDB successfully")
    print(f"⏱ ChromaDB loading took {time.time() - chroma_start:.4f}s")

except Exception as e:
    print(f"Error loading Chroma DB: {e}")
    raise

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k":10}
)

#reranker
co = cohere.ClientV2(
    api_key=os.getenv("COHERE_API_KEY")
)


def reranker_function(query: str, documents: list[Document]) -> list[Document]:
    start = time.time()
    doc_texts = [doc.page_content for doc in documents]

    results = co.rerank(
        model="rerank-v4.0-fast",
        query=query,
        documents=doc_texts,
        top_n=3
    )

    reranked = [documents[result.index] for result in results.results]
    print(f"⏱ reranker_function took {time.time() - start:.4f}s")
    return reranked