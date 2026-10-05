from dotenv import load_dotenv         
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings    
from langchain_community.document_loaders import PyPDFLoader           
from langchain_text_splitters import RecursiveCharacterTextSplitter     
from langchain_chroma import Chroma                     
from langchain_core.tools import tool
import cohere
from langchain_core.documents import Document

load_dotenv()

pdf_path = "ethiopia_rag_practice.pdf"
pdf_loader = PyPDFLoader(pdf_path)

try:
    pages = pdf_loader.load()
    print(f"PDF has been loaded and has {len(pages)} pages")
except Exception as e:
    print(f"Error loading PDF: {e}")
    raise

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)

pages_split = text_splitter.split_documents(pages)


embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
)

try:
    vector_store = Chroma.from_documents(
        documents=pages_split,
        embedding=embeddings,
        persist_directory=r"C:\Users\Lenovo\Desktop\Ali\langgraph\quiz",
        collection_name="ethiopia"
    )
    print("ChromaDB created")

except Exception as e:
    print(f"Error setting up chroma DB: {e}")
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
    doc_texts = [doc.page_content for doc in documents]

    results = co.rerank(
        model="rerank-v4.0-fast",
        query=query,
        documents=doc_texts,
        top_n=3
    )

    return [documents[result.index] for result in results.results]