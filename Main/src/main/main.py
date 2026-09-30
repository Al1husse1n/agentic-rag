from dotenv import load_dotenv          #pip install python-dotenv
import os
from langgraph.graph import StateGraph, END,START         #pip install langgraph
from typing import TypedDict, Annotated, Sequence, Optional
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage, AIMessage      #	pip install langchain-core
from operator import add as add_messages
from langgraph.prebuilt import ToolNode
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings         #pip install langchain-google-genai
from langchain_community.document_loaders import PyPDFLoader            #pip install langchain-community   or pip install pypdf
from langchain_text_splitters import RecursiveCharacterTextSplitter     #pip install langchain-text-splitters
from langchain_chroma import Chroma                     #pip install langchain-chroma
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
    search_kwargs={"k":5}
)

#reranker
co = cohere.ClientV2(
    api_key=os.getenv("COHERE_API_KEY")
)

def reranker_function(query:str, documents:list) -> list[str]:
    results = co.rerank(
        model="rerank-v4.0-fast",
        query=query,
        documents=documents,
        top_n=2
    ) 
    
    docs = []
    for result in results.results:
        docs.append(documents[result.index])
    
    return docs

@tool               #retriever is a tool, b/c it may not always need to retrieve
def retriever_tool(query:str):
    """retrieves relevant information from the knowledge base from the query"""
    infos = retriever.invoke(query)
    if not infos:
        return "no relevant information was found"

    results = []

    for info in infos:
        results.append(info)

    return results


tools = [retriever_tool]

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.5
).bind_tools(tools)


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    retry_no: int 
    max_retry: int
    question: HumanMessage



def reranker(state:AgentState) -> AgentState:
    return state

    







