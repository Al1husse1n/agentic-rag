from dotenv import load_dotenv          #pip install python-dotenv
import os
from langgraph.graph import StateGraph, END,START         #pip install langgraph
from typing import Any, TypedDict, Annotated, Sequence, Optional
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

from pydantic import BaseModel, Field

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



llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.3
)


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    retry_no: int 
    max_retry: int
    need_retrieval: bool
    question: HumanMessage
    retrieved_docs: list    
    

class InitialResponse(BaseModel):
    """Route the user query to the appropriate pipeline."""
    need_retrieval: bool = Field(
        description="Set to True if the query asks about Ethiopia or needs document retrieval. Set to False if it is a general question."
    )
    response: Optional[str] = Field(
        default=None,
        description="Provide a complete direct answer ONLY if need_retrieval is False. Leave as null/None if need_retrieval is True."
    )

def initial_agent(state: AgentState) -> dict[str, Any]:
    system_prompt = f"""
You are an intent analyzer and router for an assistant. Your task is to analyze incoming user queries and decide whether external knowledge retrieval is required.

### Retrieval Rules
1. Set `need_retrieval = True` if the user's question relates directly or indirectly to Ethiopia (e.g., Ethiopian history, culture, geography, politics, languages, landmarks, or specific local context). 
   - When `need_retrieval = True`, leave `response` as null/None. Do NOT generate an answer yet.

2. Set `need_retrieval = False` if the user asks a general question, programming request, greeting, logic puzzle, or general knowledge question unrelated to Ethiopia.
   - When `need_retrieval = False`, provide your complete, helpful answer directly inside the `response` field.   
"""

    messages = [SystemMessage(content=system_prompt)] + list(state["messages"])    #state is wrapped in list so it can satisfy pyright
    structured_llm = llm.with_structured_output(InitialResponse)
    result = structured_llm.invoke(messages)
    #result: InitialResponse = structured_llm.invoke(messages)
    assert isinstance(result, InitialResponse)
    if result.need_retrieval:
        return {
            "need_retrieval": True,
            "question": HumanMessage(state["messages"][-1])
        }
    else:
        print(result.response)
        return {
            "need_retrieval": False,
            "messages": [AIMessage(content=result.response or "")]
        }
        


def retriever_agent(state: AgentState) -> dict:
    question = str(state["question"])
    docs = retriever.invoke(question) 
    return{
        "retrieved_docs": docs,
        "trie"
    }


def reranker_agent(state: AgentState) -> AgentState:
    stringified_docs = [doc.page_content for doc in state["retrieved_docs"]]
    question = str(state["question"])
    reranked_docs = reranker_function(question, stringified_docs)
    return state


def validator_agent(state: AgentState) -> AgentState:
    return state


def summarizer_agent(state: AgentState) -> AgentState:
    return state


#conditional routers
def initial_router(state: AgentState) -> str:
    if state["need_retrieval"] and isinstance(state["messages"][-1], AIMessage):
        return "need retrieval"
    else:
        return "quit"

def retry_router(state: AgentState) -> str:
    return "str"



    








