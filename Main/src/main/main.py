from dotenv import load_dotenv         
import os
from langgraph.graph import StateGraph, END,START         
from typing import Any, TypedDict, Annotated, Sequence, Optional, NotRequired
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage, AIMessage      
from operator import add as add_messages
from langgraph.prebuilt import ToolNode
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings      
from langchain_community.document_loaders import PyPDFLoader           
from langchain_text_splitters import RecursiveCharacterTextSplitter     
from langchain_chroma import Chroma                     
from langchain_core.tools import tool
import cohere
from langchain_core.documents import Document
from pydantic import BaseModel, Field

from .llm_schemas import AgentState, InitialResponse, ValidatorResponse, ClassifierResponse
from .system_prompts import initial_prompt, classifier_prompt, validator_prompt
from .retriever import reranker_function, retriever


load_dotenv()



llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.3
)


def initial_agent(state: AgentState) -> dict[str, Any]:
    system_prompt = initial_prompt

    messages = [SystemMessage(content=system_prompt)] + list(state["messages"][-1])    #state is wrapped in list so it can satisfy pyright
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
        

def classifier_agent(state: AgentState) -> dict:
    system_prompt = SystemMessage(content=classifier_prompt)
    messages = [system_prompt] + list(state["question"])
    structured_llm = llm.with_structured_output(ClassifierResponse)
    result = structured_llm.invoke(messages)
    assert isinstance(result, ClassifierResponse)

    if result.is_single_question:
        return{
            "is_single_question" : True
        }
    
    else:
        return{
            "is_single_question" : False
        }

def retriever_agent(state: AgentState) -> dict:
    question = str(state["question"])
    docs = retriever.invoke(question) 
    return{
        "retrieved_docs": docs,
        "try_no": state["try_no"] + 1
    }


def reranker_agent(state: AgentState) -> dict:
    stringified_docs = [doc.page_content for doc in state["retrieved_docs"]]
    question = str(state["question"])
    reranked_docs = reranker_function(question, stringified_docs)
    return {
        "reranked_docs":reranked_docs
    }


def validator_agent(state: AgentState) -> AgentState:
    question = str(state["question"])
    reranked_docs = state["reranked_docs"]
    system_prompt = validator_prompt(question, reranked_docs)
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



    








