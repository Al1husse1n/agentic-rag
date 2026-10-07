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
from langgraph.constants import Send
from pydantic import BaseModel, Field

from .schemas import AgentState, InitialResponse, ValidatorResponse, DecomposerResponse, RetrievedResult, RerankedResult, SynthesizerResponse, ValidationResult
from .system_prompts import initial_prompt,validator_prompt, decomposer_prompt, synthesizer_prompt
from .retriever import reranker_function, retriever
from .helper_functions import format_reranked_results, compress_reranked_results, compress_retrieved_results


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
            
        }
    else:
        print(result.response)
        return {
            "need_retrieval": False,
            "messages": [AIMessage(content=result.response or "")]
        }
        

def decomposer_agent(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(content=decomposer_prompt)
    messages = [system_prompt] + list(state["messages"][-1])
    structured_llm = llm.with_structured_output(DecomposerResponse)
    result = structured_llm.invoke(messages)
    assert isinstance(result, DecomposerResponse)

    for sub_question in result.sub_questions:
        state["sub_questions"].append(sub_question)
    
    return state


def retriever_agent(state: AgentState) -> dict:
    question = state["question"]
    docs = retriever.invoke(question) 
    retrieved_result = RetrievedResult(
        question=question,
        documents=docs
    )
    return{
        "retrieved_results": retrieved_result,
        "documents": docs                      # each branck gets separate state
        #"try_no": state["try_no"] + 1      add this inside the validator, since its a shared state it's added per sub question
    }


def reranker_agent(state: AgentState) -> dict:
    docs = state["documents"]
    question = state["question"]
    reranked_docs = reranker_function(question, docs)
    reranked_results = RerankedResult(
        question=question,
        documents=reranked_docs    
    )
    return {
        "reranked_results":reranked_results
    }


def synthesizer_agent(state: AgentState) -> dict:
    system_prompt = SystemMessage(content=synthesizer_prompt)
    evidence = format_reranked_results(state["reranked_results"])
    user_question = state["messages"][-1]   
    # context = state["messages"][-5:]          context can be added, to the humanmessage at the bottom
    message = HumanMessage(content=f"""
    Current user query:
{user_question}

Retrieved evidence:
{evidence}

Using the conversation context and retrieved evidence, answer the
current user query.

Use the retrieved evidence as the factual source of truth.
Use conversation context only to understand references, intent,
and missing context.
    """)

    structured_llm = llm.with_structured_output(SynthesizerResponse)
    result = structured_llm.invoke([system_prompt, message])
    assert isinstance(result, SynthesizerResponse)

    return{
        "final_answer": result.final_answer
    }
    

def validator_agent(state: AgentState) -> dict:
    system_prompt = SystemMessage(content=validator_prompt)
    user_question = state["messages"][-1].content
    final_answer = state["final_answer"]
    compressed_retrieved_results = compress_retrieved_results(state["retrieved_results"])
    compressed_reranked_results = compress_reranked_results(state["reranked_results"])
    message = HumanMessage(content=f"""
    Orginal user query:
    {user_question}

    Retrieved Evidence:
    {compressed_retrieved_results}

    Reranked Evidence:
    {compressed_reranked_results}

    synthesized final answer:
    {final_answer}
    """)

    structured_llm = llm.with_structured_output(ValidatorResponse)    
    result = structured_llm.invoke([system_prompt, message])
    assert isinstance(result, ValidatorResponse)
    validation_result = ValidationResult(
        is_valid=result.is_valid,
        failure_type=result.failure_type,
        reason=result.reason
    )
    
    return {
        "validation_result": validation_result
    }


#fanout function
def retrieve_fan_out(state: AgentState):
    return[
        Send(
            "retrieve",
            {"question": question}
        )
        for question in state["sub_questions"]
    ]



#conditional routers
def initial_router(state: AgentState) -> str:
    if state["need_retrieval"] and isinstance(state["messages"][-1], AIMessage):
        return "need retrieval"         #goes to decomposer
    else:
        return "quit"



def retry_router(state: AgentState) -> str:
    return "str"



    








