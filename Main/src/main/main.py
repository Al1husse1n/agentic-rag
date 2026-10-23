from dotenv import load_dotenv         
import os
import time
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
from langgraph.types import Overwrite, Send
from pydantic import BaseModel, Field
from .utils.logger import track_node_metrics

from .schemas import AgentState, InitialResponse, ValidatorResponse, DecomposerResponse, RetrievedResult, RerankedResult, SynthesizerResponse, ValidationResult
from .system_prompts import initial_prompt,validator_prompt, decomposer_prompt, synthesizer_prompt
from .retriever import reranker_function, retriever
from .helper_functions import format_reranked_results, compress_reranked_results, compress_retrieved_results


load_dotenv()



llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3
)

@track_node_metrics("InitialAgent")
def initial_agent(state: AgentState) -> dict[str, Any]:
    try:
        start = time.time()
        print(">>> Initial Agent called")
        system_prompt = SystemMessage(content=initial_prompt)
        if not system_prompt:
            print("⚠️ [Config Warning] initial_prompt is missing. Falling back to default.")
            system_prompt = "You are a helpful assistant for Ethiopian higher education."
        messages = [system_prompt] + list(state.get("messages", [])[-5:])    
        if not messages:
            raise ValueError("State contains no messages for initial processing.")
        try:
            structured_llm = llm.with_structured_output(InitialResponse)
            result = structured_llm.invoke(messages)
        except Exception as e:
            print(f"⚠️ [Error Boundary] Structured output or API failure: {str(e)}")
            result = InitialResponse(
                need_retrieval=True,       # Default safe fallback
                need_clarification=False,  
            )
        print(result)
        assert isinstance(result, InitialResponse)
        print(f"⏱ initial_agent took {time.time() - start:.4f}s")

        return {
            "need_retrieval": result.need_retrieval,
            "need_clarification": result.need_clarification,
            "user_question": state.get("messages")[-1].content,
            **({"messages": [AIMessage(content=result.response or "")]} if not result.need_retrieval else {})
        }
    
    except ValueError as e:
        if "no messages" in str(e):
            print(str(e))
            fallback_msg = "Hello! How can I help you with Ethiopian higher education today?"
            return {
                "need_retrieval": False,
                "need_clarification": False,
                "user_question": "",
                "messages": [AIMessage(content=fallback_msg)],
                "error_occured": True
            }

        print(f"⚠️ [ValueError in InitialAgent]: {str(e)}")
        raise

    except Exception as e:
        print(f"⚠️ [Critical Error in InitialAgent]: {str(e)}")
        fallback_msg = getattr(state.get("messages")[-1], "content", "Hello")
        return {
            "need_retrieval": True,
            "need_clarification": False,
            "user_question": fallback_msg,
            "error_occurred": True
        }
        
@track_node_metrics("DecomposerAgent")
def decomposer_agent(state: AgentState) -> dict:
    start = time.time()
    print(">>> Decomposer Agent called")
    system_prompt = SystemMessage(content=decomposer_prompt)
    messages = [system_prompt] + list(state["messages"][-5:])
    structured_llm = llm.with_structured_output(DecomposerResponse)
    result = structured_llm.invoke(messages)
    assert isinstance(result, DecomposerResponse)

    print(f"⏱ decomposer_agent took {time.time() - start:.4f}s")
    return{
        "sub_questions": result.sub_questions
    }


@track_node_metrics("RetrieverAgent")
def retriever_agent(state: AgentState) -> dict:
    start = time.time()
    print(">>> Retriever Agent called")
    question = state["question"]
    docs = retriever.invoke(question) 
    retrieved_result = RetrievedResult(
        question=question,
        documents=docs
    )
    print(f"⏱ retriever_agent took {time.time() - start:.4f}s")
    return{
        "retrieved_results": [retrieved_result],
        "documents": docs,                      # each branch gets separate state
        "question": question
        #"try_no": state["try_no"] + 1      add this inside the synthesizer, since its a shared state it's added per sub question
    }


@track_node_metrics("RerankerAgent")
def reranker_agent(state: AgentState) -> dict:
    start = time.time()
    print(">>> Reranker Agent called")
    docs = state["documents"]
    question = state["question"]
    reranked_docs = reranker_function(question, docs)
    reranked_results = RerankedResult(
        question=question,
        documents=reranked_docs    
    )
    print(f"⏱ reranker_agent took {time.time() - start:.4f}s")
    return {
        "reranked_results":[reranked_results]
    }


@track_node_metrics("SynthesizerAgent")
def synthesizer_agent(state: AgentState) -> dict:
    start = time.time()
    print(">>> Synthesizer Agent called")
    system_prompt = SystemMessage(content=synthesizer_prompt)
    evidence = format_reranked_results(state["reranked_results"])
    user_question = state["user_question"]   

    recent_messages = state.get("messages", [])[:-1][-5:]
    context_transcript = "\n".join([
        f"{msg.type.capitalize()}: {msg.content}" 
        for msg in recent_messages
    ]) if recent_messages else "No prior conversation context."

    message = HumanMessage(content=f"""
Conversation History:
{context_transcript}

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

    print(f"⏱ synthesizer_agent took {time.time() - start:.4f}s")
    return{
        "final_answer": result.final_answer,
        "try_no": state["try_no"] + 1 
    }
    

@track_node_metrics("ValidatorAgent")
def validator_agent(state: AgentState) -> dict:
    start = time.time()
    print(">>> Validator Agent called")
    system_prompt = SystemMessage(content=validator_prompt)
    #replace with question
    user_question = state["user_question"]
    final_answer = state["final_answer"]
    compressed_retrieved_results = compress_retrieved_results(state["retrieved_results"])
    compressed_reranked_results = compress_reranked_results(state["reranked_results"])

    recent_messages = state.get("messages", [])[:-1][-5:]
    context_transcript = "\n".join([
        f"{msg.type.capitalize()}: {msg.content}" 
        for msg in recent_messages
    ]) if recent_messages else "No prior conversation context."

    message = HumanMessage(content=f"""
Conversation History:
{context_transcript}

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
    
    print(f"⏱ validator_agent took {time.time() - start:.4f}s")
    return {
        "validation_result": validation_result
    }


#Reset nodes for retry
def reset_retriever_node(state:AgentState) -> dict:
    start = time.time()
    print(">>> Resetting stale retrieval state for retry...")
    print(f"⏱ reset_retriever_node took {time.time() - start:.4f}s")
    return{
        "retrieved_results": Overwrite([]),
        "reranked_results": Overwrite([])
    }

def reset_rerank_node(state: AgentState) -> dict:
    start = time.time()
    print(">>> Resetting stale rerank state for retry...")
    print(f"⏱ reset_rerank_node took {time.time() - start:.4f}s")
    return {
        "reranked_results": Overwrite([])
    }


#fanout function
def retrieve_fan_out(state: AgentState):
    return[
        Send(
            "retriever",
            {"question": question}
        )
        for question in state["sub_questions"]
    ]



#conditional routers
def initial_router(state: AgentState) -> str:
    if state.get("error_occured", False):
        return "quit"
    if state.get("need_clarification", False):
        return "clarify"
    elif state.get("need_retrieval", False):
        return "need_retrieval"         #goes to decomposer
    else:
        return "quit"



def retry_router(state: AgentState):
    validation = state["validation_result"]
    if validation["is_valid"]:
        return "END"
    
    if state.get("try_no", 1) >= state.get("max_try", 1):
        return "END"

    return validation["failure_type"]



graph = StateGraph(AgentState)

graph.add_node("initial", initial_agent)
graph.add_node("decomposer", decomposer_agent)
graph.add_node("retriever", retriever_agent)
graph.add_node("reranker", reranker_agent)
graph.add_node("synthesizer", synthesizer_agent)
graph.add_node("validator", validator_agent)
graph.add_node("reset_retriever", reset_retriever_node)
graph.add_node("reset_reranker", reset_rerank_node)

graph.add_edge(START, "initial")

graph.add_conditional_edges(
    "initial",
    initial_router,
    {
        "clarify": END,
        "need_retrieval": "decomposer",
        "quit": END
    }
)

graph.add_conditional_edges(
    "decomposer",
    retrieve_fan_out,
    ["retriever"]
)

graph.add_edge("retriever", "reranker")
graph.add_edge("reranker", "synthesizer")
graph.add_edge("synthesizer", "validator")

graph.add_conditional_edges(
    "validator",
    retry_router,
    {
        "END": END,
        "decomposition": "decomposer",
        "retrieval": "reset_retriever",
        "reranking": "reset_reranker",
        "synthesis": "synthesizer"
    }
)

graph.add_conditional_edges(
    "reset_retriever",
    retrieve_fan_out,
    ["retriever"]
)
graph.add_edge("reset_reranker", "reranker")

main = graph.compile()

# Save graph visualization to Mermaid file (.mmd)
with open("graph.mmd", "w", encoding="utf-8") as f:
    f.write(main.get_graph().draw_mermaid())
print("graph.mmd was written successfully.")

# Save graph visualization as PNG image (.png) with fallback if dependencies are missing
try:
    with open("graph.png", "wb") as f:
        f.write(main.get_graph().draw_mermaid_png())
    print("graph.png was written successfully.")
except Exception:
    print("PNG skipped: install pyppeteer (pip install pyppeteer) to render PNG. Mermaid file was still written to graph.mmd.")


# --- TEST THE AGENT ---
print("\n--- Running Agent ---")
user_query = "tell me about ethiopias geography" # Ask a question about your 2-page PDF

# We must initialize the state. 
# Note: Make sure 'max_retry' is here so your retry_router works!
initial_state = {
    "messages": [HumanMessage(content=user_query)],
    "max_retry": 3,
    "retry_count": 0,
    "try_no": 0,
}

# Run the graph
result = main.invoke(initial_state)

# Print the final answer
print("\n--- FINAL ANSWER ---")
print(result.get("final_answer"))
