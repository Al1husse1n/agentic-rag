from typing import TypedDict, Annotated, Sequence, Optional, NotRequired, Literal
from langchain_core.messages import BaseMessage, HumanMessage
from pydantic import BaseModel, Field
from operator import add as add_messages
import operator
from langchain_core.documents import Document


class RetrievedResult(TypedDict):
    question: str
    documents: list[Document]

class RerankedResult(TypedDict):
    question: str
    documents: list[Document]

class ValidationResult(TypedDict):
    is_valid: bool
    
    failure_type: Literal[
        "none",
        "decomposition",
        "retrieval",
        "reranking",
        "synthesis",
    ]

    reason: str


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    user_question: NotRequired[str]
    try_no: NotRequired[int] 
    max_try: int
    need_retrieval: NotRequired[bool]
    need_clarification: NotRequired[bool]
    question: NotRequired[str]                  #used for both retriever and reducer
    error_occured: NotRequired[bool]
    documents: NotRequired[list[Document]]
    is_single_question: NotRequired[bool]
    sub_questions: NotRequired[list[str]]
    retrieved_results: NotRequired[Annotated[list[RetrievedResult], operator.add]]          # we may need the pre-reranking candidates for retry logic, debugging, evaluation, fallback if retrieval fails
    reranked_results: NotRequired[Annotated[list[RerankedResult], operator.add]]   
    final_answer: NotRequired[str] 
    validation_result: NotRequired[ValidationResult]
    

class InitialResponse(BaseModel):
    """Route the user query to the appropriate pipeline."""
    need_retrieval: bool = Field(
        description="Set to True if the query asks about Ethiopia or needs document retrieval. Set to False if it is a general question."
    )
    need_clarification: bool = Field(
        description="True if the user's question is too vague, ambiguous, or lacks necessary context and requires asking a follow-up question."
    )
    response: Optional[str] = Field(
        default=None,
        description= "Provide the direct answer IF need_retrieval is False, OR provide the exact clarification question IF need_clarification is True. Leave as null/None ONLY when need_retrieval is True.."
    )


class DecomposerResponse(BaseModel):
    """decompose questions into sub-questions"""
    sub_questions: list[str] = Field(
        description="Provide a list of sub-questions(or a single question) decomposed from the user question"
    )


class SynthesizerResponse(BaseModel):
    final_answer: str = Field(
        description="Return ONLY the final answer that should be shown to the user"
    )

class ValidatorResponse(BaseModel):
    is_valid: bool

    failure_type: Literal[
        "none",
        "decomposition",
        "retrieval",
        "reranking",
        "synthesis",
    ]

    reason: str

