from typing import TypedDict, Annotated, Sequence, Optional, NotRequired
from langchain_core.messages import BaseMessage, HumanMessage
from pydantic import BaseModel, Field
from operator import add as add_messages
import operator

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    try_no: NotRequired[int] 
    max_retry: int
    need_retrieval: NotRequired[bool]
    question: NotRequired[HumanMessage]
    is_single_question: NotRequired[bool]
    retrieved_docs: NotRequired[list] 
    reranked_docs: NotRequired[list[str]] 
    final_doc: NotRequired[str]   
    final_answer: NotRequired[str] 


class InitialResponse(BaseModel):
    """Route the user query to the appropriate pipeline."""
    need_retrieval: bool = Field(
        description="Set to True if the query asks about Ethiopia or needs document retrieval. Set to False if it is a general question."
    )
    response: Optional[str] = Field(
        default=None,
        description="Provide a complete direct answer ONLY if need_retrieval is False. Leave as null/None if need_retrieval is True."
    )

class ClassifierResponse(BaseModel):
    is_single_question: bool = Field(
        description="Set to True if the user is asking one information-seeking question and Set to False if the user is asking two or more independently answerable questions that may require different evidence or documents."
    )

class ValidatorResponse(BaseModel):
    """Retrieve the document that answers the user question, if found"""
    doc_found: bool = Field(
        description="Set to True if there is a document that answers or matches the question from the user"
    )

    document: Optional[str] = Field(
        default=None,
        description="Provide the best document that answers or matches the question from the user, ONLY if doc_found is True"     
    )

