from dotenv import load_dotenv          #pip install python-dotenv
import os
from langgraph.graph import StateGraph, END,START         #pip install langgraph
from typing import TypedDict, Annotated, Sequence, Optional
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage, AIMessage      #	pip install langchain-core
from operator import add as add_messages
from langgraph.prebuilt import ToolNode
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings         #pip install langchain-google-genai

print("Hissd")