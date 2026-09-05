import os
import requests
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_tavily import TavilySearch
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing import TypedDict, List
from pydantic import BaseModel, Field
from langgraph.checkpoint.postgres import  PostgresSaver
from langgraph.checkpoint.serde.encrypted import EncryptedSerializer
from psycopg import Connection
from psycopg.rows import dict_row


load_dotenv()





