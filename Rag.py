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


def pdf_loader(pdf_path):

    if not os.path.exists(pdf_path):
        raise FileNotFoundError("File Not Found")

    loader =  PyPDFLoader(pdf_path)
    
    try:
        doc_list =  loader.load()
        print(f"Length of the document : {len{doc_list}}")

        spliter = RecursiveCharacterTextSplitter(
            chunk_size = 1000,
            chunk_overlap = 100
        )
        splitted_chunks = spliter.split_documents()
        print(f"Length of chunks : {len(splitted_chunks)}")
    except Exception as e:
        raise ValueError(f"Error while splitting documents")
    
    return splitted_chunks

def web_loader(url):

    urls = [url]
    all_pages = []

    loader = WebBaseLoader()

    try:
        for url in urls:
            doc_list = loader.load(url)
            print(f"Length of the Document")
            spilter = RecursiveCharacterTextSplitter(
                chunk_size = 1000,
                chunk_overlap = 100
            )
            splitted_chunks = spilter.split_documents(doc_list)
            print(f"Length of Chunks : {len(splitted_chunks)}")
    except Exception as e:
        raise ValueError("Error while loading url {e}")
    
    return splitted_chunks

select_document = input("Select Your Document Type (url or pdf) : ")

if select_document.strip().lower() == "pdf":
    path = input("Enter the path of pdf : ")
    chunks = pdf_loader(path)
elif select_document.strip().lower() == "url":
    url = input("Paste your url : ")
    chunks - web_loader(url)
else:
    print("File type not valid")



