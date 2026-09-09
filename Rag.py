from urllib3 import response
import os
import requests
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from langchain_core.documents import Document
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
        print(f"Length of the document : {len(doc_list)}")

        spliter = RecursiveCharacterTextSplitter(
            chunk_size = 1000,
            chunk_overlap = 100
        )
        splitted_chunks = spliter.split_documents(doc_list)
        print(f"Length of chunks : {len(splitted_chunks)}")
    except Exception as e:
        raise ValueError(f"Error while splitting documents")
    
    return splitted_chunks

def web_loader(url):

    urls = [url]
    all_pages = []

    try:
        for url in urls:
            loader = WebBaseLoader(url)
            doc_list = loader.load()
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
    chunks = web_loader(url)
else:
    print("File type not valid")


llm = ChatGroq(
    model = "openai/gpt-oss-120b"
)
embeddings =  HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3"
)

persist_memory = r"C:\Users\JANARTHAN\Downloads\CorrectiveRag\Corrective-Rag"
collection_name = "SelfDB"

try:
    vectorstore = Chroma.from_documents(
        documents=chunks,
        persist_directory= persist_memory,
        collection_name= collection_name,
        embedding=embeddings
    )

except Exception as e:
    raise ValueError("Error while creating vectorstore")

retriever =  vectorstore.as_retriever(
    search_type = "mmr",
    search_kwargs = {
        "k" : 5,
        "fetch_k" : 10,
        "lambda_mult" : 0.5
    }
)

class AgentState(TypedDict):
    question : str
    document : list[str]
    generation : str
    web_search : bool

def retriever_node(state : AgentState) -> AgentState:

   question =  state['question']
   response = retriever.invoke(question)

   return {"document" : response}


class llm_schema(BaseModel):

    binary_score : str =  Field(..., description="yes or no, wheather document relevant to question or not")

llm_with_schema = llm.with_structured_output(llm_schema)

def grader_node(state : AgentState) -> AgentState:

    question = state['question']
    document = state['document']
    
    prompt = ChatPromptTemplate.from_messages([
           (
            "system",
            "You grade whether a retrieved document is relevant to a user question. "
            "Give 'yes' if it contains keywords or semantic meaning related to the "
            "question. This is a lenient filter to catch clearly irrelevant docs, "
            "not a strict correctness check.",
        ),
        ("human", "Retrieved document:\n\n{document}\n\nUser question: {question}"),
    ])

    chain = prompt | llm_with_schema

    web_search = False

    filtered_docs = []

    for doc in document:
        result = chain.invoke({
                            "document" : doc.page_content ,
                            "question" : question
                        })

        if result.binary_score.lower() == "yes":
            filtered_docs.append(doc)
        else: 
            web_search = True
    
    return {"document" : filtered_docs, "web_search" : web_search}


def should_continue(state : AgentState) -> str:

    document = state['document']

    if len(document) == 0:
        return "Transfrom_query_node"
    else:
        return "generator_node"


def transfrom_query_node(state : AgentState) -> AgentState:

    question = state['question']

    tranfrom_query_prompt = ChatPromptTemplate.from_messages([
        ("system",
            "You rewrite questions to be better optimized for web search. "
            "Look at the input and reason about the underlying semantic intent.",
        ),
        ("human", "Initial question:\n\n{question}\n\nFormulate an improved question."),
    ])

    chain = tranfrom_query_prompt | llm | StrOutputParser()

    result = chain.invoke({
        "question": question
    })

    return { "question" : result}

def websearch_node(state : AgentState) -> AgentState:

    question =  state['question']
    document = state['document']
    search = TavilySearch(k=5)
    doc = search.invoke(question) 

    context = [Document(page_content=d["content"]) for d in doc["results"]]
    
    return {"document" : document + context}



def generator_node(state : AgentState) -> AgentState:

    question = state['question']
    document = state['document']

    context = "\n\n".join(doc.page_content for doc in document)

    generate_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Answer the question using only the provided context. "
            "If the context doesn't contain the answer, say so plainly.",
        ),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ])

    chain = generate_prompt | llm | StrOutputParser()

    response =  chain.invoke({
        "question": question, "context": context
    })

    return {"generation" : response }




graph = StateGraph(AgentState)

graph.add_node("retriever", retriever_node)
graph.add_node("grader", grader_node)
graph.add_node("generator", generator_node)
graph.add_node("transform_query", transfrom_query_node)
graph.add_node("websearch", websearch_node)

graph.add_edge(START, "retriever")
graph.add_edge("retriever", "grader")
graph.add_conditional_edges(
    "grader",
    should_continue,
    {
        "Transfrom_query_node": "transform_query",
        "generator_node" : "generator"
    } 
)
graph.add_edge("transform_query", "websearch")
graph.add_edge("websearch", "generator")
graph.add_edge("generator", END)

Crag =  graph.compile()

print("=" * 60)
print("Corrective RAG Agent")
print("=" * 60)

while True:
    user_input =  input("Ask : ")

    if user_input == "exit":
        break

    result = Crag.invoke({
        "question" : user_input,
    })

    print(result['generation'])

















