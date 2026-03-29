"""
src/rag_system.py — RAG System Wrapper
Wraps the agri-crop-qa RAG pipeline and exposes a clean interface
for the evaluation system to use.
"""

import os
from typing import Optional

from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document


# Constants - loaded from environment


GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
VECTOR_DB_DIR: str = os.getenv("VECTOR_DB_DIR", "agri_db")
EMBEDDINGS_MODEL: str = os.getenv("EMBEDDINGS_MODEL", "models/gemini-embedding-001")
LLM_MODEL: str = os.getenv("LLM_MODEL", "gemini-2.5-flash")
NUM_RETRIEVED_DOCS: int = int(os.getenv("NUM_RETRIEVED_DOCS", "6"))
TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.0"))

if not GOOGLE_API_KEY:
    raise ValueError(
        "GOOGLE_API_KEY is not set. "
        "Add it to your .env file: GOOGLE_API_KEY=your_key_here"
    )


# Vector Store

def load_vector_store() -> Optional[Chroma]:
    """
    Load the existing Chroma vector store from disk.

    Returns:
        Chroma instance if the vector store exists, otherwise None.
    """
    if not (os.path.exists(VECTOR_DB_DIR) and os.listdir(VECTOR_DB_DIR)):
        print(f"No vector store found at '{VECTOR_DB_DIR}'.")
        return None

    print(f"Loading vector store from '{VECTOR_DB_DIR}'...")

    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDINGS_MODEL,
        google_api_key=GOOGLE_API_KEY,
        task_type="retrieval_document",
    )

    vectorstore = Chroma(
        persist_directory=VECTOR_DB_DIR,
        embedding_function=embeddings,
    )

    print("Vector store loaded successfully.")
    return vectorstore


# QA Chain
def create_qa_chain(
    vectorstore: Chroma,
    top_k: int = NUM_RETRIEVED_DOCS,
    temperature: float = TEMPERATURE,
):
    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        temperature=temperature,
        google_api_key=GOOGLE_API_KEY,
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an agricultural assistant. Answer ONLY using the provided context.\n"
            "If the answer is not in the context, say: 'I don't know based on the documents.'\n"
            "Keep answers concise and practical.\n\nContext:\n{context}",
        ),
        ("human", "{question}"),
    ])

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return {"chain": chain, "retriever": retriever}

# Ask a Question
def ask_question(qa_chain, question: str) -> dict:
    if not question or not question.strip():
        return {"answer": "", "sources": [], "error": "Empty question provided."}

    try:
        chain = qa_chain["chain"]
        retriever = qa_chain["retriever"]

        answer = chain.invoke(question)
        sources = retriever.invoke(question)

        return {"answer": answer, "sources": sources, "error": None}

    except Exception as e:
        print(f"ERROR during RAG query: {e}")
        return {"answer": "", "sources": [], "error": str(e)}