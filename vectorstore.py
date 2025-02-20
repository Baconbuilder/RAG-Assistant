# vectorstore.py
import os
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from config import DB_NAME

def create_vectorstore(documents, db_name=DB_NAME):
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    if os.path.exists(db_name):
        Chroma(persist_directory=db_name, embedding_function=embeddings).delete_collection()
    vectorstore = Chroma.from_documents(documents=documents, embedding=embeddings, persist_directory=db_name)
    return vectorstore


