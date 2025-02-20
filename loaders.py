# loaders.py
import os
import glob
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP

def add_metadata(doc, doc_type):
    doc.metadata["doc_type"] = doc_type
    return doc

def load_documents(base_path="Root/*"):
    folders = glob.glob(base_path)
    documents = []
    for folder in folders:
        doc_type = os.path.basename(folder)
        loader = DirectoryLoader(folder, glob="**/*.pdf", loader_cls=PyPDFLoader)
        folder_docs = loader.load()
        documents.extend([add_metadata(doc, doc_type) for doc in folder_docs])
    return documents

# def load_documents(base_path="Root"):
#     loader = DirectoryLoader(base_path, glob="*.pdf", loader_cls=PyPDFLoader)
#     documents = loader.load()
#     documents = [add_metadata(doc, "Manual") for doc in documents]
#     return documents

def split_documents(documents, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    # text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(documents)
