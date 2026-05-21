from pathlib import Path
import gc
import time
import shutil
import uuid

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

CHROMA_BASE = "./chroma_db"


def get_embeddings(model: str = "nomic-embed-text") -> OllamaEmbeddings:
    return OllamaEmbeddings(model=model)


def process_pdf(file_path: str, embed_model: str = "nomic-embed-text"):

    loader = PyPDFLoader(file_path)
    docs   = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    for chunk in chunks:
        chunk.metadata["source_file"] = Path(file_path).name

    session_dir = f"{CHROMA_BASE}/{uuid.uuid4().hex}"

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(embed_model),
        persist_directory=session_dir,
        collection_name="doc_store",
    )
    return len(chunks), vectorstore, session_dir


def safe_clear(vectorstore_ref, session_dir: str):

    del vectorstore_ref
    gc.collect()
    time.sleep(0.8)

    path = Path(session_dir)
    if not path.exists():
        return

    for _ in range(5):
        try:
            shutil.rmtree(path)
            return
        except PermissionError:
            gc.collect()
            time.sleep(1.2)
