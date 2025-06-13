import os
import shutil
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from env import EMBEDDING_MODEL, COLLECTION_NAME, DATABASE_LOCATION

def create_db(
    embedding_model: str = EMBEDDING_MODEL,
    collection_name: str = COLLECTION_NAME,
    persist_directory: str = DATABASE_LOCATION
) -> Chroma:
    embeddings = OllamaEmbeddings(model=embedding_model)
    vector_db = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_directory
    )
    return vector_db

def reset_db(db_location: str = DATABASE_LOCATION) -> Chroma:
    if os.path.exists(db_location):
        print(f"Resetting database at {db_location}...")
        shutil.rmtree(db_location)
    return create_db()
