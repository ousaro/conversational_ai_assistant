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
    """
    Create (or connect to) a persistent Chroma vector database for text embeddings.

    Args:
        embedding_model (str): Identifier for the embeddings model/service.
        collection_name (str): Name for the Chroma collection.
        persist_directory (str): Directory where the database files are stored.

    Returns:
        Chroma: An instance of the Chroma vector database class.
    """
    embeddings = OllamaEmbeddings(model=embedding_model)
    vector_db = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_directory
    )
    return vector_db

def reset_db(db_location: str = DATABASE_LOCATION) -> Chroma:
    """
    Delete and re-create the Chroma vector database (used to clear all history/data).

    Args:
        db_location (str): Path to the database directory.

    Returns:
        Chroma: New, empty instance of the Chroma vector database.
    """
    if os.path.exists(db_location):
        print(f"Resetting database at {db_location}...")
        shutil.rmtree(db_location)  # Delete all files in the database directory
    return create_db()
