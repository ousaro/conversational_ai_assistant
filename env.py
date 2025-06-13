import os

from dotenv import load_dotenv

load_dotenv()

# Helper to fetch environment variables or raise a friendly error
def require_env(var_name):
    value = os.getenv(var_name)
    if value is None:
        raise EnvironmentError(f"Missing required env variable: {var_name}")
    return value

EMBEDDING_MODEL = require_env("EMBEDDING_MODEL") # model for embeddings
DATABASE_LOCATION = require_env("DATABASE_LOCATION") # the path to the database directory
COLLECTION_NAME = require_env("COLLECTION_NAME") # the name of the collection in the database
CHAT_MODEL=require_env("CHAT_MODEL") # model for chat interactions
