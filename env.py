import os
from dotenv import load_dotenv

# Load environment variables from a .env file into the process environment
load_dotenv()

def require_env(var_name: str) -> str:
    """
    Fetch a required environment variable or raise an error if it's not set.

    Args:
        var_name (str): Name of the environment variable.

    Returns:
        str: The variable's value.

    Raises:
        EnvironmentError: If the variable is missing or not set.
    """
    value = os.getenv(var_name)
    if value is None:
        raise EnvironmentError(f"Missing required env variable: {var_name}")
    return value

# Model and database configuration fetched from environment for flexibility and security
EMBEDDING_MODEL = require_env("EMBEDDING_MODEL")         # Name of model for text embeddings
DATABASE_LOCATION = require_env("DATABASE_LOCATION")     # Filesystem path for persistent vector DB
COLLECTION_NAME = require_env("COLLECTION_NAME")         # Collection name inside the vector DB
CHAT_MODEL = require_env("CHAT_MODEL")                   # Name of chat LLM model to use
