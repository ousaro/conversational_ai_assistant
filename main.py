from assistant import run_assistant, create_vector_db

def main():
    """
    Program entry point.

    - Initializes the vector database at startup (ensures the DB is ready).
    - Starts the assistant's interaction loop.
    """
    create_vector_db()   # Prepare vector DB for use (may initialize or connect)
    run_assistant()      # Launch the main assistant logic (likely runs an interactive loop)

if __name__ == "__main__":
    main()
