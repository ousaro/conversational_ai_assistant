from assistant import run_assistant, create_vector_db


def main():
    # Initialize on startup
    create_vector_db()
    run_assistant()

if __name__ == "__main__":
    main()
