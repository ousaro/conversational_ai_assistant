import ast
from tqdm import tqdm

from vector_db_utils import reset_db, create_db
from llm_utils import stream_llm, invoke_llm
from prompt_utils import build_prompt_from_convo, create_query_convo, create_system_prompt
from conversation_memory import (
    load_convo, load_message_history, add_to_history, add_to_convo, remove_from_history
)

from colorama import Fore, Style, init
init(autoreset=True)  # Automatically reset colorama colors after each print

# Load conversation and message history from persistent storage
convo = load_convo()
message_history = load_message_history()

def stream_response(prompt, RECALL=False):
    """
    Stream the LLM response for a given prompt and display it to the user in real time.
    Optionally saves the interaction to message history.

    Args:
        prompt (str): The input from the user.
        RECALL (bool): If True, does not save to history (used for recalling past info).
    """
    print(Fore.MAGENTA + "[INFO] Starting stream response..." + Style.RESET_ALL)
    full_prompt = build_prompt_from_convo(convo)  # Build conversation context prompt
    stream = stream_llm(full_prompt)  # Start LLM streaming

    response = ''
    print(Fore.LIGHTGREEN_EX + "\nASSISTANT: " + Style.RESET_ALL, end='')

    for chunk in stream:
        response += chunk
        print(chunk, end='', flush=True)  # Print streamed chunks in real time

    print(Style.RESET_ALL)
    print(Fore.CYAN + "\n[INFO] Finished streaming response." + Style.RESET_ALL)
    add_to_convo({"role": "assistant", "content": response})  # Save response to convo context

    if not RECALL:
        print(Fore.YELLOW + "[INFO] Saving to history..." + Style.RESET_ALL)
        add_to_history(prompt, response)  # Save conversation to persistent history


def create_vector_db():
    """
    Creates or resets the vector database with texts from message history
    for retrieval-augmented generation.

    Returns:
        vector_db: An object representing the vector store/database.
    """
    print(Fore.MAGENTA + "[INFO] Creating vector database from history..." + Style.RESET_ALL)

    conversations = message_history
    vector_db = reset_db()  # Fresh vector database

    for conv in conversations:
        cid = conv['id']
        combined = f"{conv['prompt']} {conv['response']}"
        print(Fore.GREEN + f"[VECTOR] Adding conversation ID {cid}" + Style.RESET_ALL)
        vector_db.add_texts([combined], ids=[f"{cid}"])

    print(Fore.CYAN + "[INFO] Vector DB created." + Style.RESET_ALL)
    return vector_db


def retrieve_relevant_embeddings(queries, results_per_query=2, similarity_threshold=0.5):
    """
    Retrieve relevant text snippets from the vector DB based on similarity to queries.

    Args:
        queries (list): List of search query strings.
        results_per_query (int): Number of top results per query.
        similarity_threshold (float): Upper limit for similarity score.

    Returns:
        list: Relevant content snippets or a default message if none found.
    """
    print(Fore.MAGENTA + "[INFO] Retrieving relevant embeddings..." + Style.RESET_ALL)
    print(Fore.LIGHTYELLOW_EX + "Generated queries: " + Style.RESET_ALL, queries)

    results = []
    seen_hashes = set()
    vector_db = create_db()  # Open existing vector DB

    for query in tqdm(queries, desc="Retrieving embeddings"):
        matches = vector_db.similarity_search_with_score(query, k=results_per_query)
        for doc, score in matches:
            content = getattr(doc, 'page_content', str(doc))  # Handle doc object or string
            if content not in seen_hashes and score < similarity_threshold:
                print(Fore.LIGHTBLUE_EX + f"[MATCH] {content} (Score: {score})" + Style.RESET_ALL)
                seen_hashes.add(content)
                results.append(content)

    if not results:
        print(Fore.RED + "[WARNING] No relevant matches found." + Style.RESET_ALL)
        results.append("No relevant information found.")

    return results


def create_queries(prompt):
    """
    Uses an LLM to generate search queries from a user prompt.

    Args:
        prompt (str): The user prompt to convert into search queries.

    Returns:
        list: List of search query strings.
    """
    print(Fore.MAGENTA + "[INFO] Creating search queries using LLM..." + Style.RESET_ALL)

    query_convo = create_query_convo(prompt)
    response = invoke_llm(query_convo)

    try:
        # LLM's response is expected to be a Python list (string). Parse safely.
        parsed = ast.literal_eval(response)
        print(Fore.GREEN + f"[QUERIES] Parsed queries: {parsed}" + Style.RESET_ALL)
        return parsed
    except (SyntaxError, ValueError):
        print(Fore.RED + "[ERROR] Could not parse queries. Using original prompt." + Style.RESET_ALL)
        return [prompt]


def remove_last_conversation():
    """
    Remove the most recent conversation from message history.
    """
    history = load_message_history()
    remove_from_history(len(history))


def recall_conversation(prompt):
    """
    Retrieve and inject relevant memory/context into the current conversation.

    Args:
        prompt (str): The user prompt for which memory recall is requested.
    """
    print(Fore.MAGENTA + "[INFO] Initiating memory recall..." + Style.RESET_ALL)
    queries = create_queries(prompt)
    context = retrieve_relevant_embeddings([prompt, *queries])

    print(Fore.CYAN + "[MEMORY] Adding recalled memories to conversation context..." + Style.RESET_ALL)
    add_to_convo({"role": "user", "content": f"MEMORIES: {context} \n\n USER PROMPT: {prompt}"})


def run_assistant():
    """
    Start and manage the main assistant interaction loop for user input and commands.

    Commands available:
      /recall [prompt] - Recall similar memories for a prompt
      /forget          - Remove last conversation
      /exit or /quit   - Exit the assistant
    """
    print(Fore.MAGENTA + "[INFO] Assistant ready. Type your prompt or use commands." + Style.RESET_ALL)
    print(Fore.YELLOW + "Commands: /recall [prompt], /forget, /exit" + Style.RESET_ALL)

    # Add initial system instruction/context
    add_to_convo({"role": "system", "content": create_system_prompt()})

    while True:
        user_input = input(Fore.LIGHTWHITE_EX + "USER: " + Style.RESET_ALL).strip()

        if user_input.lower().startswith('/recall'):
            new_prompt = user_input[8:].strip()
            recall_conversation(new_prompt)
            stream_response(new_prompt, RECALL=True)
        elif user_input.lower().startswith('/forget'):
            remove_last_conversation()
            print(Fore.YELLOW + "[INFO] Last conversation removed.\n" + Style.RESET_ALL)
        elif user_input.lower() in ['/exit', '/quit']:
            print(Fore.GREEN + "[INFO] Exiting the assistant. Goodbye!" + Style.RESET_ALL)
            break
        else:
            add_to_convo({"role": "user", "content": user_input})
            stream_response(user_input, RECALL=False)
