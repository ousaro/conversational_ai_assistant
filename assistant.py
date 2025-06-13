import ast

from tqdm import tqdm
from colorama import Fore, Style

from create_db import reset_db, create_db
from create_llm import stream_llm, invoke_llm
from prompt_builder import build_prompt_from_convo, create_query_convo, create_system_prompt
from conversation_memory import load_convo, load_message_history, add_to_history, add_to_convo, remove_from_history

convo = load_convo()
message_history = load_message_history()

def stream_response(prompt, RECALL=False):
    """Stream the LLM response live while appending to conversation and database."""

    full_prompt = build_prompt_from_convo(convo)
    stream = stream_llm(full_prompt)

    response = ''
    print(Fore.LIGHTGREEN_EX + "\nASSISTANT: " + Style.RESET_ALL, end='')

    for chunk in stream:
        response += chunk
        print(chunk, end='', flush=True)

    print(Style.RESET_ALL)  # Ensures color reset and new line
    add_to_convo({"role": "assistant", "content": response})
    if not RECALL:
        # Only add to history if this is not a recall operation
        add_to_history(prompt, response)


def create_vector_db():

    conversations = message_history
    vector_db = reset_db()

    for conv in conversations:
        cid = conv['id']
        vector_db.add_texts([f"{conv['prompt']} {conv['response']}"], ids=[f"{cid}"])

    return vector_db


def retrieve_relevant_embeddings(queries, results_per_query=2, similarity_threshold=0.4):
    """Retrieve relevant chunks from the vector database, filtered by classifier."""
    print(Fore.LIGHTYELLOW_EX + "Generated queries: " + Style.RESET_ALL, queries)
    results = []
    seen_hashes = set()  # To avoid duplicates

    vector_db = create_db()

    for query in tqdm(queries, desc="Retrieving embeddings"):
        matches = vector_db.similarity_search_with_score(query, k=results_per_query)
        for doc, score in matches:  # assume (doc, score)
            content = getattr(doc, 'page_content', str(doc))
            if content not in seen_hashes and score < similarity_threshold:
                print(Fore.LIGHTBLUE_EX + f"Found relevant content: {content} with score {score}" + Style.RESET_ALL)
                seen_hashes.add(content)
                results.append(content)
    if not results:
        results.append("No relevant information found.")
    return results


def create_queries(prompt):
    """Generate retrieval search queries using the LLM, based on the latest prompt."""
    query_convo = create_query_convo(prompt)

    response = invoke_llm(query_convo)
    try:
        return ast.literal_eval(response)
    except (SyntaxError, ValueError):
        print("Error parsing queries. Using original prompt as fallback.")
        return [prompt]


def remove_last_conversation():
    """Remove the most recent conversation from message history."""
    history = load_message_history()
    remove_from_history(len(history))


def recall_conversation(prompt):
    """Recall relevant memory and add to conversation context before assistant reply."""
    queries = create_queries(prompt)
    context = retrieve_relevant_embeddings([prompt, *queries])
    add_to_convo({"role": "user", "content": f"MEMORIES: {context} \n\n USER PROMPT: {prompt}"})


def run_assistant():
    """Main REPL loop for interactive chat."""
    add_to_convo({"role": "system", "content": create_system_prompt()})
    while True:
        user_input = input("USER: ").strip()

        if user_input.lower().startswith('/recall'):
            new_prompt = user_input[8:].strip()
            recall_conversation(new_prompt)
            stream_response(new_prompt, RECALL=True)
        elif user_input.lower().startswith('/forget'):
            remove_last_conversation()
            print("Last conversation removed.\n")
        elif user_input.lower() in ['/exit', '/quit']:
            print("Exiting the assistant. Goodbye!")
            break
        else:
            add_to_convo({"role": "user", "content": user_input})
            stream_response(user_input, RECALL=False)


