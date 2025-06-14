import json
from typing import List, Dict, Optional

MESSAGE_HISTORY_FILE = "message_history.json"

convo = []  # Global variable to hold in-memory conversation history

def load_convo() -> List[Dict]:
    """
    Get the in-memory conversation history (for the current session).

    Returns:
        List[Dict]: List of conversation message dicts.
    """
    return convo

def add_to_convo(msg: Dict):
    """
    Append a message to the in-memory conversation history.

    Args:
        msg (Dict): A message dictionary to add (format: {role, content}).
    """
    convo.append(msg)

def load_message_history(filename: str = MESSAGE_HISTORY_FILE) -> List[Dict]:
    """
    Load conversation message history from the specified JSON file.

    Args:
        filename (str): Path to JSON file with message history.

    Returns:
        List[Dict]: Loaded list of conversation history entries.
    """
    try:
        with open(filename, "r") as f:
            data = json.load(f)
            return data.get("message_history", [])
    except (FileNotFoundError, json.JSONDecodeError):
        return []  # Return empty list if file doesn't exist or is corrupted

def save_message_history(history: List[Dict], filename: str = MESSAGE_HISTORY_FILE):
    """
    Save the provided message history list to a JSON file.

    Args:
        history (List[Dict]): The conversation history to save.
        filename (str): Destination file name.
    """
    with open(filename, "w") as f:
        data = {"message_history": history}
        json.dump(data, f, indent=2)

def remove_from_history(conversation_id: int, filename: str = MESSAGE_HISTORY_FILE):
    """
    Remove a conversation entry from history by ID (usually removes last if using length).

    Args:
        conversation_id (int): The ID of the conversation to remove.
        filename (str): Path to the history file.
    """
    history = load_message_history(filename)
    # Filter out the conversation with matching ID
    history = [entry for entry in history if entry["id"] != conversation_id]
    save_message_history(history, filename)

def add_to_history(prompt: str, response: str, filename: str = MESSAGE_HISTORY_FILE):
    """
    Add a user/assistant turn to persistent history for durable memory.

    Args:
        prompt (str): The user prompt/input.
        response (str): The assistant response/output.
        filename (str): File to store the updated history.
    """
    history = load_message_history(filename)
    conversation_id = len(history) + 1  # Simple incremental ID (may repeat if deleted)
    entry = {
        "id": conversation_id,
        "prompt": prompt,
        "response": response
    }
    history.append(entry)
    save_message_history(history, filename)
