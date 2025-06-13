import json
from typing import List, Dict, Optional

MESSAGE_HISTORY_FILE = "message_history.json"

convo = []  # Global variable to hold conversation history

def load_convo() -> List[Dict]:
    return convo

def add_to_convo(msg: Dict):
    convo.append(msg)


def load_message_history(filename: str = MESSAGE_HISTORY_FILE) -> List[Dict]:
    """Load message history from a file."""
    try:
        with open(filename, "r") as f:
            data = json.load(f)
            return data.get("message_history", [])
    except (FileNotFoundError, json.JSONDecodeError):
        return []

def save_message_history(history: List[Dict], filename: str = MESSAGE_HISTORY_FILE):
    """Save message history to a file."""
    with open(filename, "w") as f:
        data = {"message_history": history}
        json.dump(data, f, indent=2)


def remove_from_history(conversation_id: int, filename: str = MESSAGE_HISTORY_FILE):
    """Remove a conversation entry from the message history by ID."""
    history = load_message_history(filename)
    history = [entry for entry in history if entry["id"] != conversation_id]
    save_message_history(history, filename)

def add_to_history(prompt: str, response: str, filename: str = MESSAGE_HISTORY_FILE):
    """Add a conversation entry to the message history."""
    history = load_message_history(filename)
    conversation_id = len(history) + 1  # Simple ID based on length
    entry = {
        "id": conversation_id,
        "prompt": prompt,
        "response": response
    }
    history.append(entry)
    save_message_history(history, filename)