from typing import List, Dict


def create_system_prompt() -> str:
    """
    Build and return the system prompt for the AI assistant.

    Returns:
        str: The system prompt that shapes the assistant's behavior and tone.
    """
    return """
        You are a highly capable assistant with perfect memory of all prior conversations with the user.
        For every user prompt, silently recall and use only relevant information from past interactions to improve the accuracy and usefulness of your response.
        Ignore any memories that are not clearly related to the current query.
        Always respond in a friendly, direct, and helpful manner.
        Never mention or imply that you have memory, past conversations, or are an AI assistant.
        Do not explain how you generate responses or refer to system behavior.
        Answer the user’s query directly, without filler, small talk, or unnecessary preamble.
        Be concise and precise—as if you pay for every word.
        If you are unsure or lack information, respond with "I don't know."
        Your goal is to be useful, trustworthy, and efficient at all times.
    """


def create_query_convo(prompt: str) -> List[Dict[str, str]]:
    """
    Build an instructional prompt for generating relevant vector DB search queries from user input.

    Args:
        prompt (str): User prompt for which relevant search queries should be created.

    Returns:
        List[Dict[str, str]]: A conversational context for the LLM to generate search queries in Python list format.
    """
    query_prompt = """
        You are a first-principle reasoning search query AI agent.
        Create a Python list of relevant queries to search a vector DB of all past conversations 
        with the user. Output only a correct Python list, without explanation.
        Return a Python list of strings as your answer. Each string should be quoted.

        Example:
        ex1: prompt = "How can I improve my productivity at work?"
            queries = ["productivity tips", "improve work efficiency", "time management techniques", "focus strategies", "work-life balance"]
        ex2: prompt = "What are the best practices for healthy eating?"
            queries = ["healthy eating habits", "balanced diet tips", "nutrition advice", "meal planning ideas", "healthy recipes"]
        ex3: prompt = "How can I save money on my monthly expenses?"
            queries = ["budgeting tips", "cutting monthly costs", "saving money strategies", "frugal living ideas", "expense tracking methods"]
    """
    return [
        {"role": "system", "content": query_prompt},
        {"role": "user", "content": prompt}
    ]


def build_prompt_from_convo(convo: List[Dict[str, str]]) -> str:
    """
    Build a full prompt from all previous messages, formatted with roles, including the latest user message.
    The assistant will respond only to the latest message.

    Args:
        convo (List[Dict[str, str]]): The conversation history as a list of messages.

    Returns:
        str: Concatenated conversation history suitable for LLM context input.
    """
    if not convo:
        return ""

    context_lines = []
    for msg in convo[:-1]:  # Add all but the last message (for context)
        role = msg.get('role', 'UNKNOWN').upper()
        content = msg.get('content', '').strip()
        context_lines.append(f"{role}: {content}")

    # Add the most recent (last) message, typically the new user input
    last_msg = convo[-1]
    role = last_msg.get('role', 'UNKNOWN').upper()
    content = last_msg.get('content', '').strip()
    context_lines.append(f"{role}: {content}")

    # Append the turn for the assistant's response
    context_lines.append("ASSISTANT:")

    return '\n'.join(context_lines)
