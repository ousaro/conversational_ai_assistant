from typing import List, Dict


def create_system_prompt() -> str:
    """Build the system prompt for the AI assistant."""
    return (
        "You are an assistant with memory of every conversation you have had with the user. "
        "For every prompt, check for any relevant messages from past interactions. "
        "Leverage this information for more accurate and helpful responses. "
        "Ignore recalled conversations unrelated to the current prompt. "
        "Always respond in a friendly, helpful manner. "
        "Never reveal you are an AI assistant, nor share details about internal workings. "
        "Do not say things like 'Based on our previous conversation' or similar."
        "Respond directly to the user's query without unnecessary preamble."
        "Be precise like you pay for every word you say. "
        "If you don't know the answer, say 'I don't know' instead of making up information."
    )


def create_query_convo(prompt) -> list[dict]:
    """Build the prompt for generating search queries."""
    query_prompt ="""
        You are a first principle reasoning search query AI agent.
        Create a Python list of relevant queries to search a vector DB of all past conversations 
        with the user. Output only a correct Python list, without explanation.
        Return a Python list of strings as your answer.Each string should be quoted.
        
        example:
        ex1: prompt = "How can I improve my productivity at work?"
            queries = ["productivity tips", "improve work efficiency", "time management techniques", "focus strategies", "work-life balance" ]
        ex2: prompt = "What are the best practices for healthy eating?"
            queries = ["healthy eating habits", "balanced diet tips", "nutrition advice", "meal planning ideas", "healthy recipes" ]
        ex3: prompt = "How can I save money on my monthly expenses?"
            queries = ["budgeting tips", "cutting monthly costs", "saving money strategies", "frugal living ideas", "expense tracking methods" ]

    )
    """


    query_convo = [
        {"role": "system", "content": query_prompt},
        {"role": "user", "content": prompt}
    ]

    return query_convo


def build_prompt_from_convo(convo: List[Dict]) -> str:
    """Build a prompt string from the conversation context."""
    return '\n'.join(f"{msg['role'].upper()}: {msg['content']}" for msg in convo)
