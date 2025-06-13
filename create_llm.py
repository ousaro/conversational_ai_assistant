from langchain_ollama import OllamaLLM
from env import CHAT_MODEL

def invoke_llm(prompt: str | list[dict], model: str = CHAT_MODEL) -> str:
    """Invoke the LLM and get a single response."""
    try:
        llm = OllamaLLM(model=model, temperature=0.7)
        return llm.invoke(prompt)
    except Exception as e:
        return f"LLM Error: {e}"

def stream_llm(prompt: str, model: str = CHAT_MODEL):
    """Stream LLM response chunks."""
    try:
        llm = OllamaLLM(model=model, temperature=0.7)
        return llm.stream(prompt)
    except Exception as e:
        return f"LLM Stream Error: {e}"
