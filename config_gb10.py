"""
Configuration for using GB10 llama.cpp server with Step-3.5-Flash.

Usage:
    from config_gb10 import get_gb10_llm, get_gb10_embeddings, GB10_BASE_URL

    # For chat/completion
    llm = get_gb10_llm(max_tokens=4096)
    response = llm.invoke("Hello")

    # Or set environment variables before importing other modules:
    import os
    os.environ["OPENAI_API_BASE"] = "http://192.168.4.38:8080/v1"
    os.environ["OPENAI_API_KEY"] = "not-needed"
"""

import os
from typing import Optional

# GB10 Server Configuration
GB10_HOST = os.getenv("GB10_HOST", "192.168.4.38")
GB10_PORT = os.getenv("GB10_PORT", "8080")
GB10_BASE_URL = f"http://{GB10_HOST}:{GB10_PORT}/v1"
GB10_MODEL_NAME = "step3.5-flash"  # Can be any string, llama.cpp ignores it

# Context sizes for different use cases
CONTEXT_SIZES = {
    "kg_building": 4096,      # Entity extraction, chunking
    "qa_simple": 8192,        # Simple QA queries
    "qa_complex": 16384,      # Multi-hop reasoning
    "qa_full": 131072,        # Full 128K context (single request only)
}


def get_gb10_llm(
    max_tokens: int = 4096,
    temperature: float = 0.1,
    context_size: str = "kg_building",
):
    """
    Get a LangChain ChatOpenAI instance configured for GB10.

    Args:
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature (0.0-1.0)
        context_size: One of "kg_building", "qa_simple", "qa_complex", "qa_full"

    Returns:
        ChatOpenAI instance
    """
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=GB10_MODEL_NAME,
        base_url=GB10_BASE_URL,
        api_key="not-needed",  # llama.cpp doesn't require API key
        max_tokens=max_tokens,
        temperature=temperature,
        # Note: context size is set server-side (131K), not per-request
    )


def get_gb10_chat_client():
    """
    Get a raw OpenAI client for GB10.

    Returns:
        OpenAI client
    """
    from openai import OpenAI

    return OpenAI(
        base_url=GB10_BASE_URL,
        api_key="not-needed",
    )


def check_gb10_health() -> dict:
    """Check if GB10 server is healthy and ready."""
    import requests
    try:
        response = requests.get(f"http://{GB10_HOST}:{GB10_PORT}/health", timeout=5)
        return response.json()
    except Exception as e:
        return {"status": "error", "message": str(e)}


def get_gb10_model_info() -> dict:
    """Get model information from GB10 server."""
    import requests
    try:
        response = requests.get(f"http://{GB10_HOST}:{GB10_PORT}/v1/models", timeout=5)
        return response.json()
    except Exception as e:
        return {"error": str(e)}


# Quick test
if __name__ == "__main__":
    print(f"GB10 Server: {GB10_BASE_URL}")
    print(f"Health: {check_gb10_health()}")

    # Test completion
    llm = get_gb10_llm(max_tokens=50)
    response = llm.invoke("What is 2+2? Answer briefly.")
    print(f"Test response: {response.content}")
