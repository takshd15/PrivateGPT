import requests

from app.config import OPENAI_API_KEY, OPENAI_MODEL

_CHAT_URL = "https://api.openai.com/v1/chat/completions"
_EMBEDDINGS_URL = "https://api.openai.com/v1/embeddings"
_EMBEDDING_MODEL = "text-embedding-3-small"


def ask_llm(
    system_prompt: str,
    user_prompt: str,
    json_mode: bool = False,
    timeout: int = 120,
    num_predict: int | None = None,
) -> str:
    if not OPENAI_API_KEY:
        raise ValueError("OpenAI API key is not configured in .env.")

    payload = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }

    # json_mode requests strict JSON output. OpenAI requires the word "json" to
    # appear somewhere in the prompt when this is set; every caller's prompt
    # already says so (it's what made json_mode meaningful under Ollama too).
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    if num_predict is not None:
        payload["max_tokens"] = num_predict

    response = requests.post(
        _CHAT_URL,
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()

    return response.json()["choices"][0]["message"]["content"]


def embed(text: str, timeout: int = 10) -> list[float]:
    """Return a text-embedding-3-small (1536-dim) embedding for ``text``."""
    if not OPENAI_API_KEY:
        raise ValueError("OpenAI API key is not configured in .env.")

    response = requests.post(
        _EMBEDDINGS_URL,
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        },
        json={"model": _EMBEDDING_MODEL, "input": text},
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()["data"][0]["embedding"]


def warmup() -> None:
    """No-op: OpenAI's API has no local model to keep resident."""
    return
