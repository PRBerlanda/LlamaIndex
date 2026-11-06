"""LLM configuration shared by the scripts of this project.

The key is read from the environment (never hard-code it):
    GROQ_API_KEY   your key from https://console.groq.com/keys
                   (GROQ_API is also accepted: it is the secret name used in the Colab notebooks)
    GROQ_MODEL     optional, default "openai/gpt-oss-20b"

Groq retires models from time to time (llama3-70b-8192, llama-3.1-70b-versatile and,
on the free tier, llama-3.3-70b-versatile are gone). Check the current list at
https://console.groq.com/docs/models and change GROQ_MODEL if needed.
"""
import os

DEFAULT_MODEL = "openai/gpt-oss-20b"


def groq_api_key() -> str:
    key = os.environ.get("GROQ_API_KEY") or os.environ.get("GROQ_API")
    if not key:
        try:  # running in Google Colab with a secret called GROQ_API
            from google.colab import userdata  # type: ignore

            key = userdata.get("GROQ_API")
        except Exception:
            key = None
    if not key:
        raise RuntimeError("Set the GROQ_API_KEY environment variable (see .env.example).")
    return key


def get_llm(**kwargs):
    """Return the Groq LLM used by the project."""
    from llama_index.llms.groq import Groq

    model = os.environ.get("GROQ_MODEL", DEFAULT_MODEL)
    return Groq(model=model, api_key=groq_api_key(), **kwargs)
