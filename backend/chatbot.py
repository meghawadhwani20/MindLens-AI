import os
from typing import Optional

import httpx
import openai
from dotenv import dotenv_values, load_dotenv

load_dotenv()

SYSTEM_PROMPT = (
    "You are a helpful and empathetic wellness assistant for MindLens AI. "
    "You provide support for mental health, stress management, sleep, and general wellness. "
    "Be concise, kind, supportive, and professional. Do not claim to diagnose medical conditions."
)

DEFAULT_HF_MODEL = "deepseek-ai/DeepSeek-V3-0324"
HF_ROUTER_BASE_URL = "https://router.huggingface.co/v1"

_client = None
_provider = None
_initialization_error = None


def _get_env_value(*names: str) -> str:
    env_values = dotenv_values()
    for name in names:
        value = (os.environ.get(name) or env_values.get(name) or "").strip()
        if value:
            return value
    return ""


def _resolve_provider_and_key() -> tuple[Optional[str], str]:
    hf_key = _get_env_value("HUGGINGFACE_API_KEY", "HF_API_KEY")
    openai_key = _get_env_value("OPENAI_API_KEY")

    if hf_key:
        return "huggingface", hf_key

    if openai_key.startswith("hf_"):
        return "huggingface", openai_key

    if openai_key:
        return "openai", openai_key

    return None, ""


def init_chatbot():
    """
    Initializes the chatbot provider client.
    Returns the client instance or None if initialization fails.
    """
    global _client, _provider, _initialization_error

    provider, api_key = _resolve_provider_and_key()

    if not api_key:
        _provider = None
        _initialization_error = (
            "No chatbot API key found. Set HUGGINGFACE_API_KEY for Hugging Face or OPENAI_API_KEY for OpenAI."
        )
        print(f"Chatbot Warning: {_initialization_error}")
        return None

    if api_key.lower() in ["undefined", "null", "none", ""]:
        _provider = None
        _initialization_error = "Chatbot API key is invalid or empty."
        print(f"Chatbot Warning: {_initialization_error}")
        return None

    if provider == "huggingface":
        model_name = _get_env_value("HUGGINGFACE_MODEL") or DEFAULT_HF_MODEL
        try:
            _client = openai.OpenAI(
                api_key=api_key,
                base_url=HF_ROUTER_BASE_URL,
                http_client=httpx.Client(timeout=60.0),
            )
            _provider = provider
            _initialization_error = None
            print(f"Chatbot: Hugging Face router initialized successfully with model {model_name}.")
            return _client
        except Exception as e:
            _provider = None
            _initialization_error = f"Failed to initialize Hugging Face router client: {str(e)}"
            print(f"Chatbot Error: {_initialization_error}")
            return None

    if not api_key.startswith(("sk-", "sk-proj-")):
        _provider = None
        _initialization_error = (
            "OPENAI_API_KEY does not look valid. Use an OpenAI key starting with sk- or set HUGGINGFACE_API_KEY."
        )
        print(f"Chatbot Warning: {_initialization_error}")
        return None

    try:
        _client = openai.OpenAI(
            api_key=api_key,
            http_client=httpx.Client(),
        )
        _provider = provider
        _initialization_error = None
        print("Chatbot: OpenAI client initialized successfully.")
        return _client
    except Exception as e:
        _provider = None
        _initialization_error = f"Failed to initialize OpenAI client: {str(e)}"
        print(f"Chatbot Error: {_initialization_error}")
        return None


def get_chatbot_client():
    global _client
    if _client is None:
        return init_chatbot()
    return _client


def get_initialization_error():
    return _initialization_error


async def get_chat_response(message: str):
    """
    Sends a message to the configured AI provider and returns the response.
    """
    client = get_chatbot_client()
    if not client:
        raise ValueError(f"Chatbot is not available: {get_initialization_error()}")

    try:
        if _provider == "huggingface":
            model_name = _get_env_value("HUGGINGFACE_MODEL") or DEFAULT_HF_MODEL
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": message},
                ],
                max_tokens=180,
                temperature=0.7,
            )
            return response.choices[0].message.content.strip()

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": message},
            ],
            max_tokens=150,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except openai.AuthenticationError as e:
        provider_name = "Hugging Face" if _provider == "huggingface" else "OpenAI"
        print(f"{provider_name} API Error: {e}")
        if _provider == "huggingface":
            raise ValueError(
                "Hugging Face rejected the API key. Make sure your token has Inference Providers permission."
            )
        raise ValueError("OpenAI rejected the API key. Replace OPENAI_API_KEY in .env with a valid key from OpenAI.")
    except openai.NotFoundError as e:
        print(f"Hugging Face API Error: {e}")
        raise ValueError("The configured Hugging Face model was not found on the router. Try another supported chat model.")
    except openai.BadRequestError as e:
        print(f"Hugging Face API Error: {e}")
        raise ValueError("The Hugging Face chat request was rejected. Check the selected model and token permissions.")
    except ValueError:
        raise
    except Exception as e:
        provider_name = _provider or "AI"
        print(f"{provider_name} API Error: {e}")
        raise ValueError("Failed to get response from AI assistant.")
