"""
Multi-LLM provider abstraction layer.

Job postings often ask for "integrating multiple LLM APIs (OpenAI / Claude / Gemini)".
This file demonstrates how to turn "switching model providers" into changing a single
setting value, instead of hardcoding one provider's API throughout the codebase — this
is an architecture design point you can bring up directly in interviews.

Currently only Gemini is actually called (free tier); OpenAI / Claude have their
interfaces reserved. When you later want to hook up a real paid API, you just need
to fill in the corresponding class — no other code needs to change.
"""
from abc import ABC, abstractmethod

from app.core.config import settings


class BaseLLMProvider(ABC):
    """Every LLM provider must implement this interface, so upstream code doesn't
    need to care which model is running underneath."""

    @abstractmethod
    def get_chat_model(self):
        """Return a LangChain-compatible ChatModel object."""
        raise NotImplementedError

    @abstractmethod
    def get_embedding_model(self):
        """Return a LangChain-compatible Embeddings object."""
        raise NotImplementedError


class GeminiProvider(BaseLLMProvider):
    """Google AI Studio offers a daily free quota, suitable for learning and portfolio use."""

    def get_chat_model(self):
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=settings.CHAT_MODEL,
            google_api_key=settings.GOOGLE_API_KEY,
            temperature=0.2,
        )

    def get_embedding_model(self):
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            google_api_key=settings.GOOGLE_API_KEY,
        )


class OpenAIProvider(BaseLLMProvider):
    """Reserved interface: once you have a paid OpenAI quota, just fill in the
    implementation here — nothing upstream needs to change."""

    def get_chat_model(self):
        from langchain_openai import ChatOpenAI  # noqa: F401  (requires installing langchain-openai)

        return ChatOpenAI(api_key=settings.OPENAI_API_KEY, model="gpt-4o-mini")

    def get_embedding_model(self):
        from langchain_openai import OpenAIEmbeddings  # noqa: F401

        return OpenAIEmbeddings(api_key=settings.OPENAI_API_KEY)


class ClaudeProvider(BaseLLMProvider):
    """Reserved interface: for hooking up the Anthropic Claude API."""

    def get_chat_model(self):
        from langchain_anthropic import ChatAnthropic  # noqa: F401  (requires installing langchain-anthropic)

        return ChatAnthropic(api_key=settings.ANTHROPIC_API_KEY, model="claude-3-5-sonnet-latest")

    def get_embedding_model(self):
        raise NotImplementedError("Anthropic 目前不提供 Embedding API，建議搭配 Gemini 或本地端 embedding。")


_PROVIDERS = {
    "gemini": GeminiProvider,
    "openai": OpenAIProvider,
    "claude": ClaudeProvider,
}


def get_llm_provider(name: str | None = None) -> BaseLLMProvider:
    """Get the corresponding LLM provider instance, based on settings (or a given name)."""
    provider_name = (name or settings.LLM_PROVIDER).lower()
    if provider_name not in _PROVIDERS:
        raise ValueError(f"不支援的 LLM_PROVIDER：{provider_name}，可用選項：{list(_PROVIDERS.keys())}")
    return _PROVIDERS[provider_name]()
