from typing import Dict, Any
from .base_llm import BaseLLM
from .groq_llm import GroqLLM


class LLMFactory:
    """Factory for creating LLM service instances."""

    # Registry of available providers
    _providers = {
        'groq': GroqLLM,
        # Other providers can be added here in the future
        # 'openai': OpenAILLM,
        # 'anthropic': AnthropicLLM,
    }

    @classmethod
    def create(cls, provider: str, config: Dict[str, Any], api_key: str) -> BaseLLM:
        """
        Creates an instance of the specified LLM service.

        Args:
            provider: Provider name (e.g., 'groq', 'openai')
            config: LLM configuration
            api_key: API key for the provider

        Returns:
            Instance of the LLM service

        Raises:
            ValueError: If the provider is not supporteds
        """
        provider_lower = provider.lower()

        if provider_lower not in cls._providers:
            available = ', '.join(cls._providers.keys())
            raise ValueError(
                f"Provider '{provider}' not supported. "
                f"Available providers: {available}"
            )

        llm_class = cls._providers[provider_lower]
        return llm_class(config, api_key)

    @classmethod
    def register_provider(cls, name: str, provider_class: type) -> None:
        """
        Registers a new LLM provider.

        Args:
            name: Provider name
            provider_class: Class implementing BaseLLM
        """
        if not issubclass(provider_class, BaseLLM):
            raise TypeError(f"{provider_class} must inherit from BaseLLM")

        cls._providers[name.lower()] = provider_class

    @classmethod
    def get_available_providers(cls) -> list:
        """
        Returns the list of available providers.

        Returns:
            List of available provider names
        """
        return list(cls._providers.keys())
