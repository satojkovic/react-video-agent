from abc import ABC, abstractmethod


class BaseLLM(ABC):
    @abstractmethod
    def call_with_tools(self, messages: list, tools: list, **kwargs) -> dict:
        """Call the LLM with tool definitions and return the response message."""
        ...

    @abstractmethod
    def get_embeddings(self, text: str) -> list:
        """Generate embeddings for the given text."""
        ...
