from typing import Dict, Any, List
from .groq_failover import GroqFailover
from .base_llm import BaseLLM
from ..utils import get_logger


class GroqLLM(BaseLLM):
    """Implementation of LLM service using Groq with automatic API key failover."""

    def __init__(self, config: Dict[str, Any], api_key: str):
        """
        Initialize Groq service.

        Args:
            config: Dizionario con la configurazione dell'LLM
            api_key: API key per Groq (kept for compatibility, failover loads all env keys)
        """
        super().__init__(config)
        self.api_key = api_key

        # Initialize Groq client with automatic key rotation
        self.client = GroqFailover(
            model_name=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )

    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate a response from the prompt using Groq.

        Args:
            prompt: The prompt to send to the LLM
            **kwargs: Additional parameters (temperature, max_tokens, etc.)

        Returns:
            The generated response from the LLM
        """
        try:
            response = self.client.invoke(prompt)
            return response.content
        except Exception as e:
            raise Exception(f"Error generating with Groq: {str(e)}")

    def generate_with_history(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        """
        Generate a response considering message history.

        Args:
            messages: List of messages in format [{"role": "user/assistant/system", "content": "..."}]
            **kwargs: Additional parameters

        Returns:
            The generated response from the LLM
        """
        try:
            from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

            # Convert messages to Langchain format
            lc_messages = []
            for msg in messages:
                role = msg.get('role', 'user')
                content = msg.get('content', '')

                if role == 'system':
                    lc_messages.append(SystemMessage(content=content))
                elif role == 'assistant':
                    lc_messages.append(AIMessage(content=content))
                else:
                    lc_messages.append(HumanMessage(content=content))

            response = self.client.invoke(lc_messages)
            return response.content

        except Exception as e:
            raise Exception(f"Error in generation with history (Groq): {str(e)}")

    def is_available(self) -> bool:
        """
        Check if Groq service is available.

        Returns:
            True if available, False otherwise
        """
        try:
            self.client.invoke("test")
            return True
        except Exception:
            return False

    def generate_with_tools(
        self,
        messages: List[Dict[str, str]],
        tools: List[Any],
        **kwargs
    ) -> str:
        """
        Generate a response using function calling with external tools.

        The LLM can autonomously decide if and when to call tools.

        Args:
            messages: List of messages in format [{"role": "user/assistant/system", "content": "..."}]
            tools: List of available Langchain tools
            **kwargs: Additional parameters

        Returns:
            The generated response from the LLM after calling tools if necessary
        """
        try:
            from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage

            # Convert messages to Langchain format
            lc_messages = []
            for msg in messages:
                role = msg.get('role', 'user')
                content = msg.get('content', '')

                if role == 'system':
                    lc_messages.append(SystemMessage(content=content))
                elif role == 'assistant':
                    lc_messages.append(AIMessage(content=content))
                else:
                    lc_messages.append(HumanMessage(content=content))

            # Bind tools to model through failover wrapper
            llm_with_tools = self.client.bind_tools(tools)

            # First model call
            response = llm_with_tools.invoke(lc_messages)

            # If there are tool calls, execute them and recall model
            if hasattr(response, 'tool_calls') and response.tool_calls:
                lc_messages.append(response)

                for tool_call in response.tool_calls:
                    tool_name = tool_call['name']
                    tool_args = tool_call['args']

                    selected_tool = None
                    for tool in tools:
                        if tool.name == tool_name:
                            selected_tool = tool
                            break

                    if selected_tool:
                        tool_result = selected_tool.invoke(tool_args)

                        try:
                            logger = get_logger()
                            logger.log_tool_call(tool_name, tool_args, str(tool_result))
                        except Exception:
                            pass

                        lc_messages.append(
                            ToolMessage(
                                content=str(tool_result),
                                tool_call_id=tool_call['id']
                            )
                        )

                # Second model call after tool results
                final_response = llm_with_tools.invoke(lc_messages)
                return final_response.content

            return response.content

        except Exception as e:
            raise Exception(f"Error in generation with tools (Groq): {str(e)}")

    def get_model_info(self) -> Dict[str, Any]:
        """Returns information about current Groq model."""
        info = super().get_model_info()
        info['provider'] = 'Groq'
        info['supports_tools'] = True
        return info