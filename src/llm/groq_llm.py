from typing import Dict, Any, List
from langchain_groq import ChatGroq
from .base_llm import BaseLLM
from ..utils import get_logger


class GroqLLM(BaseLLM):
    """Implementation of LLM service using Groq."""

    def __init__(self, config: Dict[str, Any], api_key: str):
        """
        Initialize Groq service.

        Args:
            config: Dizionario con la configurazione dell'LLM
            api_key: API key per Groq
        """
        super().__init__(config)
        self.api_key = api_key

        # Initialize Groq client via Langchain
        self.client = ChatGroq(
            groq_api_key=self.api_key,
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
            # Update parameters if provided
            temperature = kwargs.get('temperature', self.temperature)
            max_tokens = kwargs.get('max_tokens', self.max_tokens)

            # Update client with new parameters if necessary
            if temperature != self.temperature or max_tokens != self.max_tokens:
                self.client.temperature = temperature
                self.client.max_tokens = max_tokens

            # Invoke model
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
            # Update parameters if provided
            temperature = kwargs.get('temperature', self.temperature)
            max_tokens = kwargs.get('max_tokens', self.max_tokens)

            if temperature != self.temperature or max_tokens != self.max_tokens:
                self.client.temperature = temperature
                self.client.max_tokens = max_tokens

            # Convert messages to Langchain format
            from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

            lc_messages = []
            for msg in messages:
                role = msg.get('role', 'user')
                content = msg.get('content', '')

                if role == 'system':
                    lc_messages.append(SystemMessage(content=content))
                elif role == 'assistant':
                    lc_messages.append(AIMessage(content=content))
                else:  # user
                    lc_messages.append(HumanMessage(content=content))

            # Invoke model
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
            # Simple test to check availability
            test_response = self.client.invoke("test")
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
            messages: List of messages in format [{"role": "user/assistant", "content": "..."}]
            tools: List of available Langchain tools
            **kwargs: Additional parameters

        Returns:
            The generated response from the LLM after calling tools if necessary
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
                else:  # user
                    lc_messages.append(HumanMessage(content=content))

            # Bind tools to model
            llm_with_tools = self.client.bind_tools(tools)

            # Invoke model with tools
            response = llm_with_tools.invoke(lc_messages)

            # If there are tool calls, execute them and recall model
            if hasattr(response, 'tool_calls') and response.tool_calls:
                # Add model response to messages
                lc_messages.append(response)

                # Execute each tool call
                from langchain_core.messages import ToolMessage

                for tool_call in response.tool_calls:
                    # Find corresponding tool
                    tool_name = tool_call['name']
                    tool_args = tool_call['args']

                    # Search for tool in list
                    selected_tool = None
                    for tool in tools:
                        if tool.name == tool_name:
                            selected_tool = tool
                            break

                    if selected_tool:
                        # Execute tool
                        tool_result = selected_tool.invoke(tool_args)

                        # Log tool call
                        try:
                            logger = get_logger()
                            logger.log_tool_call(tool_name, tool_args, str(tool_result))
                        except:
                            pass  # Ignore logging errors TODO fix

                        # Add result to messages
                        lc_messages.append(
                            ToolMessage(
                                content=str(tool_result),
                                tool_call_id=tool_call['id']
                            )
                        )

                # Recall model with tool results
                final_response = llm_with_tools.invoke(lc_messages)
                return final_response.content
            else:
                # No tool called, return direct response
                return response.content

        except Exception as e:
            raise Exception(f"Error in generation with tools (Groq): {str(e)}")

    def get_model_info(self) -> Dict[str, Any]:
        """Returns information about current Groq model."""
        info = super().get_model_info()
        info['provider'] = 'Groq'
        info['supports_tools'] = True
        return info
