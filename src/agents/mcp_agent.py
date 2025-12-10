"""
Agent that integrates LLM with MCP tools for autonomous interactions.
"""

from typing import Dict, Any, List, Optional
from src.llm.base_llm import BaseLLM
from src.mcp.mcp_tools import get_mcp_tools
from src.utils import get_logger
from src.prompts.prompt_manager import PromptManager


class MCPAgent:
    """
    Agent that allows the LLM to autonomously interact with the MCP server.
    Uses function calling to allow the LLM to decide when to retrieve data from the server.
    """

    def __init__(
        self,
        llm: BaseLLM,
        mcp_base_url: str = "http://localhost:8000",
        system_prompt: Optional[str] = None,
        enable_logging: bool = True
    ):
        """
        Initialize the MCP agent.

        Args:
            llm: LLM service instance
            mcp_base_url: Base URL of the MCP server
            system_prompt: Custom system prompt (optional)
            enable_logging: Enable Rich logging (default: True)
        """
        self.llm = llm
        self.mcp_base_url = mcp_base_url
        self.tools = get_mcp_tools(mcp_base_url)

        # Default system prompt
        self.system_prompt = system_prompt or self._get_default_system_prompt()

        # Conversation history
        self.conversation_history: List[Dict[str, str]] = []

        # Logger
        self.enable_logging = enable_logging
        self.logger = get_logger() if enable_logging else None

    def _get_default_system_prompt(self) -> str:
        """Returns the default system prompt for the agent."""
        prompt_manager = PromptManager()
        return prompt_manager.get_system_prompt('mcp_agent')

    def reset_conversation(self) -> None:
        """Resets the conversation history."""
        self.conversation_history = []

    def chat(self, user_message: str, include_history: bool = True) -> Dict[str, Any]:
        """
        Send a message to the agent and receive a response.

        Args:
            user_message: User's message
            include_history: If True, includes conversation history

        Returns:
            Dictionary with response and metadata:
            {
                "response": str,
                "tools_used": List[str],
                "conversation_history": List[Dict]
            }
        """
        # Costruisci i messaggi
        messages = []

        # Add system prompt
        messages.append({
            "role": "system",
            "content": self.system_prompt
        })

        # Add history if requested
        if include_history and self.conversation_history:
            messages.extend(self.conversation_history)

        # Add current message
        messages.append({
            "role": "user",
            "content": user_message
        })

        # Log user message
        if self.logger:
            self.logger.log_user_message(user_message)

        # Generate response using function calling
        try:
            # Log LLM call
            if self.logger:
                model_info = self.llm.get_model_info()

            response = self.llm.generate_with_tools(messages, self.tools)

            # Log LLM response
            if self.logger:
                self.logger.log_llm_call(messages, response, model_info)

            # Update history
            self.conversation_history.append({
                "role": "user",
                "content": user_message
            })
            self.conversation_history.append({
                "role": "assistant",
                "content": response
            })

            # Log agent response
            if self.logger:
                self.logger.log_agent_response(response)

            return {
                "response": response,
                "tools_used": [],  # TODO: track used tools
                "conversation_history": self.conversation_history.copy()
            }

        except NotImplementedError as e:
            # LLM provider does not support function calling
            if self.logger:
                self.logger.log_error(str(e), "Function calling not supported")

            return {
                "response": f"Errore: {str(e)}",
                "tools_used": [],
                "conversation_history": self.conversation_history.copy(),
                "error": str(e)
            }
        except Exception as e:
            if self.logger:
                self.logger.log_error(str(e), "Agent chat error")

            return {
                "response": f"Errore durante la generazione: {str(e)}",
                "tools_used": [],
                "conversation_history": self.conversation_history.copy(),
                "error": str(e)
            }

    def chat_stream(self, user_message: str) -> str:
        """
        Simplified version of chat that returns only the response.

        Args:
            user_message: User's message

        Returns:
            Agent's response
        """
        result = self.chat(user_message)
        return result.get("response", "Errore nella generazione della risposta")

    def get_available_tools(self) -> List[Dict[str, str]]:
        """
        Returns information about available tools.

        Returns:
            List of dictionaries with tool name and description
        """
        tools_info = []
        for tool in self.tools:
            tools_info.append({
                "name": tool.name,
                "description": tool.description
            })
        return tools_info

    def set_system_prompt(self, prompt: str) -> None:
        """
        Sets a new system prompt.

        Args:
            prompt: New system prompt
        """
        self.system_prompt = prompt
