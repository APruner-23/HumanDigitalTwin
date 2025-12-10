"""
Centralized session state management.
"""

import streamlit as st
from typing import Any, Optional, List, Dict


class AppState:
    """Wrapper for st.session_state with typed methods."""

    @staticmethod
    def get(key: str, default: Any = None) -> Any:
        """Get value from session state."""
        return st.session_state.get(key, default)

    @staticmethod
    def set(key: str, value: Any) -> None:
        """Set value in session state."""
        st.session_state[key] = value

    @staticmethod
    def has(key: str) -> bool:
        """Check if key exists in session state."""
        return key in st.session_state

    @staticmethod
    def delete(key: str) -> None:
        """Delete key from session state."""
        if key in st.session_state:
            del st.session_state[key]

    # Specific getters/setters for common state
    @staticmethod
    def get_extracted_triplets() -> Optional[List[Dict]]:
        """Get extracted triplets from session state."""
        return st.session_state.get('extracted_triplets')

    @staticmethod
    def set_extracted_triplets(triplets: List[Dict]) -> None:
        """Set extracted triplets in session state."""
        st.session_state['extracted_triplets'] = triplets

    @staticmethod
    def get_validation_results() -> Optional[List[Dict]]:
        """Get validation results from session state."""
        return st.session_state.get('validation_results')

    @staticmethod
    def set_validation_results(results: List[Dict]) -> None:
        """Set validation results in session state."""
        st.session_state['validation_results'] = results

    @staticmethod
    def get_validation_threshold() -> float:
        """Get validation threshold from session state."""
        return st.session_state.get('validation_threshold', 0.5)

    @staticmethod
    def set_validation_threshold(threshold: float) -> None:
        """Set validation threshold in session state."""
        st.session_state['validation_threshold'] = threshold

    @staticmethod
    def get_loaded_from_file() -> Optional[str]:
        """Get loaded file name from session state."""
        return st.session_state.get('loaded_from_file')

    @staticmethod
    def set_loaded_from_file(filename: str) -> None:
        """Set loaded file name in session state."""
        st.session_state['loaded_from_file'] = filename

    @staticmethod
    def get_last_saved_session() -> Optional[str]:
        """Get last saved session path from session state."""
        return st.session_state.get('last_saved_session')

    @staticmethod
    def set_last_saved_session(path: str) -> None:
        """Set last saved session path in session state."""
        st.session_state['last_saved_session'] = path

    @staticmethod
    def get_extraction_result() -> Optional[Dict]:
        """Get full extraction result from session state."""
        return st.session_state.get('extraction_result')

    @staticmethod
    def set_extraction_result(result: Dict) -> None:
        """Set full extraction result in session state."""
        st.session_state['extraction_result'] = result

    @staticmethod
    def get_chat_history() -> List[Dict]:
        """Get chat history from session state."""
        return st.session_state.get('chat_history', [])

    @staticmethod
    def set_chat_history(history: List[Dict]) -> None:
        """Set chat history in session state."""
        st.session_state['chat_history'] = history

    @staticmethod
    def append_chat_message(role: str, content: str) -> None:
        """Append message to chat history."""
        if 'chat_history' not in st.session_state:
            st.session_state['chat_history'] = []
        st.session_state['chat_history'].append({
            "role": role,
            "content": content
        })

    @staticmethod
    def get_mcp_agent():
        """Get MCP agent from session state."""
        return st.session_state.get('mcp_agent')

    @staticmethod
    def set_mcp_agent(agent) -> None:
        """Set MCP agent in session state."""
        st.session_state['mcp_agent'] = agent

    @staticmethod
    def get_data_generator():
        """Get data generator from session state."""
        return st.session_state.get('data_generator')

    @staticmethod
    def set_data_generator(generator) -> None:
        """Set data generator in session state."""
        st.session_state['data_generator'] = generator
