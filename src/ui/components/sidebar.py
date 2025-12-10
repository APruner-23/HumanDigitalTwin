"""
Sidebar component with configuration info.
"""

import streamlit as st


def render_sidebar(config, llm) -> None:
    """
    Render sidebar with configuration information.

    Args:
        config: ConfigManager instance
        llm: LLM instance
    """
    with st.sidebar:
        st.header("Configurazione")

        # Model info
        model_info = llm.get_model_info()
        st.write(f"**Provider:** {model_info['provider']}")
        st.write(f"**Modello:** {model_info['model']}")

        # MCP configuration
        mcp_config = config.get_mcp_config()
        mcp_url = f"http://{mcp_config.get('host')}:{mcp_config.get('port')}"
        st.write(f"**MCP Server:** {mcp_url}")
