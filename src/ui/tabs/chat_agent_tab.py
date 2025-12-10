"""
Tab 5: Chat Agent with MCP Access.
"""

import streamlit as st
from src.agents import MCPAgent
from ..services.app_state import AppState


def render_chat_agent_tab(config, llm) -> None:
    """
    Render the chat agent tab.

    Args:
        config: ConfigManager instance
        llm: LLM instance
    """
    st.header("Chat Agent con Accesso MCP")
    st.write("Chatta con l'AI che può accedere autonomamente ai dati IoT tramite il server MCP")

    # Initialize agent
    if not AppState.get_mcp_agent():
        mcp_config = config.get_mcp_config()
        mcp_url = f"http://{mcp_config.get('host')}:{mcp_config.get('port')}"
        AppState.set_mcp_agent(MCPAgent(llm, mcp_base_url=mcp_url))

    agent = AppState.get_mcp_agent()

    # Display available tools
    with st.expander("Tools Disponibili per l'Agent"):
        tools_info = agent.get_available_tools()
        for tool in tools_info:
            st.write(f"**{tool['name']}**: {tool['description']}")

    # Reset conversation button
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("Reset Conversazione"):
            agent.reset_conversation()
            AppState.set_chat_history([])
            st.success("Conversazione resettata!")
            st.rerun()

    # Display chat history
    chat_history = AppState.get_chat_history()
    for message in chat_history:
        role = message["role"]
        content = message["content"]

        if role == "user":
            st.chat_message("user").write(content)
        else:
            st.chat_message("assistant").write(content)

    # User input
    user_input = st.chat_input("Scrivi un messaggio all'agent...")

    if user_input:
        # Add user message to history
        AppState.append_chat_message("user", user_input)

        # Display user message
        with st.chat_message("user"):
            st.write(user_input)

        # Generate response from agent
        with st.chat_message("assistant"):
            with st.spinner("L'agent sta pensando..."):
                try:
                    result = agent.chat(user_input)

                    response = result.get("response", "Errore nella generazione")
                    error = result.get("error")

                    if error:
                        st.error(f"Errore: {error}")
                    else:
                        st.write(response)

                        # Show tools used
                        tools_used = result.get("tools_used", [])
                        if tools_used:
                            with st.expander("Tools Utilizzati"):
                                for tool in tools_used:
                                    st.write(f"- {tool}")

                    # Add response to history
                    AppState.append_chat_message("assistant", response)

                except Exception as e:
                    error_msg = f"Errore durante la generazione: {str(e)}"
                    st.error(error_msg)
                    AppState.append_chat_message("assistant", error_msg)

    # Example questions
    if not chat_history:
        st.markdown("### Esempi di domande:")
        st.markdown("""
        - "Quali dispositivi IoT sono disponibili?"
        - "Mostrami gli ultimi dati del mio smartwatch"
        - "Calcola le statistiche della mia frequenza cardiaca"
        - "Come sta la mia salute oggi?"
        """)
