"""
Streamlit Frontend for Human Digital Twin
Refactored version with modular architecture.
"""

import streamlit as st
import sys
from pathlib import Path

# Add project root to PYTHONPATH
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.ui.services import (
    init_config,
    init_prompt_manager,
    init_llm,
    init_triplet_graph
)
from src.ui.components import render_sidebar
from src.ui.tabs import (
    render_triplet_extraction_tab,
    render_ontology_validation_tab,
    render_iot_data_tab,
    render_external_services_tab,
    render_chat_agent_tab
)


def main():
    """Main function for the Streamlit app."""

    # Page configuration - MUST be first Streamlit call
    st.set_page_config(
        page_title="Human Digital Twin",
        layout="wide"
    )

    # Initialize configuration
    config = init_config()
    streamlit_config = config.get_streamlit_config()

    st.title(streamlit_config.get('title', 'Human Digital Twin'))

    # Initialize components
    try:
        prompt_manager = init_prompt_manager()
        llm = init_llm(config)
        triplet_graph = init_triplet_graph(config, llm)
    except ValueError as e:
        st.error(str(e))
        st.stop()
        return

    # Render sidebar
    render_sidebar(config, llm)

    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Estrazione Triplette",
        "Ontology Validation",
        "Dati IoT",
        "Servizi Esterni",
        "Chat Agent"
    ])

    # Render each tab
    with tab1:
        render_triplet_extraction_tab(config, triplet_graph)

    with tab2:
        render_ontology_validation_tab(config)

    with tab3:
        render_iot_data_tab(config, prompt_manager, llm)

    with tab4:
        render_external_services_tab(config)

    with tab5:
        render_chat_agent_tab(config, llm)


if __name__ == "__main__":
    main()
