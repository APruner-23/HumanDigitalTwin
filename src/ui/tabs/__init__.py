"""
Tab modules for Streamlit app.
"""

from .triplet_extraction_tab import render_triplet_extraction_tab
from .ontology_validation_tab import render_ontology_validation_tab
from .iot_data_tab import render_iot_data_tab
from .external_services_tab import render_external_services_tab
from .chat_agent_tab import render_chat_agent_tab
from .knowledge_graph_tab import render_knowledge_graph_tab

__all__ = [
    'render_triplet_extraction_tab',
    'render_ontology_validation_tab',
    'render_iot_data_tab',
    'render_external_services_tab',
    'render_chat_agent_tab',
    'render_knowledge_graph_tab'
]
