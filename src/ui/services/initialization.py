"""
Initialization functions for app components.
All functions are cached with @st.cache_resource.
"""

import streamlit as st
from src.config import ConfigManager
from src.llm import LLMFactory
from src.prompts import PromptManager


@st.cache_resource
def init_config():
    """Initialize configuration."""
    return ConfigManager()


@st.cache_resource
def init_prompt_manager():
    """Initialize PromptManager."""
    return PromptManager()


@st.cache_resource
def init_llm(_config):
    """Initialize LLM service."""
    llm_config = _config.get_llm_config()
    provider = llm_config.get('provider', 'groq')
    api_key = _config.get_env('GROQ_API_KEY')

    if not api_key:
        raise ValueError("GROQ_API_KEY non trovata nel file .env")

    return LLMFactory.create(provider, llm_config, api_key)


@st.cache_resource
def init_triplet_graph(_config, _llm, _graph_version="v5_cascade_text_iot_aug"):
    """Initialize LangGraph for triplet extraction."""
    from src.agents.triplet_extraction_graph import TripletExtractionGraph

    api_key = _config.get_env('GROQ_API_KEY')
    mcp_config = _config.get_mcp_config()
    mcp_url = f"http://{mcp_config.get('host')}:{mcp_config.get('port')}"

    return TripletExtractionGraph(
        llm_api_key=api_key,
        llm_model=_llm.model,
        mcp_base_url=mcp_url,
        enable_logging=True,
        delta_mode=0
    )


@st.cache_resource
def init_ontology_services(_config):
    """Initialize services for ontology validation."""
    from src.ontology.schema_downloader import ensure_schema_org
    from src.ontology import SchemaOrgLoader, EmbeddingService

    # Download schema.jsonld se necessario
    with st.spinner("📥 Verifico disponibilità Schema.org ontology..."):
        schema_path = ensure_schema_org(_config)
        if not schema_path:
            st.error("❌ Impossibile scaricare Schema.org ontology")
            return None, None

    # Carica ontologia
    with st.spinner("📚 Caricamento Schema.org ontology..."):
        ontology = SchemaOrgLoader(schema_path)
        st.success(
            f"✅ Ontologia caricata: {len(ontology.get_all_classes())} classi, "
            f"{len(ontology.get_all_properties())} proprietà"
        )

    # Inizializza embedding service
    ontology_config = _config.get_ontology_config()
    st.write("DEBUG ontology_config:", ontology_config)
    provider = ontology_config.get('embedding_provider', 'minilm')

    # NEW: default ora è 'minilm', non più 'cohere'
    provider = ontology_config.get('embedding_provider', 'minilm')
    cache_dir = ontology_config.get('cache_dir', 'data/ontology/cache')

    # DEBUG da togliere
    st.write(f"DEBUG embedding provider from config: {provider!r}")
    print("DEBUG embedding provider from config:", provider)

    api_key = None
    if provider == 'cohere':
        api_key = _config.get_env('COHERE_API_KEY')
    elif provider == 'mistral':
        api_key = _config.get_env('MISTRAL_API_KEY')

    # Solo i provider remoti richiedono API key
    if provider in ['cohere', 'mistral'] and not api_key:
        st.error(f"❌ {provider.upper()}_API_KEY non trovata nel file .env")
        return None, None

    # Per provider locali (minilm, gemma, qwen) api_key viene ignorata
    embeddings = EmbeddingService(
        provider=provider,
        api_key=api_key,
        use_cache=True,
        cache_dir=cache_dir
    )

    return ontology, embeddings
