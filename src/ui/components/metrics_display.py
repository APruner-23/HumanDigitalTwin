"""
Components for displaying metrics and statistics.
"""

import streamlit as st
import plotly.express as px
from typing import List, Dict


def display_validation_metrics(results: List[Dict], threshold: float) -> None:
    """
    Display validation metrics and statistics.

    Args:
        results: List of validation results
        threshold: Confidence threshold
    """
    if not results:
        st.info("Nessun risultato disponibile")
        return

    # Filter by threshold
    valid_results = [r for r in results if r.get('mu', 0.0) >= threshold]
    low_conf_results = [r for r in results if r.get('mu', 0.0) < threshold]

    # Statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("✅ Valid", len(valid_results))
    with col2:
        st.metric("⚠️ Low Confidence", len(low_conf_results))
    with col3:
        avg_score = sum(r.get('mu', 0.0) for r in results) / len(results) if results else 0
        st.metric("📈 Score Medio", f"{avg_score:.3f}")

    # Score distribution histogram
    scores = [r.get('mu', 0.0) for r in results]
    fig = px.histogram(
        x=scores,
        nbins=20,
        title="Distribuzione Score di Confidenza",
        labels={'x': 'Score μ', 'y': 'Frequenza'}
    )
    fig.add_vline(x=threshold, line_dash="dash", line_color="red", annotation_text="Soglia")
    st.plotly_chart(fig, use_container_width=True)


def display_embedding_cache_metrics(ontology, embeddings) -> None:
    """
    Display embedding cache metrics.

    Args:
        ontology: SchemaOrgLoader instance
        embeddings: EmbeddingService instance
    """
    cache_size = len(embeddings.cache.cache) if embeddings.cache else 0
    total_items = len(ontology.get_all_classes()) + len(ontology.get_all_properties())

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.metric("Embeddings in cache", f"{cache_size}/{total_items}")
    with col2:
        completion_pct = (cache_size / total_items * 100) if total_items > 0 else 0
        st.metric("Completamento", f"{completion_pct:.1f}%")

    return cache_size, total_items
