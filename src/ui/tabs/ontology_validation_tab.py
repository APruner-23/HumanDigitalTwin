"""
Tab 2: Ontology Validation with Schema.org.
"""

import streamlit as st
import json
from ..services.app_state import AppState
from ..components.session_loader import render_session_loader
from ..components.metrics_display import display_validation_metrics, display_embedding_cache_metrics
from ..components.triplet_display import display_validation_result


def _pre_compute_embeddings(ontology, embeddings, rate_limit: float) -> None:
    """
    Pre-compute all embeddings for ontology classes and properties.

    Args:
        ontology: SchemaOrgLoader instance
        embeddings: EmbeddingService instance
        rate_limit: Rate limit delay in seconds
    """
    progress_bar = st.progress(0.0)
    status_text = st.empty()

    # Collect all texts to embed (same logic as TripleMatcher)
    all_texts = []

    # Classes with enriched context
    for class_name in ontology.get_all_classes():
        # 1. Simple name (used for direct matching)
        all_texts.append(class_name)

        # 2. Enriched context
        class_desc = ontology.get_class_description(class_name)
        class_info = ontology.get_class_info(class_name)

        parts = [f"{class_name}"]
        if class_desc:
            parts.append(class_desc)

        parent_classes = class_info.get('subClassOf', [])
        if parent_classes:
            parents_str = ", ".join(parent_classes[:3])
            parts.append(f"Type of: {parents_str}")

        text = ". ".join(parts)
        all_texts.append(text)

    # Properties with enriched context
    for prop_name in ontology.get_all_properties():
        # 1. Simple name (used for direct matching)
        all_texts.append(prop_name)

        # 2. Enriched context
        prop_desc = ontology.get_property_description(prop_name)
        prop_info = ontology.get_property_info(prop_name)

        parts = [f"{prop_name}"]
        if prop_desc:
            parts.append(prop_desc)

        domain_classes = prop_info.get('domainIncludes', [])
        if domain_classes:
            domain_str = ", ".join(domain_classes[:3])
            parts.append(f"Used with: {domain_str}")

        range_classes = prop_info.get('rangeIncludes', [])
        if range_classes:
            range_str = ", ".join(range_classes[:3])
            parts.append(f"Points to: {range_str}")

        text = ". ".join(parts)
        all_texts.append(text)

    # Process in batches
    batch_size = 20
    for i in range(0, len(all_texts), batch_size):
        batch = all_texts[i:i + batch_size]

        try:
            embeddings.embed_texts(batch, input_type="search_document", rate_limit_delay=rate_limit)
            embeddings.cache.save_cache()

            progress = min(1.0, (i + batch_size) / len(all_texts))
            progress_bar.progress(progress)
            status_text.text(f"Processati {min(i + batch_size, len(all_texts))}/{len(all_texts)} embeddings...")

        except Exception as e:
            st.error(f"❌ Errore durante pre-calcolo: {str(e)}")
            break

    progress_bar.empty()
    status_text.empty()
    st.success("✅ Pre-calcolo completato!")


def _validate_triplets(triplets, ontology, embeddings, rate_limit: float) -> list:
    """
    Validate triplets against Schema.org ontology.

    Args:
        triplets: List of triplets to validate
        ontology: SchemaOrgLoader instance
        embeddings: EmbeddingService instance
        rate_limit: Rate limit delay in seconds

    Returns:
        List of validation results
    """
    from src.ontology import TripleMatcher

    matcher = TripleMatcher(ontology, embeddings, rate_limit_delay=rate_limit)

    validated_results = []

    progress_bar = st.progress(0.0)
    status_text = st.empty()

    for idx, triplet in enumerate(triplets):
        status_text.text(f"Validazione tripletta {idx + 1}/{len(triplets)}...")

        # Extract values and types from 2x3 matrix
        subject_data = triplet.get('subject', {})
        predicate_data = triplet.get('predicate', {})
        obj_data = triplet.get('object', {})

        subject_value = subject_data.get('value', '') if isinstance(subject_data, dict) else subject_data
        predicate_value = predicate_data.get('value', '') if isinstance(predicate_data, dict) else predicate_data
        obj_value = obj_data.get('value', '') if isinstance(obj_data, dict) else obj_data

        subject_type = subject_data.get('type', None) if isinstance(subject_data, dict) else None
        predicate_type = predicate_data.get('type', None) if isinstance(predicate_data, dict) else None
        obj_type = obj_data.get('type', None) if isinstance(obj_data, dict) else None

        if subject_value and predicate_value and obj_value:
            result = matcher.match_triple(
                subject_value, predicate_value, obj_value,
                subject_type, predicate_type, obj_type
            )
            validated_results.append(result)

        progress_bar.progress((idx + 1) / len(triplets))

    status_text.empty()
    progress_bar.empty()

    return validated_results


def render_ontology_validation_tab(config) -> None:
    """
    Render the ontology validation tab.

    Args:
        config: ConfigManager instance
    """
    st.header("🔍 Ontology Validation con Schema.org")
    st.write("Valida le triplette estratte confrontandole semanticamente con l'ontologia Schema.org")

    # Session loader
    source_info = render_session_loader(config)

    # Check if triplets exist
    triplets_to_validate = AppState.get_extracted_triplets()

    if not triplets_to_validate:
        return

    st.success(f"✅ Trovate **{len(triplets_to_validate)}** triplette da validare{source_info or ''}")

    # Sidebar for validation config
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 🛠️ Validazione Config")

        ontology_config = config.get_ontology_config()
        validation_threshold = st.slider(
            "Soglia confidence minima:",
            min_value=0.0,
            max_value=1.0,
            value=ontology_config.get('validation_threshold', 0.5),
            step=0.05,
            help="Triplette con score sotto questa soglia saranno marcate come low-confidence"
        )

        rate_limit = st.slider(
            "Rate limit (sec):",
            min_value=0.0,
            max_value=10.0,
            value=ontology_config.get('rate_limit_delay', 2.0),
            step=0.5,
            help="Pausa tra richieste API (2.0+ raccomandato per Cohere trial, evita rate limit 429)"
        )

    # Initialize ontology services
    from ..services.initialization import init_ontology_services
    ontology, embeddings = init_ontology_services(config)

    if not ontology or not embeddings:
        return

    # Pre-compute embeddings button
    st.markdown("### 🛠️ Gestione Cache Embeddings")
    if st.button("🚀 Pre-calcola/Aggiorna Embeddings", help="Calcola o aggiorna gli embeddings mancanti per classi e proprietà"):
        with st.spinner("⏳ Pre-calcolo embeddings in corso..."):
            _pre_compute_embeddings(ontology, embeddings, rate_limit)
            st.rerun()

    # Validate triplets button
    st.markdown("---")
    if st.button("🔍 Valida Triplette con Schema.org", type="primary"):
        validated_results = _validate_triplets(triplets_to_validate, ontology, embeddings, rate_limit)

        # Save results to session state
        AppState.set_validation_results(validated_results)
        AppState.set_validation_threshold(validation_threshold)

        st.success(f"✅ Validazione completata per {len(validated_results)} triplette!")
        st.rerun()

        # Display validation results
    validation_results = AppState.get_validation_results()
    if validation_results:
        st.markdown("---")
        st.subheader("📊 Risultati Validazione")

        threshold = AppState.get_validation_threshold()

        # Display metrics (usano tutte le triplette)
        display_validation_metrics(validation_results, threshold)

        # --- Download JSON (deve stare prima del loop altrimenti potrebbe non essere renderizzato) ---
        json_output = json.dumps(validation_results, indent=2, ensure_ascii=False)
        st.download_button(
            label="⬇️ Download Risultati Validazione (JSON)",
            data=json_output,
            file_name="validation_results.json",
            mime="application/json"
        )

        # Filter by threshold
        valid_results = [r for r in validation_results if r.get('mu', 0.0) >= threshold]
        low_conf_results = [r for r in validation_results if r.get('mu', 0.0) < threshold]

        # Limite massimo di triplette da visualizzare per tab
        max_display = 200

        st.caption(
            f"Mostro al massimo {max_display} triplette per tab per non appesantire l'interfaccia. "
            f"Il download JSON contiene comunque tutte le {len(validation_results)} triplette validate."
        )

        # Tabs for valid and low-confidence results
        result_tab1, result_tab2 = st.tabs(["✅ Validated Triplets", "⚠️ Low Confidence Triplets"])

        with result_tab1:
            if valid_results:
                subset = valid_results[:max_display]
                st.write(f"Mostro {len(subset)} / {len(valid_results)} triplette sopra soglia {threshold}.")
                for idx, result in enumerate(subset, 1):
                    display_validation_result(result, idx, show_branches=True)

                if len(valid_results) > max_display:
                    st.info(f"...e altre {len(valid_results) - max_display} non mostrate qui ma presenti nel JSON.")
            else:
                st.info("Nessuna tripletta sopra la soglia")

        with result_tab2:
            if low_conf_results:
                subset = low_conf_results[:max_display]
                st.warning(f"⚠️ Mostro {len(subset)} / {len(low_conf_results)} triplette sotto la soglia {threshold}.")
                for idx, result in enumerate(subset, 1):
                    display_validation_result(result, idx, show_branches=True)

                if len(low_conf_results) > max_display:
                    st.info(f"...e altre {len(low_conf_results) - max_display} non mostrate qui ma presenti nel JSON.")
            else:
                st.success("✅ Tutte le triplette sopra la soglia!")
