"""
Knowledge Graph Builder Tab
"""

import streamlit as st
import json
from typing import Dict, Any


def render_knowledge_graph_tab(config: Any) -> None:
    """
    Render the Knowledge Graph Builder tab.

    Args:
        config: Configuration manager instance
    """
    st.header("🕸️ Knowledge Graph Builder")
    st.write("Costruisci il Knowledge Graph classificando le triplette in broad/narrow topics")

    # Inizializza il KG builder (cached)
    @st.cache_resource
    def init_kg_builder(_config):
        """Inizializza il Knowledge Graph Builder."""
        from src.agents import KnowledgeGraphBuilder, InMemoryKnowledgeGraph, Neo4jKnowledgeGraph
        from src.prompts import PromptManager
        from src.llm import LLMFactory

        api_key = _config.get_env('GROQ_API_KEY')
        prompt_mgr = PromptManager()

        # Inizializza LLM
        llm_config = _config.get_llm_config()
        llm = LLMFactory.create(llm_config.get('provider', 'groq'), llm_config, api_key)

        # Configurazione storage
        kg_config = _config.get('knowledge_graph', {})
        storage_type = kg_config.get('storage_type', 'in_memory')

        if storage_type == 'neo4j':
            # Usa Neo4j
            neo4j_config = kg_config.get('neo4j', {})
            neo4j_uri = _config.get_env('NEO4J_URI', neo4j_config.get('uri', 'bolt://localhost:7687'))
            neo4j_username = _config.get_env('NEO4J_USERNAME', 'neo4j')
            neo4j_password = _config.get_env('NEO4J_PASSWORD')
            neo4j_database = _config.get_env('NEO4J_DATABASE', neo4j_config.get('database', 'neo4j'))

            # Recupera person_id e person_name da session_state se disponibili
            person_id = st.session_state.get('selected_person_id')
            person_name = st.session_state.get('selected_person_name')

            # Fallback a default se non specificati
            if not person_id:
                person_id = 'main_person'
            if not person_name:
                person_name = 'User'

            if not neo4j_password:
                st.warning("⚠️ NEO4J_PASSWORD non trovata in .env. Uso storage in-memory come fallback.")
                storage = InMemoryKnowledgeGraph()
            else:
                try:
                    storage = Neo4jKnowledgeGraph(
                        uri=neo4j_uri,
                        username=neo4j_username,
                        password=neo4j_password,
                        database=neo4j_database,
                        person_id=person_id,
                        person_name=person_name
                    )
                    st.success(f"✅ Connesso a Neo4j: {neo4j_uri} (database: {neo4j_database}, profilo: {person_name})")
                except Exception as e:
                    st.error(f"❌ Errore connessione Neo4j: {str(e)}. Uso storage in-memory come fallback.")
                    storage = InMemoryKnowledgeGraph()
        else:
            # Usa in-memory storage
            storage = InMemoryKnowledgeGraph()
            st.info("💾 Uso storage in-memory (temporaneo)")

        return KnowledgeGraphBuilder(
            llm_api_key=api_key,
            llm_model=llm.model,
            prompt_manager=prompt_mgr,
            storage=storage,
            enable_logging=True
        )

    try:
        kg_builder = init_kg_builder(config)
    except Exception as e:
        st.error(f"❌ Errore inizializzazione Knowledge Graph Builder: {str(e)}")
        return

    # Gestione Profili Person (solo per Neo4j)
    storage = kg_builder.get_storage()
    kg_config = config.get('knowledge_graph', {})
    storage_type = kg_config.get('storage_type', 'in_memory')

    if storage_type == 'neo4j' and hasattr(storage, 'get_all_persons'):
        st.subheader("👤 Gestione Profili Person")

        # Recupera tutti i profili esistenti
        all_persons = storage.get_all_persons()

        # Mostra profilo corrente
        current_person_info = f"**Profilo corrente:** {storage.person_name} (ID: `{storage.person_id}`)"
        st.info(current_person_info)

        # Expander per gestire profili
        with st.expander("🔧 Gestione Profili", expanded=False):
            # Tab per creare nuovo profilo o cancellare
            profile_tab1, profile_tab2 = st.tabs(["➕ Crea Nuovo Profilo", "🗑️ Elimina Profilo"])

            with profile_tab1:
                st.markdown("Crea un nuovo profilo Person per il Knowledge Graph")

                new_person_name = st.text_input("Nome del profilo:", placeholder="Mario Rossi", key="new_person_name")
                new_person_id = st.text_input(
                    "ID univoco (opzionale):",
                    placeholder="Lascia vuoto per generazione automatica",
                    key="new_person_id",
                    help="Se vuoto, verrà generato automaticamente dal nome"
                )

                if st.button("✨ Crea Profilo", type="primary", key="create_profile_btn"):
                    if new_person_name:
                        # Genera ID se non fornito
                        if not new_person_id:
                            import re
                            new_person_id = re.sub(r'[^a-z0-9_]', '_', new_person_name.lower().strip())

                        try:
                            # Crea nuovo storage con il nuovo person_id
                            from src.agents import Neo4jKnowledgeGraph

                            neo4j_config = kg_config.get('neo4j', {})
                            neo4j_uri = config.get_env('NEO4J_URI', neo4j_config.get('uri', 'bolt://localhost:7687'))
                            neo4j_username = config.get_env('NEO4J_USERNAME', 'neo4j')
                            neo4j_password = config.get_env('NEO4J_PASSWORD')
                            neo4j_database = config.get_env('NEO4J_DATABASE', neo4j_config.get('database', 'neo4j'))

                            # Crea la Person (setup_schema la crea)
                            temp_storage = Neo4jKnowledgeGraph(
                                uri=neo4j_uri,
                                username=neo4j_username,
                                password=neo4j_password,
                                database=neo4j_database,
                                person_id=new_person_id,
                                person_name=new_person_name
                            )

                            st.success(f"✅ Profilo '{new_person_name}' creato con successo!")
                            st.info(f"💡 Per usare questo profilo, riavvia l'app o cambia profilo dalla lista")

                        except Exception as e:
                            st.error(f"❌ Errore creazione profilo: {str(e)}")
                    else:
                        st.warning("⚠️ Inserisci un nome per il profilo")

            with profile_tab2:
                st.markdown("⚠️ **Attenzione:** Eliminare un profilo rimuove tutti i suoi dati dal Knowledge Graph!")

                if all_persons:
                    profile_to_delete = st.selectbox(
                        "Seleziona profilo da eliminare:",
                        options=[p['id'] for p in all_persons],
                        format_func=lambda x: next((f"{p['name']} ({p['id']})" for p in all_persons if p['id'] == x), x),
                        key="profile_to_delete"
                    )

                    if st.button("🗑️ Elimina Profilo", type="secondary", key="delete_profile_btn"):
                        if profile_to_delete == storage.person_id:
                            st.error("❌ Non puoi eliminare il profilo correntemente in uso!")
                        else:
                            try:
                                storage.delete_person(profile_to_delete)
                                st.success(f"✅ Profilo '{profile_to_delete}' eliminato")
                            except Exception as e:
                                st.error(f"❌ Errore eliminazione profilo: {str(e)}")
                else:
                    st.info("Nessun profilo disponibile per l'eliminazione")

        # Selettore profilo
        if all_persons and len(all_persons) > 1:
            st.markdown("---")
            selected_person_id = st.selectbox(
                "🔄 Cambia Profilo:",
                options=[p['id'] for p in all_persons],
                index=[p['id'] for p in all_persons].index(storage.person_id) if storage.person_id in [p['id'] for p in all_persons] else 0,
                format_func=lambda x: next((f"{p['name']} ({p['id']})" for p in all_persons if p['id'] == x), x),
                key="selected_person_selector"
            )

            if selected_person_id != storage.person_id:
                st.session_state['selected_person_id'] = selected_person_id
                st.session_state['selected_person_name'] = next((p['name'] for p in all_persons if p['id'] == selected_person_id), selected_person_id)
                st.info("💡 Ricarica la pagina per applicare il cambio profilo")
                st.button("🔄 Ricarica App", on_click=lambda: st.rerun())

    # Sezione caricamento triplette
    st.markdown("---")
    st.subheader("📂 Caricamento Triplette")

    triplets_to_process = []
    source_info = ""

    # Opzioni di input
    input_option = st.radio(
        "Sorgente dati:",
        ["Carica da File JSON", "Recupera da Session State"],
        horizontal=True
    )

    if input_option == "Carica da File JSON":
        uploaded_file = st.file_uploader("Carica file JSON con triplette", type=['json'])

        if uploaded_file:
            try:
                triplets_data = json.load(uploaded_file)

                # Rileva il formato
                if isinstance(triplets_data, list):
                    triplets_to_process = triplets_data
                elif isinstance(triplets_data, dict):
                    # Prova a estrarre da chiavi comuni
                    for key in ['triplets', 'data', 'results', 'items']:
                        if key in triplets_data and isinstance(triplets_data[key], list):
                            triplets_to_process = triplets_data[key]
                            break

                if triplets_to_process:
                    source_info = f"file {uploaded_file.name}"
                    st.success(f"✅ Caricato file con {len(triplets_to_process)} triplette")

                    with st.expander("🔍 Preview Triplette"):
                        st.json(triplets_to_process[:3])

            except json.JSONDecodeError as e:
                st.error(f"❌ Errore nel parsing JSON: {str(e)}")
            except Exception as e:
                st.error(f"❌ Errore: {str(e)}")

    else:
        # Recupera da session state
        if 'validated_triplets' in st.session_state:
            triplets_to_process = st.session_state['validated_triplets']
            source_info = "Session State (tab Ontology Validation)"
            st.success(f"✅ Recuperate {len(triplets_to_process)} triplette da Session State")
        else:
            st.info("💡 Nessuna tripletta trovata in Session State. Esegui prima la validazione ontologica.")

    # Configurazione e costruzione KG
    if triplets_to_process:
        st.success(f"✅ Trovate **{len(triplets_to_process)}** triplette da processare ({source_info})")

        with st.sidebar:
            st.markdown("---")
            st.markdown("### 🛠️ KG Builder Config")

            ontology_check_enabled = st.checkbox(
                "Abilita check ontologico",
                value=True,
                help="Se disabilitato, le triplette vengono processate senza validazione ontologica"
            )

            # Pulsante per pulire nodi legacy (solo Neo4j)
            if storage_type == 'neo4j' and hasattr(storage, 'cleanup_legacy_nodes'):
                st.markdown("---")
                st.markdown("### 🧹 Manutenzione")

                if st.button("🧹 Pulisci Nodi Legacy", key="sidebar_cleanup_btn", help="Rimuove nodi BroaderTopic/NarrowerTopic/Triplet del vecchio schema"):
                    try:
                        with st.spinner("Pulizia in corso..."):
                            cleanup_result = storage.cleanup_legacy_nodes()
                            if cleanup_result["deleted_broader_topics"] > 0 or cleanup_result["deleted_narrower_topics"] > 0:
                                st.success(f"✅ Rimossi: {cleanup_result['deleted_broader_topics']} broader, {cleanup_result['deleted_narrower_topics']} narrower, {cleanup_result['deleted_triplets']} triplet")
                            else:
                                st.info("✅ Nessun nodo legacy trovato")
                    except Exception as e:
                        st.error(f"❌ Errore: {str(e)}")

        # Pulsante per costruire il KG
        if st.button("🚀 Costruisci Knowledge Graph", type="primary"):
            with st.spinner("⏳ Processamento triplette in corso..."):
                progress_bar = st.progress(0.0)
                status_text = st.empty()

                try:
                    # Esegui il builder
                    result = kg_builder.run(
                        triplets=triplets_to_process,
                        ontology_check_enabled=ontology_check_enabled
                    )

                    progress_bar.progress(1.0)

                    if result["success"]:
                        st.success("✅ Knowledge Graph costruito con successo!")

                        # Mostra statistiche aggiornate
                        updated_stats = result["kg_stats"]
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Broad Topics", updated_stats["num_broader_topics"])
                        with col2:
                            st.metric("Narrow Topics", updated_stats["num_narrower_topics"])
                        with col3:
                            st.metric("Triplette Totali", updated_stats["num_triplets"])

                        # Mostra triplette processate
                        processed = result["processed_triplets"]
                        st.subheader(f"📋 Triplette Processate: {len(processed)}")

                        for idx, triplet in enumerate(processed[:10], 1):
                            broader = triplet.get("broader_topic", "N/A")
                            narrower = triplet.get("narrower_topic", "N/A")

                            # Estrai valori (gestisce sia dict che stringa)
                            def get_value(field):
                                val = triplet.get(field, "")
                                return val.get("value", "") if isinstance(val, dict) else str(val)

                            subj = get_value("subject")
                            pred = get_value("predicate")
                            obj = get_value("object")

                            with st.expander(f"Tripletta #{idx}: {broader} → {narrower}"):
                                st.markdown(f"**Topics:** `{broader}` → `{narrower}`")
                                st.markdown(f"**Tripletta:** {subj} → {pred} → {obj}")

                                # Metadata
                                metadata = triplet.get("topic_metadata", {})
                                if metadata:
                                    action = metadata.get("action", "N/A")
                                    st.markdown(f"**Action:** {action}")

                                # Reasoning
                                reasoning = triplet.get("classification_reasoning", "")
                                if reasoning:
                                    with st.expander("💡 Reasoning"):
                                        st.markdown(reasoning)

                        if len(processed) > 10:
                            st.info(f"Mostrate prime 10 triplette. Totale: {len(processed)}")

                        # Download risultati
                        json_output = json.dumps(processed, indent=2, ensure_ascii=False)
                        st.download_button(
                            label="⬇️ Download Triplette Processate (JSON)",
                            data=json_output,
                            file_name="kg_processed_triplets.json",
                            mime="application/json"
                        )

                    else:
                        st.error(f"❌ Errore nella costruzione del KG: {result.get('error', 'Unknown error')}")

                except Exception as e:
                    st.error(f"❌ Errore: {str(e)}")
                    import traceback
                    with st.expander("🔍 Traceback"):
                        st.code(traceback.format_exc())

    # Visualizzazione KG esistente
    st.markdown("---")
    st.subheader("📊 Visualizza Knowledge Graph Esistente")

    if st.button("🔍 Mostra Statistiche KG"):
        try:
            stats = kg_builder.get_kg_stats()

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🌐 Broad Topics", stats["num_broader_topics"])
            with col2:
                st.metric("🎯 Narrow Topics", stats["num_narrower_topics"])
            with col3:
                st.metric("🔗 Triplette", stats["num_triplets"])

            # Mostra tutti i topic se storage Neo4j
            if hasattr(storage, 'get_all_broader_topics'):
                st.markdown("### 🌐 Tutti i Broad Topics")
                all_broader = storage.get_all_broader_topics()
                for topic in all_broader:
                    with st.expander(f"📁 {topic['name']}"):
                        st.markdown(f"**ID:** `{topic['id']}`")

                        # Mostra narrow topics collegati
                        narrow_topics = storage.get_narrower_topics_for_broader(topic['id'])
                        if narrow_topics:
                            st.markdown(f"**Narrow Topics ({len(narrow_topics)}):**")
                            for nt in narrow_topics:
                                st.markdown(f"- {nt['name']}")

        except Exception as e:
            st.error(f"❌ Errore recupero statistiche: {str(e)}")
