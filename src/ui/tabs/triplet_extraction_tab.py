"""
Tab 1: Triplet Extraction with LangGraph.
"""

import streamlit as st
import json
from pathlib import Path
from ..services.app_state import AppState
from ..components.triplet_display import display_triplets_list
from src.utils import SessionManager


def render_triplet_extraction_tab(config, triplet_graph) -> None:
    """
    Render the triplet extraction tab.

    Args:
        config: ConfigManager instance
        triplet_graph: TripletExtractionGraph instance
    """
    st.header("Estrazione Triplette da Testo con LangGraph")
    st.write("Estrazione multi-stage di triplette RDF con augmentation dati IoT")

    # Input mode selection
    input_mode = st.radio(
        "Sorgente del testo:",
        ["Textbox", "File JSON"],
        horizontal=True
    )

    text_input = None

    # Textbox mode
    if input_mode == "Textbox":
        text_input = st.text_area(
            "Testo da analizzare:",
            height=200,
            placeholder="Inserisci qui il testo..."
        )
    # JSON file mode
    else:
        uploaded_file = st.file_uploader("Carica file JSON", type=['json'])
        if uploaded_file:
            try:
                json_data = json.load(uploaded_file)
                text_input = json_data.get("text", json.dumps(json_data))
                st.text_area("Testo estratto:", text_input, height=150, disabled=True)
            except Exception as e:
                st.error(f"Errore nel parsing del JSON: {str(e)}")

    # Chunk size slider
    chunk_size = st.slider(
        "Dimensione chunk (caratteri):",
        min_value=200,
        max_value=3000,
        value=1000,
        step=100
    )

    # Extract button
    if st.button("Estrai Triplette con LangGraph", key="extract_triplets"):
        if text_input:
            with st.spinner("Estrazione multi-stage in corso..."):
                try:
                    # Run the graph
                    canonical_subject_name = st.session_state.get("selected_person_name")
                    result = triplet_graph.run(
                        input_text=text_input,
                        chunk_size=chunk_size,
                        canonical_subject_name=canonical_subject_name
                    )

                    # Display results
                    if result.get("error"):
                        st.error(f"Errore: {result['error']}")
                    else:
                        st.success("Estrazione completata!")

                        # Display extracted triplets
                        triplets = result.get("triplets", [])
                        display_triplets_list(triplets, "Triplette estratte", max_display=10)

                        # Display augmented triplets
                        augmented = result.get("augmented_triplets", [])
                        if augmented:
                            display_triplets_list(augmented, "Triplette augmented (text + IoT)", max_display=10)

                        # Display final triplets count
                        final = result.get("final_triplets", [])
                        st.subheader(f"✅ Totale triplette finali: {len(final)}")

                        # Save to session state
                        AppState.set_extracted_triplets(final)
                        AppState.set_extraction_result(result)

                        # Auto-save session
                        sessions_config = config.get('sessions', {})
                        if sessions_config.get('auto_save', True):
                            session_mgr = SessionManager(sessions_config.get('sessions_dir', 'data/sessions'))

                            metadata = {
                                'input_text_preview': text_input[:200] + "..." if len(text_input) > 200 else text_input,
                                'chunk_size': chunk_size,
                                'total_chunks': len(result.get('chunks', [])),
                                'extracted_count': len(result.get('triplets', [])),
                                'augmented_count': len(result.get('augmented_triplets', []))
                            }

                            saved_path = session_mgr.save_session(final, metadata)
                            AppState.set_last_saved_session(saved_path)
                            st.success(f"💾 Sessione salvata automaticamente: `{Path(saved_path).name}`")

                        # Download button
                        json_output = json.dumps(final, indent=2)
                        st.download_button(
                            label="Download triplette (JSON)",
                            data=json_output,
                            file_name="triplets.json",
                            mime="application/json"
                        )

                        # Info message
                        st.info("💡 Triplette salvate! Vai alla tab **Ontology Validation** per validarle con Schema.org")

                except Exception as e:
                    st.error(f"Errore durante l'estrazione: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
        else:
            st.warning("Inserisci del testo o carica un file JSON prima di procedere")
