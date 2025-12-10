"""
Component for loading sessions (saved or custom JSON files).
"""

import streamlit as st
import json
from typing import Optional
from src.utils import SessionManager
from ..services.app_state import AppState


def render_session_loader(config) -> Optional[str]:
    """
    Render session loader UI with three modes:
    - Current session in memory
    - Load from saved session
    - Load custom JSON file

    Args:
        config: ConfigManager instance

    Returns:
        Source info string if triplets loaded, None otherwise
    """
    st.subheader("📂 Carica Sessione")

    col1, col2 = st.columns([3, 1])

    with col1:
        load_mode = st.radio(
            "Sorgente triplette:",
            ["Sessione corrente (in memoria)", "Carica da sessione salvata", "Carica file JSON custom"],
            horizontal=False
        )

    # Mode 1: Load from saved session
    if load_mode == "Carica da sessione salvata":
        sessions_config = config.get('sessions', {})
        session_mgr = SessionManager(sessions_config.get('sessions_dir', 'data/sessions'))

        sessions = session_mgr.list_sessions()

        if not sessions:
            st.warning("⚠️ Nessuna sessione salvata trovata. Estrai triplette prima nella tab **Estrazione Triplette**.")
            return None
        else:
            st.info(f"📦 Trovate {len(sessions)} sessioni salvate")

            session_options = [
                f"{s['filename']} - {s['triplets_count']} triplette ({s['timestamp'][:19]})"
                for s in sessions
            ]

            selected_idx = st.selectbox(
                "Seleziona sessione da caricare:",
                range(len(sessions)),
                format_func=lambda i: session_options[i]
            )

            selected_session = sessions[selected_idx]

            with st.expander("🔍 Info Sessione"):
                st.write(f"**File:** {selected_session['filename']}")
                st.write(f"**Timestamp:** {selected_session['timestamp']}")
                st.write(f"**Triplette:** {selected_session['triplets_count']}")

                metadata = selected_session.get('metadata', {})
                if metadata:
                    st.write("**Metadata:**")
                    st.json(metadata)

            if st.button("📥 Carica Sessione", type="primary"):
                session_data = session_mgr.load_session(selected_session['filepath'])
                AppState.set_extracted_triplets(session_data['triplets'])
                AppState.set_loaded_from_file(selected_session['filename'])
                st.success(f"✅ Sessione caricata: {selected_session['filename']}")
                st.rerun()

    # Mode 2: Load custom JSON file
    elif load_mode == "Carica file JSON custom":
        uploaded_file = st.file_uploader("Carica file JSON con triplette", type=['json'])

        if uploaded_file:
            try:
                loaded_data = json.load(uploaded_file)

                if 'triplets' in loaded_data:
                    triplets = loaded_data['triplets']
                elif isinstance(loaded_data, list):
                    triplets = loaded_data
                else:
                    st.error("❌ Formato JSON non riconosciuto. Atteso: lista di triplette o oggetto con campo 'triplets'")
                    triplets = []

                if triplets:
                    st.success(f"✅ Caricato file con {len(triplets)} triplette")

                    with st.expander("🔍 Preview Triplette"):
                        st.json(triplets[:3])

                    if st.button("📥 Usa Queste Triplette", type="primary"):
                        AppState.set_extracted_triplets(triplets)
                        AppState.set_loaded_from_file(uploaded_file.name)
                        st.success(f"✅ Triplette caricate da: {uploaded_file.name}")
                        st.rerun()

            except json.JSONDecodeError as e:
                st.error(f"❌ Errore nel parsing JSON: {str(e)}")
            except Exception as e:
                st.error(f"❌ Errore: {str(e)}")

    # Mode 3: Current session (just check if exists)
    else:
        triplets = AppState.get_extracted_triplets()
        if not triplets:
            st.warning("⚠️ Nessuna tripletta in memoria. Vai alla tab **Estrazione Triplette** o carica una sessione salvata.")
            return None

    # Return source info if triplets exist
    triplets = AppState.get_extracted_triplets()
    if triplets:
        loaded_from = AppState.get_loaded_from_file()
        if loaded_from:
            return f" (caricato da: `{loaded_from}`)"
        return ""

    return None
