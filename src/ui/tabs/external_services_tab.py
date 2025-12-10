"""
Tab 4: External Services Integration (Gmail, Calendar, etc).
"""

import streamlit as st
import json
import requests


def render_external_services_tab(config) -> None:
    """
    Render the external services tab.

    Args:
        config: ConfigManager instance
    """
    st.header("Servizi Esterni")
    st.write("Integrazione con Gmail e altri servizi")

    service_type = st.selectbox(
        "Tipo di servizio:",
        ["gmail", "calendar", "altro"]
    )

    external_data_input = st.text_area(
        "Dati dal servizio (formato JSON):",
        height=150,
        placeholder='{"subject": "...", "body": "...", ...}'
    )

    if st.button("Invia Dati Esterni", key="send_external"):
        if external_data_input:
            try:
                external_data = json.loads(external_data_input)

                with st.spinner("Invio dati al server MCP..."):
                    mcp_config = config.get_mcp_config()
                    mcp_url = f"http://{mcp_config.get('host')}:{mcp_config.get('port')}"

                    payload = {
                        "source": service_type,
                        "data_id": "external_001",  # TODO: generare ID univoco
                        "timestamp": "2024-01-01T00:00:00Z",
                        "content": external_data
                    }

                    response = requests.post(
                        f"{mcp_url}/api/external/gmail",
                        json=payload,
                        timeout=10
                    )

                    if response.status_code == 200:
                        st.success("Dati inviati al server MCP!")
                        st.json(response.json())
                    else:
                        st.error(f"Errore server MCP: {response.status_code}")

            except json.JSONDecodeError:
                st.error("Formato JSON non valido")
            except requests.exceptions.ConnectionError:
                st.error("Impossibile connettersi al server MCP. Assicurati che sia in esecuzione.")
            except Exception as e:
                st.error(f"Errore: {str(e)}")
        else:
            st.warning("Inserisci i dati da inviare")
