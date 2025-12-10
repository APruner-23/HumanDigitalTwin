"""
Tab 3: IoT Data Generation and Management.
"""

import streamlit as st
import json
import requests
from datetime import datetime
from src.data_generator import OntologyDataGenerator
from ..services.app_state import AppState


def render_iot_data_tab(config, prompt_manager, llm) -> None:
    """
    Render the IoT data tab.

    Args:
        config: ConfigManager instance
        prompt_manager: PromptManager instance
        llm: LLM instance
    """
    st.header("Analisi Dati IoT")
    st.write("Genera o inserisci dati IoT conformi all'ontologia")

    # Initialize generator
    if not AppState.get_data_generator():
        AppState.set_data_generator(OntologyDataGenerator())

    generator = AppState.get_data_generator()

    # Two modes: Generate or Manual Insert
    mode = st.radio(
        "Modalità:",
        ["Genera da Ontologia", "Inserisci Manualmente"],
        horizontal=True
    )

    mcp_config = config.get_mcp_config()
    mcp_url = f"http://{mcp_config.get('host')}:{mcp_config.get('port')}"

    # Mode 1: Generate from Ontology
    if mode == "Genera da Ontologia":
        st.subheader("Generazione Automatica Dati")

        col1, col2 = st.columns(2)

        with col1:
            device_type = st.selectbox(
                "Tipo di dispositivo:",
                generator.get_available_devices()
            )

            device_id = st.text_input(
                "ID Dispositivo (opzionale):",
                placeholder="Lascia vuoto per generazione automatica"
            )

        with col2:
            num_records = st.slider(
                "Numero di record da generare:",
                min_value=1,
                max_value=50,
                value=5
            )

            time_interval = st.slider(
                "Intervallo tra record (minuti):",
                min_value=1,
                max_value=1440,
                value=60
            )

        # Device info
        with st.expander("Info Dispositivo e Metriche"):
            device_info = generator.get_device_metrics(device_type)
            st.write(f"**Sensori:** {', '.join(device_info['sensors'])}")
            st.write(f"**Metriche disponibili:** {len(device_info['metrics'])}")

            metrics_list = []
            for metric, range_val in device_info['metrics'].items():
                if range_val:
                    metrics_list.append(f"- {metric}: {range_val}")
                else:
                    metrics_list.append(f"- {metric}: (stringa)")

            st.text("\n".join(metrics_list[:10]))
            if len(metrics_list) > 10:
                st.text(f"... e altre {len(metrics_list) - 10} metriche")

        if st.button("Genera e Invia Dati", key="generate_iot"):
            with st.spinner("Generazione dati..."):
                try:
                    # Generate data
                    generated_data = generator.generate_data(
                        device_type=device_type,
                        device_id=device_id if device_id else None,
                        num_records=num_records,
                        time_interval_minutes=time_interval
                    )

                    st.success(f"Generati {len(generated_data)} record!")

                    # Preview
                    with st.expander("Preview Dati Generati"):
                        st.json(generated_data[0])

                    # Send to MCP server
                    success_count = 0
                    for record in generated_data:
                        try:
                            response = requests.post(
                                f"{mcp_url}/api/iot/data",
                                json=record,
                                timeout=10
                            )
                            if response.status_code == 200:
                                success_count += 1
                        except Exception as e:
                            st.warning(f"Errore invio record: {str(e)}")

                    if success_count == len(generated_data):
                        st.success(f"Tutti i {success_count} record inviati al server MCP!")
                    else:
                        st.warning(f"Inviati {success_count}/{len(generated_data)} record")

                except Exception as e:
                    st.error(f"Errore: {str(e)}")

    # Mode 2: Manual Insert
    else:
        st.subheader("Inserimento Manuale")

        device_type = st.selectbox(
            "Tipo di dispositivo:",
            ["fitbit", "garmin", "jawbone", "altro"]
        )

        device_id = st.text_input("ID Dispositivo:", placeholder="device_001")

        iot_data_input = st.text_area(
            "Dati IoT (formato JSON):",
            height=150,
            placeholder='{"heartrate": 75, "steps": 5000, ...}'
        )

        if st.button("Analizza Dati IoT", key="analyze_iot"):
            if device_id and iot_data_input:
                try:
                    # Validate JSON
                    iot_data = json.loads(iot_data_input)

                    with st.spinner("Invio dati al server MCP..."):
                        payload = {
                            "device_type": device_type,
                            "device_id": device_id,
                            "timestamp": datetime.now().isoformat(),
                            "data": iot_data
                        }

                        response = requests.post(
                            f"{mcp_url}/api/iot/data",
                            json=payload,
                            timeout=10
                        )

                        if response.status_code == 200:
                            st.success("Dati inviati al server MCP!")
                            st.json(response.json())

                            # Analyze with LLM
                            with st.spinner("Analisi in corso..."):
                                messages = prompt_manager.build_messages(
                                    'iot_data_processing',
                                    iot_data=json.dumps(iot_data, indent=2)
                                )

                                llm_response = llm.generate_with_history(messages)

                                st.subheader("Analisi:")
                                st.write(llm_response)
                        else:
                            st.error(f"Errore server MCP: {response.status_code}")

                except json.JSONDecodeError:
                    st.error("Formato JSON non valido")
                except requests.exceptions.ConnectionError:
                    st.error("Impossibile connettersi al server MCP. Assicurati che sia in esecuzione.")
                except Exception as e:
                    st.error(f"Errore: {str(e)}")
            else:
                st.warning("Compila tutti i campi")
