"""
LangGraph pipeline per estrazione multi-stage di triplette da testo con augmentation IoT.
"""

from typing import TypedDict, List, Dict, Optional, Any, Annotated
from langgraph.graph import StateGraph, END
from src.llm.groq_failover import GroqFailover
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field, field_validator, model_validator
import operator
from copy import deepcopy
import os, json
from datetime import datetime
from pathlib import Path


# Pydantic models per structured output
class TripletEntity(BaseModel):
    """Entità con valore e tipo."""
    value: str = Field(description="Valore dell'entità (es. 'Marco')")
    type: str = Field(description="Tipo/classe dell'entità (es. 'Person', 'Location', 'Integer')")

    @field_validator("value", "type", mode="before")
    @classmethod
    def coerce_to_string(cls, v):
        if v is None:
            return ""
        if isinstance(v, str):
            return v
        if isinstance(v, (int, float, bool)):
            return str(v)
        try:
            return json.dumps(v, ensure_ascii=False)
        except Exception:
            return str(v)


class Triplet(BaseModel):
    """Singola tripletta RDF con tipizzazione."""
    subject: TripletEntity = Field(description="Soggetto della tripletta con tipo")
    predicate: TripletEntity = Field(description="Predicato/relazione della tripletta con tipo")
    object: TripletEntity = Field(description="Oggetto della tripletta con tipo")


class TripletList(BaseModel):
    """Lista di triplette."""
    triplets: List[Triplet] = Field(description="Lista di triplette RDF estratte")

    @model_validator(mode="before")
    @classmethod
    def sanitize_triplets(cls, data):
        if not isinstance(data, dict):
            return data

        raw_triplets = data.get("triplets", [])
        if not isinstance(raw_triplets, list):
            data["triplets"] = []
            return data

        cleaned_triplets = []

        for t in raw_triplets:
            if not isinstance(t, dict):
                continue

            subj = t.get("subject")
            pred = t.get("predicate")
            obj = t.get("object")

            # Scarta triplette incomplete prima che Pydantic esploda
            if not isinstance(subj, dict) or not isinstance(pred, dict) or not isinstance(obj, dict):
                continue

            if "value" not in subj or "value" not in pred or "value" not in obj:
                continue

            # Se manca il type, usa fallback minimale
            subj.setdefault("type", "Unknown")
            pred.setdefault("type", "Unknown")
            obj.setdefault("type", "Unknown")

            cleaned_triplets.append(
                {
                    "subject": subj,
                    "predicate": pred,
                    "object": obj,
                }
            )

        data["triplets"] = cleaned_triplets
        return data

    def filter_valid(self) -> List[Triplet]:
        return [
            t for t in self.triplets
            if t.subject.value and t.predicate.value and t.object.value
        ]


class GraphState(TypedDict):
    """
    State del grafo per l'estrazione di triplette.

    Campi:
    - input_text: Testo originale da processare
    - chunk_size: Dimensione dei chunk (numero di caratteri)
    - chunks: Lista di chunk di testo
    - current_chunk_index: Indice del chunk corrente in elaborazione
    - previous_summary: Summary del chunk precedente
    - triplets: Lista accumulata di tutte le triplette estratte
    - augmented_triplets: Triplette aggiunte dall'augmentation (text + IoT)
    - final_triplets: Triplette finali (estratte + augmented)
    - error: Eventuale errore durante l'elaborazione

    IoT ReAct Loop:
    - iot_should_explore: Flag per decidere se iniziare esplorazione IoT
    - iot_react_iteration: Contatore iterazioni ReAct (max 5)
    - iot_data_collected: Dati IoT raccolti durante l'esplorazione
    - iot_reasoning_history: Storia del reasoning dell'agent IoT
    - iot_conversation: Conversazione completa con LLM (per context in loop)
    """
    input_text: str
    chunk_size: int
    chunks: List[str]
    current_chunk_index: int
    previous_summary: Optional[str]
    summary_history: List[str]  # Storia delle summary generate
    triplets: Annotated[List[Dict[str, str]], operator.add]
    augmented_triplets: Annotated[List[Dict[str, str]], operator.add]
    final_triplets: List[Dict[str, str]]
    error: Optional[str]

    # IoT ReAct Loop State
    iot_should_explore: bool
    iot_react_iteration: int
    iot_data_collected: Dict[str, Any]
    iot_reasoning_history: Annotated[List[str], operator.add]
    iot_conversation: List[Any]  # Lista di messaggi LangChain

    # Validation Guardrail State
    validation_should_run: bool
    validation_iteration: int
    validation_reasoning: Annotated[List[str], operator.add]
    validated_triplets: List[Dict[str, str]]  # Triplette dopo validazione
    removed_triplets: Annotated[List[Dict[str, str]], operator.add]  # Triplette rimosse con reason


class TripletExtractionGraph:
    """
    Grafo LangGraph per estrazione multi-stage di triplette.

    Pipeline:
    1. Chunking del testo
    2. Per ogni chunk:
       - Crea summary del chunk precedente (se esiste)
       - Estrai triplette dal chunk corrente
    3. Analisi per augmentation IoT
    4. Augmentation delle triplette con dati IoT (se necessario)
    """

    def __init__(
        self,
        llm_api_key: str,
        llm_model: str = "meta-llama/llama-4-scout-17b-16e-instruct",
        mcp_base_url: str = "http://localhost:8000",
        temperature: float = 0.3,
        enable_logging: bool = True,
        extraction_only: bool = False,
        # 0 no context mode, 1 simple context (summary precedente), 2 summary ultime 2 iterazioni
        delta_mode: int = 1,
        metrics_dir: str = "token_metrics" 
    ):
        """
        Inizializza il grafo di estrazione triplette.

        Args:
            llm_api_key: API key per Groq
            llm_model: Modello LLM da utilizzare
            mcp_base_url: URL del server MCP
            temperature: Temperature per l'LLM
            enable_logging: Abilita logging dettagliato
            extraction_only: Se True, salta augmentation/IoT/validation (solo extraction)
        """
        self.llm = GroqFailover(
            model_name=llm_model,
            temperature=temperature
        )
        self.mcp_base_url = mcp_base_url
        self.enable_logging = enable_logging
        self.extraction_only = extraction_only
        self.delta_mode = delta_mode
        self.metrics_dir = Path(metrics_dir)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self._current_run_metrics = None

        # Directory per debug della fase del nodo di validazione
        self.validation_debug_dir = self.metrics_dir / "validation_debug"
        self.validation_debug_dir.mkdir(parents=True, exist_ok=True)

        self.subject_repair_dir = self.metrics_dir / "subject_repairs"
        self.subject_repair_dir.mkdir(parents=True, exist_ok=True)

        self._current_run_id = None

        # Logger
        if self.enable_logging:
            from src.utils import get_logger
            self.logger = get_logger()
        else:
            self.logger = None

        # Prompt Manager
        from src.prompts import PromptManager
        self.prompt_manager = PromptManager()

        # MCP Tools per l'agent
        from src.mcp.mcp_tools import get_mcp_tools
        self.mcp_tools = get_mcp_tools(self.mcp_base_url)

        # Costruisci il grafo
        self.graph = self._build_graph()

        # Salva il diagramma Mermaid del grafo
        self._save_graph_diagram()

    def _init_token_metrics(self):
        """Inizializza la struttura di logging per una singola run della pipeline."""
        timestamp = datetime.now().isoformat(timespec="seconds")
        self._current_run_id = timestamp.replace(":", "-")

        self._current_run_metrics = {
            "timestamp": timestamp,
            "delta_mode": self.delta_mode,
            "stages": {
                "triple_extraction": {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "calls": []  # lista di chiamate per nodo
                },
                "context_generation": {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "calls": []
                },
                "triple_refinement": {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "calls": []
                },
            }
        }
    
    def _log_token_usage(self, stage: str, node: str, usage_metadata: dict):
        """
        Registra i token per una singola chiamata LLM.

        stage: 'triple_extraction' | 'context_generation' | 'triple_refinement'
        node: nome logico del nodo (es. 'extract_triples')
        usage_metadata: di solito usage_metadata={'input_tokens':..., 'output_tokens':..., 'total_tokens':...}
        """
        if not self.enable_logging:
            return
        if self._current_run_metrics is None:
            # Non inizializzato (safety)
            self._init_token_metrics()

        stage_bucket = self._current_run_metrics["stages"].get(stage)
        if not stage_bucket:
            return

        inp = usage_metadata.get("input_tokens", 0)
        out = usage_metadata.get("output_tokens", 0)
        tot = usage_metadata.get("total_tokens", 0)

        stage_bucket["input_tokens"] += inp
        stage_bucket["output_tokens"] += out
        stage_bucket["total_tokens"] += tot

        stage_bucket["calls"].append({
            "node": node,
            "input_tokens": inp,
            "output_tokens": out,
            "total_tokens": tot,
        })

    def _save_token_metrics(self, label: str = "run"):
        """Salva i token della run corrente in un file JSON in token_metrics/."""
        if not self.enable_logging or self._current_run_metrics is None:
            return

        ts = self._current_run_metrics.get("timestamp", datetime.now().isoformat(timespec="seconds"))
        safe_ts = ts.replace(":", "-")
        filename = f"{label}_{safe_ts}.json"
        path = self.metrics_dir / filename

        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self._current_run_metrics, f, indent=2, ensure_ascii=False)
        except Exception as e:
            # In caso di problemi con IO non blocchiamo la pipeline
            print(f"[TOKEN METRICS] Errore salvataggio {path}: {e}")

    def _triplet_key(self, triplet: Dict[str, Any]) -> tuple:
        """Crea una chiave hashable per confrontare triplette."""
        subj = triplet.get("subject", {}) or {}
        pred = triplet.get("predicate", {}) or {}
        obj = triplet.get("object", {}) or {}

        return (
            str(subj.get("value", "")),
            str(subj.get("type", "")),
            str(pred.get("value", "")),
            str(pred.get("type", "")),
            str(obj.get("value", "")),
            str(obj.get("type", "")),
        )
    
    def _save_validation_snapshot(
        self,
        iteration: int,
        input_triplets: List[Dict[str, Any]],
        output_triplets: List[Dict[str, Any]]
    ) -> None:
        """Salva input/output/removed della validation in JSON."""
        if not self.enable_logging:
            return

        run_id = self._current_run_id or datetime.now().isoformat(timespec="seconds").replace(":", "-")
        run_dir = self.validation_debug_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        input_keys = {self._triplet_key(t): t for t in input_triplets}
        output_keys = {self._triplet_key(t): t for t in output_triplets}

        removed_keys = [k for k in input_keys.keys() if k not in output_keys]
        removed_triplets = [input_keys[k] for k in removed_keys]

        files = {
            f"validation_iter_{iteration}_input.json": input_triplets,
            f"validation_iter_{iteration}_output.json": output_triplets,
            f"validation_iter_{iteration}_removed.json": removed_triplets,
        }

        for filename, payload in files.items():
            path = run_dir / filename
            try:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2, ensure_ascii=False)
            except Exception as e:
                print(f"[VALIDATION DEBUG] Errore salvataggio {path}: {e}")

    
    def _is_user_placeholder(self, value: Any) -> bool:
        """Riconosce placeholder generici del soggetto principale."""
        if value is None:
            return False

        normalized = str(value).strip().lower()
        return normalized in {
            "user",
            "the user",
            "main user",
            "current user",
            "me",
        }


    def _repair_subject_placeholders(
        self,
        triplets: List[Dict[str, Any]],
        canonical_subject_name: str
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Sostituisce subject.value = user/User/... con il nome corretto del soggetto principale.
        Ritorna:
        - lista di triplette riparate
        - log delle riparazioni effettuate
        """
        if not canonical_subject_name or not triplets:
            return triplets, []

        repaired_triplets = []
        repairs_log = []

        for idx, triplet in enumerate(triplets):
            repaired_triplet = deepcopy(triplet)

            subject = repaired_triplet.get("subject", {})
            old_value = subject.get("value")

            if self._is_user_placeholder(old_value):
                subject["value"] = canonical_subject_name
                repaired_triplet["subject"] = subject

                repairs_log.append({
                    "index": idx,
                    "old_subject_value": old_value,
                    "new_subject_value": canonical_subject_name,
                    "triplet_before": triplet,
                    "triplet_after": repaired_triplet,
                })

            repaired_triplets.append(repaired_triplet)

        return repaired_triplets, repairs_log


    def _save_subject_repairs(
        self,
        canonical_subject_name: str,
        repairs: List[Dict[str, Any]]
    ) -> None:
        """Salva su file JSON il log delle triple riparate."""
        if not repairs:
            return

        run_id = self._current_run_id or datetime.now().isoformat(timespec="seconds").replace(":", "-")
        run_dir = self.subject_repair_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        payload = {
            "canonical_subject_name": canonical_subject_name,
            "repairs_count": len(repairs),
            "repairs": repairs
        }

        path = run_dir / "subject_repairs.json"

        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[SUBJECT REPAIR] Errore salvataggio {path}: {e}")

    
    def _extract_usage_metadata(self, response: Any) -> Optional[dict]:
        """
        Prova a estrarre i token da un oggetto response di LangChain/Groq.
        Ritorna un dict con chiavi input_tokens, output_tokens, total_tokens
        oppure None se non trova nulla.
        """
        # Caso 1: response.usage_metadata (pattern standard LangChain)
        usage = getattr(response, "usage_metadata", None)
        if isinstance(usage, dict):
            # Ci aspettiamo già queste chiavi, ma normalizziamo
            inp = usage.get("input_tokens") or usage.get("input") or 0
            out = usage.get("output_tokens") or usage.get("output") or 0
            tot = usage.get("total_tokens") or usage.get("total") or (inp + out)
            return {
                "input_tokens": inp,
                "output_tokens": out,
                "total_tokens": tot,
            }

        # Caso 2: response.response_metadata["token_usage"] / "usage"
        resp_meta = getattr(response, "response_metadata", None)
        if isinstance(resp_meta, dict):
            tu = resp_meta.get("token_usage") or resp_meta.get("usage") or {}
            if isinstance(tu, dict):
                inp = tu.get("input_tokens") or tu.get("prompt_tokens") or 0
                out = tu.get("output_tokens") or tu.get("completion_tokens") or 0
                tot = tu.get("total_tokens") or tu.get("total") or (inp + out)
                return {
                    "input_tokens": inp,
                    "output_tokens": out,
                    "total_tokens": tot,
                }

        # Nessuna info trovata
        return None

        

    def _build_graph(self) -> StateGraph:
        """Costruisce il grafo LangGraph con ReAct loop per IoT augmentation."""
        workflow = StateGraph(GraphState)

        # === NODI DEL GRAFO ===
        # Chunking & Extraction
        workflow.add_node("chunk_text", self._chunk_text_node)
        workflow.add_node("extract_triplets", self._extract_triplets_node)
        workflow.add_node("summarize_chunk", self._summarize_chunk_node)

        # Finalizzazione
        workflow.add_node("finalize", self._finalize_node)

        # === FLUSSO DEL GRAFO ===
        workflow.set_entry_point("chunk_text")

        # Chunk → Extract
        workflow.add_edge("chunk_text", "extract_triplets")

        if self.extraction_only:
            # ===== EXTRACTION ONLY MODE =====
            # Extract → Summarize OR Finalize (loop chunks, poi direttamente a finalize)
            workflow.add_conditional_edges(
                "extract_triplets",
                self._should_continue_chunks_simple,
                {
                    "summarize": "summarize_chunk",
                    "finalize": "finalize"
                }
            )
            # Summarize → Extract (loop back)
            workflow.add_edge("summarize_chunk", "extract_triplets")
        else:
            # ===== FULL PIPELINE MODE =====
            # Text Augmentation
            workflow.add_node("text_augmentation", self._text_augmentation_node)

            # IoT ReAct Loop (3 nodi)
            workflow.add_node("iot_decide", self._iot_decide_node)              # Decisione iniziale
            workflow.add_node("iot_react", self._iot_react_node)                # Iterazione ReAct (loop)
            workflow.add_node("iot_generate", self._iot_generate_triplets_node) # Genera triplette finali

            # Validation Guardrail (2 nodi)
            workflow.add_node("validation_decide", self._validation_decide_node)    # Decisione validazione
            workflow.add_node("validation_iterate", self._validation_iterate_node)  # Iterazione validazione (loop)

            # Extract → Summarize OR Text Augmentation (loop chunks)
            workflow.add_conditional_edges(
                "extract_triplets",
                self._should_continue_chunks,
                {
                    "summarize": "summarize_chunk",
                    "augment": "text_augmentation"
                }
            )

            # Summarize → Extract (loop back)
            workflow.add_edge("summarize_chunk", "extract_triplets")

            # Text Augmentation → IoT Decision
            workflow.add_edge("text_augmentation", "iot_decide")

            # IoT Decision → Explore OR Skip
            workflow.add_conditional_edges(
                "iot_decide",
                self._should_use_iot,
                {
                    "explore": "iot_react",           # Inizia ReAct loop
                    "skip": "validation_decide"       # Salta IoT, vai a validazione
                }
            )

            # IoT ReAct → Continue (self-loop) OR Finish
            workflow.add_conditional_edges(
                "iot_react",
                self._should_continue_react,
                {
                    "continue": "iot_react",      # Loop: chiama altri tools
                    "finish": "iot_generate"      # Genera triplette IoT
                }
            )

            # IoT Generate → Validation Decision
            workflow.add_edge("iot_generate", "validation_decide")

            # Validation Decision → Validate OR Skip
            workflow.add_conditional_edges(
                "validation_decide",
                self._should_validate,
                {
                    "validate": "validation_iterate",  # Inizia validation loop
                    "skip": "finalize"                  # Salta validazione
                }
            )

            # Validation Iterate → Continue (self-loop) OR Finish
            workflow.add_conditional_edges(
                "validation_iterate",
                self._should_continue_validation,
                {
                    "continue": "validation_iterate",  # Loop: refina ancora
                    "finish": "finalize"                # Validazione completa
                }
            )

        # Finalize → END
        workflow.add_edge("finalize", END)

        return workflow.compile()

    def _save_graph_diagram(self) -> None:
        """Salva il diagramma Mermaid del grafo come PNG."""
        try:
            from pathlib import Path

            # Crea la cartella assets se non esiste
            assets_dir = Path(__file__).parent.parent.parent / "assets"
            assets_dir.mkdir(exist_ok=True)

            # Genera e salva il diagramma
            output_path = assets_dir / "triplet_extraction_graph.png"
            png_data = self.graph.get_graph().draw_mermaid_png()

            with open(output_path, "wb") as f:
                f.write(png_data)

            msg = f"Graph diagram saved to: {output_path}"
            if self.logger:
                self.logger.console.print(f"[dim]{msg}[/dim]")
            else:
                print(msg)

        except Exception as e:
            msg = f"Warning: Could not save graph diagram: {str(e)}"
            if self.logger:
                self.logger.console.print(f"[yellow]{msg}[/yellow]")
            else:
                print(msg)

    def _chunk_text_node(self, state: GraphState) -> Dict[str, Any]:
        """
        Nodo 1: Divide il testo in chunk.

        Args:
            state: Stato corrente del grafo

        Returns:
            Update dello stato con i chunk creati
        """
        text = state["input_text"]
        chunk_size = state.get("chunk_size", 1000)

        # Log inizio nodo
        if self.logger:
            self.logger.console.print(f"\n[bold blue]═══ Chunking Node ═══[/bold blue]")
            self.logger.console.print(f"[dim]Text length: {len(text)} chars, Chunk size: {chunk_size}[/dim]")

        # Chunking semplice per caratteri
        chunks = []
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i + chunk_size]
            chunks.append(chunk)

        if self.logger:
            self.logger.console.print(f"[green]Created {len(chunks)} chunks[/green]")

        return {
            "chunks": chunks,
            "current_chunk_index": 0,
            "previous_summary": None,
            "summary_history": [],
            "triplets": []
        }

    def _extract_triplets_node(self, state: GraphState) -> Dict[str, Any]:
        """
        Nodo 2: Estrae triplette dal chunk corrente.

        Usa il contesto della summary del chunk precedente se disponibile.

        Args:
            state: Stato corrente del grafo

        Returns:
            Update dello stato con le triplette estratte
        """
        chunks = state["chunks"]
        current_index = state["current_chunk_index"]
        summary_history = state.get("summary_history", [])

        if current_index >= len(chunks):
            return {}

        current_chunk = chunks[current_index]

        # Log inizio nodo
        if self.logger:
            self.logger.console.print(
                f"\n[bold cyan]═══ Triplet Extraction Node - Chunk {current_index + 1}/{len(chunks)} ═══[/bold cyan]"
            )

        # 🔹 Costruisci il contesto in base a delta_mode
        context_text = ""
        if self.delta_mode == 0:
            # Nessun contesto
            context_text = ""
        elif self.delta_mode == 1:
            # Usa solo l'ultima summary, se esiste
            if summary_history:
                last = summary_history[-1]
                context_text = f"Context from previous chunk:\n{last}\n"
        else:
            # Usa le ultime N summary (N = delta_mode), o quante ce ne sono
            if summary_history:
                last_n = summary_history[-self.delta_mode:]
                joined = "\n\n".join(last_n)
                context_text = (
                    f"Context from last {len(last_n)} chunks:\n{joined}\n"
                )

        messages = self.prompt_manager.build_messages(
            'triplet_extraction_chunk',
            context=context_text,
            chunk=current_chunk
        )

        # Converti in formato Langchain
        from langchain_core.messages import HumanMessage, SystemMessage
        lc_messages = []
        for msg in messages:
            role = msg.get('role')
            content = msg.get('content')
            if role == 'system':
                lc_messages.append(SystemMessage(content=content))
            else:
                lc_messages.append(HumanMessage(content=content))

        # Log prompt
        if self.logger:
            self.logger.log_llm_call(messages, "", {"model": self.llm.model_name, "provider": "Groq"})

        try:
            # Usa structured output con Pydantic (json_mode for better compatibility)
            llm_with_structure = self.llm.with_structured_output(TripletList, method="json_mode")
            response = llm_with_structure.invoke(lc_messages)

            # 🔹 Logging token per Triple Extraction / extract_triplets
            usage = self._extract_usage_metadata(response)
            if usage:
                self._log_token_usage(
                    stage="triple_extraction",
                    node="extract_triplets",
                    usage_metadata=usage,
                )

            # Filtra solo triplette valide e converti da Pydantic a dict (matrice 2x3)
            valid_triplets = response.filter_valid()
            new_triplets = [
                {
                    "subject": {"value": t.subject.value, "type": t.subject.type},
                    "predicate": {"value": t.predicate.value, "type": t.predicate.type},
                    "object": {"value": t.object.value, "type": t.object.type}
                }
                for t in valid_triplets
            ]

            # Log risposta
            if self.logger:
                import json
                filtered_count = len(response.triplets) - len(valid_triplets)
                if filtered_count > 0:
                    self.logger.console.print(f"[dim]Filtered out {filtered_count} invalid triplets with null/empty values[/dim]")
                self.logger.log_agent_response(f"Extracted {len(new_triplets)} triplets:\n{json.dumps(new_triplets, indent=2)}")

            return {
                "triplets": new_triplets,
                "current_chunk_index": current_index + 1
            }

        except Exception as e:
            error_msg = f"Errore estrazione triplette chunk {current_index}: {str(e)}"
            if self.logger:
                self.logger.log_error(error_msg, "Triplet Extraction Node")
            else:
                print(error_msg)
            return {
                "error": f"Errore estrazione triplette: {str(e)}",
                "current_chunk_index": current_index + 1,
                "triplets": []
            }

    def _summarize_chunk_node(self, state: GraphState) -> Dict[str, Any]:
        """
        Nodo 3: Crea una summary del chunk appena processato.

        Questa summary sarà usata come contesto per il prossimo chunk.

        Args:
            state: Stato corrente del grafo

        Returns:
            Update dello stato con la summary
        """
        chunks = state["chunks"]
        current_index = state["current_chunk_index"]

        # Il chunk precedente è quello appena processato
        previous_chunk_index = current_index - 1

        if previous_chunk_index < 0 or previous_chunk_index >= len(chunks):
            return {
                "previous_summary": None,
                "summary_history": state.get("summary_history", [])
            }

        previous_chunk = chunks[previous_chunk_index]

        # Log inizio nodo
        if self.logger:
            self.logger.console.print(f"\n[bold magenta]═══ Summarization Node - Chunk {previous_chunk_index + 1} ═══[/bold magenta]")

        # Usa PromptManager
        messages = self.prompt_manager.build_messages(
            'triplet_summarization',
            chunk=previous_chunk
        )

        # Converti in formato Langchain
        lc_messages = []
        for msg in messages:
            role = msg.get('role')
            content = msg.get('content')
            if role == 'system':
                lc_messages.append(SystemMessage(content=content))
            else:
                lc_messages.append(HumanMessage(content=content))

        # Log prompt
        if self.logger:
            self.logger.log_llm_call(messages, "", {"model": self.llm.model_name, "provider": "Groq"})

        try:
            response = self.llm.invoke(lc_messages)

            # 🔹 Logging token per Triple Extraction / summarize_chunk
            usage = self._extract_usage_metadata(response)
            if usage:
                self._log_token_usage(
                    stage="triple_extraction",
                    node="summarize_chunk",
                    usage_metadata=usage,
                )

            summary = response.content.strip()

            # Log risposta
            if self.logger:
                self.logger.log_agent_response(f"Summary:\n{summary}")

            # 🔹 aggiorna la history delle summary
            history = state.get("summary_history", [])
            history = history + [summary]   # nuovo array per evitare di mutare in-place

            return {
                "previous_summary": summary,
                "summary_history": history
            }

        except Exception as e:
            error_msg = f"Errore summarization: {str(e)}"
            if self.logger:
                self.logger.log_error(error_msg, "Summarization Node")
            return {
                "error": error_msg,
                "previous_summary": None
            }

    def _should_continue_chunks(self, state: GraphState) -> str:
        """
        Conditional edge: decide se continuare con altri chunk o passare ad augmentation.

        Args:
            state: Stato corrente del grafo

        Returns:
            "summarize" se ci sono altri chunk da processare, "augment" altrimenti
        """
        current_index = state["current_chunk_index"]
        chunks = state["chunks"]

        if self.logger:
            self.logger.console.print(f"\n[bold cyan]→ Conditional Edge: _should_continue_chunks[/bold cyan]")
            self.logger.console.print(f"[dim]Chunk {current_index}/{len(chunks)}[/dim]")

        if current_index < len(chunks):
            if self.logger:
                self.logger.console.print(f"[green]→ Going to: summarize (more chunks to process)[/green]")
            return "summarize"
        else:
            if self.logger:
                self.logger.console.print(f"[yellow]→ Going to: augment (all chunks processed, starting augmentation)[/yellow]")
            return "augment"

    def _should_continue_chunks_simple(self, state: GraphState) -> str:
        """
        Conditional edge (extraction_only mode): decide se continuare con altri chunk o finalizzare.

        Args:
            state: Stato corrente del grafo

        Returns:
            "summarize" se ci sono altri chunk da processare, "finalize" altrimenti
        """
        current_index = state["current_chunk_index"]
        chunks = state["chunks"]

        if self.logger:
            self.logger.console.print(f"\n[bold cyan]→ Conditional Edge: _should_continue_chunks_simple (extraction_only)[/bold cyan]")
            self.logger.console.print(f"[dim]Chunk {current_index}/{len(chunks)}[/dim]")

        if current_index < len(chunks):
            if self.logger:
                self.logger.console.print(f"[green]→ Going to: summarize (more chunks to process)[/green]")
            return "summarize"
        else:
            if self.logger:
                self.logger.console.print(f"[yellow]→ Going to: finalize (all chunks processed, extraction complete)[/yellow]")
            return "finalize"

    def _text_augmentation_node(self, state: GraphState) -> Dict[str, Any]:
        """
        Nodo 4: Augmenta le triplette estratte con relazioni implicite e correzioni.

        Arricchisce il knowledge graph correggendo inconsistenze,
        aggiungendo relazioni implicite e rendendo i predicati coerenti.

        Args:
            state: Stato corrente del grafo

        Returns:
            Update dello stato con le triplette augmented (via text reasoning)
        """
        triplets = state.get("triplets", [])

        # Log inizio nodo
        if self.logger:
            self.logger.console.print(f"\n[bold yellow]═══ Text Augmentation Node ═══[/bold yellow]")
            self.logger.console.print(f"[dim]Refining and enriching {len(triplets)} extracted triplets[/dim]")

        if not triplets:
            if self.logger:
                self.logger.console.print(f"[dim]No triplets to augment, skipping[/dim]")
            return {"augmented_triplets": []}

        triplets_str = "\n".join([
            f"- {t.get('subject', {}).get('value', '')} {t.get('predicate', {}).get('value', '')} {t.get('object', {}).get('value', '')}"
            for t in triplets
        ])

        # Usa PromptManager
        messages = self.prompt_manager.build_messages(
            'text_augmentation',
            triplets=triplets_str
        )

        # Converti in formato Langchain
        lc_messages = []
        for msg in messages:
            role = msg.get('role')
            content = msg.get('content')
            if role == 'system':
                lc_messages.append(SystemMessage(content=content))
            else:
                lc_messages.append(HumanMessage(content=content))

        # Log prompt
        if self.logger:
            self.logger.log_llm_call(messages, "", {"model": self.llm.model_name, "provider": "Groq"})

        try:
            # Usa structured output con Pydantic (json_mode)
            llm_with_structure = self.llm.with_structured_output(TripletList, method="json_mode")
            response = llm_with_structure.invoke(lc_messages)

            # 🔹 Logging token per Context Generation / text_augmentation
            usage = self._extract_usage_metadata(response)
            if usage:
                self._log_token_usage(
                    stage="context_generation",
                    node="text_augmentation",
                    usage_metadata=usage,
                )

            # Filtra solo triplette valide e converti da Pydantic a dict (matrice 2x3)
            valid_triplets = response.filter_valid()
            augmented = [
                {
                    "subject": {"value": t.subject.value, "type": t.subject.type},
                    "predicate": {"value": t.predicate.value, "type": t.predicate.type},
                    "object": {"value": t.object.value, "type": t.object.type}
                }
                for t in valid_triplets
            ]

            # Log risposta
            if self.logger:
                import json
                filtered_count = len(response.triplets) - len(valid_triplets)
                if filtered_count > 0:
                    self.logger.console.print(f"[dim]Filtered out {filtered_count} invalid triplets with null/empty values[/dim]")
                self.logger.log_agent_response(f"Generated {len(augmented)} augmented triplets:\n{json.dumps(augmented, indent=2)}")

            return {"augmented_triplets": augmented}

        except Exception as e:
            error_msg = f"Errore text augmentation: {str(e)}"
            if self.logger:
                self.logger.log_error(error_msg, "Text Augmentation Node")
            return {
                "error": error_msg,
                "augmented_triplets": []
            }

    def _iot_decide_node(self, state: GraphState) -> Dict[str, Any]:
        """
        Nodo 4a: Decide se utilizzare dati IoT per augmentation.

        Analizza le triplette text-augmented e decide se ha senso
        arricchirle con dati da sensori IoT.

        Args:
            state: Stato corrente del grafo

        Returns:
            Update dello stato con la decisione (iot_should_explore)
        """
        augmented_triplets = state.get("augmented_triplets", [])

        # Log inizio nodo
        if self.logger:
            self.logger.console.print(f"\n[bold magenta]═══ IoT Decision Node ═══[/bold magenta]")
            self.logger.console.print(f"[dim]Analyzing {len(augmented_triplets)} triplets to decide if IoT data could add value[/dim]")

        if not augmented_triplets:
            if self.logger:
                self.logger.console.print(f"[yellow]No triplets to augment, skipping IoT exploration[/yellow]")
            return {
                "iot_should_explore": False,
                "iot_react_iteration": 0,
                "iot_data_collected": {},
                "iot_conversation": []
            }

        # Prepara le triplette per il prompt
        triplets_str = "\n".join([
            f"- {t.get('subject', {}).get('value', '')} {t.get('predicate', {}).get('value', '')} {t.get('object', {}).get('value', '')}"
            for t in augmented_triplets
        ])

        # Usa PromptManager per il prompt di decisione
        messages = self.prompt_manager.build_messages(
            'iot_decide',
            triplets=triplets_str
        )

        # Converti in formato Langchain
        lc_messages = []
        for msg in messages:
            role = msg.get('role')
            content = msg.get('content')
            if role == 'system':
                lc_messages.append(SystemMessage(content=content))
            else:
                lc_messages.append(HumanMessage(content=content))

        # Log prompt
        if self.logger:
            self.logger.log_llm_call(messages, "", {"model": self.llm.model_name, "provider": "Groq"})

        try:
            # Chiedi all'LLM se vale la pena esplorare IoT
            response = self.llm.invoke(lc_messages)

            # 🔹 Logging token per Context Generation / iot_decide
            usage = self._extract_usage_metadata(response)
            if usage:
                self._log_token_usage(
                    stage="context_generation",
                    node="iot_decide",
                    usage_metadata=usage,
                )

            decision_text = response.content.strip().upper()

            # Parsing semplice: cerca YES/SI o NO
            should_explore = any(keyword in decision_text for keyword in ["YES", "SI", "SÌ", "USEFUL", "RELEVANT"])

            # Log risposta
            if self.logger:
                decision_emoji = "✅" if should_explore else "❌"
                self.logger.log_agent_response(f"{decision_emoji} Decision: {'EXPLORE IoT data' if should_explore else 'SKIP IoT exploration'}\n\nReasoning:\n{response.content}")

            return {
                "iot_should_explore": should_explore,
                "iot_react_iteration": 0,
                "iot_data_collected": {},
                "iot_reasoning_history": [f"Initial Decision: {'Explore' if should_explore else 'Skip'}"],
                "iot_conversation": []  # Inizia vuoto, il ReAct node costruirà la sua conversazione
            }

        except Exception as e:
            error_msg = f"Errore IoT decision: {str(e)}"
            if self.logger:
                self.logger.log_error(error_msg, "IoT Decision Node")
            return {
                "error": error_msg,
                "iot_should_explore": False,
                "iot_react_iteration": 0,
                "iot_data_collected": {},
                "iot_conversation": []
            }

    def _iot_react_node(self, state: GraphState) -> Dict[str, Any]:
        """
        Nodo 4b: Esegue UNA iterazione del ReAct loop per IoT augmentation.

        Segue il pattern Think → Act → Observe:
        1. THINK: L'agent analizza i dati già raccolti e decide la prossima azione
        2. ACT: Chiama UN tool MCP (list_devices, get_latest_value, etc.)
        3. OBSERVE: Riceve il risultato e aggiorna lo state

        Il loop continua finché l'agent decide di fermarsi o max_iterations.

        Args:
            state: Stato corrente del grafo

        Returns:
            Update dello stato con:
            - iot_conversation: conversazione aggiornata
            - iot_data_collected: dati aggiornati
            - iot_react_iteration: contatore incrementato
            - iot_reasoning_history: reasoning dell'iterazione
        """
        # Recupera lo state del ReAct loop
        iteration = state.get("iot_react_iteration", 0)
        conversation = state.get("iot_conversation", [])
        data_collected = state.get("iot_data_collected", {})
        augmented_triplets = state.get("augmented_triplets", [])

        # Log inizio iterazione
        if self.logger:
            self.logger.console.print(f"\n[bold green]🔄 IoT ReAct Iteration {iteration + 1}/5[/bold green]")

            # Mostra i dati già raccolti
            if data_collected:
                from rich.panel import Panel
                import json
                data_panel = Panel(
                    json.dumps(data_collected, indent=2),
                    title="[yellow]📊 Data Collected So Far[/yellow]",
                    border_style="yellow"
                )
                self.logger.console.print(data_panel)

        # Se è la prima iterazione, inizializza la conversazione
        if not conversation:
            triplets_str = "\n".join([
                f"- {t.get('subject', {}).get('value', '')} {t.get('predicate', {}).get('value', '')} {t.get('object', {}).get('value', '')}"
                for t in augmented_triplets
            ])

            # Usa il prompt iot_react_iteration
            messages = self.prompt_manager.build_messages(
                'iot_react_iteration',
                triplets=triplets_str,
                data_collected=json.dumps(data_collected, indent=2) if data_collected else "None"
            )

            # Converti in formato Langchain
            for msg in messages:
                role = msg.get('role')
                content = msg.get('content')
                if role == 'system':
                    conversation.append(SystemMessage(content=content))
                else:
                    conversation.append(HumanMessage(content=content))

        # Log prompt (solo prima iterazione)
        if self.logger and iteration == 0:
            self.logger.log_llm_call(
                [{"role": m.type, "content": m.content} for m in conversation],
                "",
                {"model": self.llm.model_name, "provider": "Groq", "with_tools": len(self.mcp_tools)}
            )

        try:
            from langchain_core.messages import ToolMessage
            import json

            # Chiama LLM con i tools disponibili
            llm_with_tools = self.llm.bind_tools(self.mcp_tools)
            response = llm_with_tools.invoke(conversation)

            # 🔹 Logging token per Context Generation / iot_react
            usage = self._extract_usage_metadata(response)
            if usage:
                self._log_token_usage(
                    stage="context_generation",
                    node="iot_react",
                    usage_metadata=usage,
                )

            # 🤔 THINK: Log reasoning dell'agent
            reasoning = ""
            if hasattr(response, 'content') and response.content:
                reasoning = response.content
                if self.logger:
                    from rich.panel import Panel
                    think_panel = Panel(
                        reasoning,
                        title="[cyan]🤔 THINK: Agent Reasoning[/cyan]",
                        border_style="cyan",
                        padding=(1, 2)
                    )
                    self.logger.console.print(think_panel)

            # Aggiungi la risposta AI alla conversazione
            conversation.append(response)

            # Se ci sono tool calls: ACT + OBSERVE
            if hasattr(response, 'tool_calls') and response.tool_calls:
                # 🔧 ACT: Esegui i tool calls
                if self.logger:
                    self.logger.console.print(f"\n[bold yellow]🔧 ACT: Agent calling {len(response.tool_calls)} tool(s)[/bold yellow]")

                for idx, tool_call in enumerate(response.tool_calls, 1):
                    tool_name = tool_call['name']
                    tool_args = tool_call['args']

                    if self.logger:
                        from rich.panel import Panel
                        from rich.syntax import Syntax

                        # Mostra la chiamata
                        self.logger.console.print(f"\n[bold cyan]📞 Tool #{idx}: {tool_name}[/bold cyan]")
                        args_json = json.dumps(tool_args, indent=2)
                        args_syntax = Syntax(args_json, "json", theme="monokai", line_numbers=False)
                        args_panel = Panel(
                            args_syntax,
                            title=f"[yellow]Parameters[/yellow]",
                            border_style="yellow",
                            padding=(0, 1)
                        )
                        self.logger.console.print(args_panel)

                    # Trova ed esegui il tool
                    selected_tool = None
                    for tool in self.mcp_tools:
                        if tool.name == tool_name:
                            selected_tool = tool
                            break

                    if selected_tool:
                        tool_result = selected_tool.invoke(tool_args)

                        # 👁️ OBSERVE: Mostra il risultato
                        if self.logger:
                            result_str = str(tool_result)
                            try:
                                result_json = json.loads(result_str) if result_str.startswith(('{', '[')) else result_str
                                if isinstance(result_json, (dict, list)):
                                    result_formatted = json.dumps(result_json, indent=2)
                                    result_syntax = Syntax(result_formatted, "json", theme="monokai", line_numbers=False)
                                else:
                                    result_syntax = result_str
                            except:
                                result_syntax = result_str

                            observe_panel = Panel(
                                result_syntax,
                                title=f"[green]👁️  OBSERVE: Result[/green]",
                                border_style="green",
                                padding=(0, 1)
                            )
                            self.logger.console.print(observe_panel)

                            # Log nel file
                            self.logger.log_tool_call(tool_name, tool_args, str(tool_result))

                        # Aggiungi risultato alla conversazione
                        conversation.append(
                            ToolMessage(
                                content=str(tool_result),
                                tool_call_id=tool_call['id']
                            )
                        )

                        # Salva dati raccolti nello state
                        data_collected[f"{tool_name}_{iteration}"] = {
                            "tool": tool_name,
                            "args": tool_args,
                            "result": str(tool_result)
                        }

                # Incrementa iterazione e continua il loop
                return {
                    "iot_conversation": conversation,
                    "iot_data_collected": data_collected,
                    "iot_react_iteration": iteration + 1,
                    "iot_reasoning_history": [reasoning] if reasoning else []
                }

            else:
                # ✅ FINISH: Nessun tool chiamato, l'agent ha deciso di fermarsi
                if self.logger:
                    self.logger.console.print(f"\n[bold green]✅ DECIDE: Agent decided to FINISH (no more tools needed)[/bold green]")

                # L'agent ha finito, non continuare il loop
                return {
                    "iot_conversation": conversation,
                    "iot_react_iteration": iteration + 1,
                    "iot_reasoning_history": [reasoning] if reasoning else [],
                    "iot_should_explore": False  # Stop il loop
                }

        except Exception as e:
            error_msg = f"Errore IoT ReAct iteration: {str(e)}"
            if self.logger:
                self.logger.log_error(error_msg, "IoT ReAct Node")
            return {
                "error": error_msg,
                "iot_should_explore": False  # Stop il loop in caso di errore
            }

    def _iot_generate_triplets_node(self, state: GraphState) -> Dict[str, Any]:
        """
        Nodo 4c: Genera triplette IoT finali dai dati raccolti.

        Dopo aver completato il ReAct loop, questo nodo analizza tutti i dati
        raccolti e genera le triplette finali da aggiungere al knowledge graph.

        Args:
            state: Stato corrente del grafo

        Returns:
            Update dello stato con augmented_triplets (IoT)
        """
        conversation = state.get("iot_conversation", [])
        data_collected = state.get("iot_data_collected", {})
        augmented_triplets = state.get("augmented_triplets", [])

        # Log inizio nodo
        if self.logger:
            self.logger.console.print(f"\n[bold blue]═══ IoT Triplet Generation Node ═══[/bold blue]")
            self.logger.console.print(f"[dim]Generating triplets from collected IoT data[/dim]")

        if not data_collected:
            if self.logger:
                self.logger.console.print(f"[yellow]No IoT data collected, skipping triplet generation[/yellow]")
            # Non ritornare augmented_triplets: [] perché cancellerebbe le text-augmented!
            # Ritorna {} per non modificare lo state
            return {}

        try:
            import json

            # Prepare triplets string
            triplets_str = "\n".join([
                f"- {t.get('subject', {}).get('value', '')} {t.get('predicate', {}).get('value', '')} {t.get('object', {}).get('value', '')}"
                for t in augmented_triplets
            ])

            # Prepare data summary
            data_summary = json.dumps(data_collected, indent=2)

            # Usa PromptManager
            messages = self.prompt_manager.build_messages(
                'iot_generate_triplets',
                triplets=triplets_str,
                data_collected=data_summary
            )

            # Converti in formato Langchain
            conversation = state.get("iot_conversation", []) # Recupera conversazione corrente
            for msg in messages:
                role = msg.get('role')
                content = msg.get('content')
                if role == 'system':
                    # In teoria system message dovrebbe essere all'inizio, ma per ora appendiamo
                    conversation.append(SystemMessage(content=content))
                else:
                    conversation.append(HumanMessage(content=content))

            # Log final prompt
            if self.logger:
                self.logger.log_llm_call(
                    messages, 
                    "", 
                    {"model": self.llm.model_name, "provider": "Groq"}
                )

            # Genera triplette con structured output
            llm_with_structure = self.llm.with_structured_output(TripletList, method="json_mode")
            response = llm_with_structure.invoke(conversation)

            # 🔹 Logging token per Context Generation / iot_generate
            usage = self._extract_usage_metadata(response)
            if usage:
                self._log_token_usage(
                    stage="context_generation",
                    node="iot_generate",
                    usage_metadata=usage,
                )

            # Filtra e converti (matrice 2x3)
            valid_triplets = response.filter_valid()
            iot_triplets = [
                {
                    "subject": {"value": t.subject.value, "type": t.subject.type},
                    "predicate": {"value": t.predicate.value, "type": t.predicate.type},
                    "object": {"value": t.object.value, "type": t.object.type}
                }
                for t in valid_triplets
            ]

            # Log risultato
            if self.logger:
                filtered_count = len(response.triplets) - len(valid_triplets)
                if filtered_count > 0:
                    self.logger.console.print(f"[dim]Filtered out {filtered_count} invalid triplets[/dim]")
                self.logger.log_agent_response(f"Generated {len(iot_triplets)} IoT triplets:\n{json.dumps(iot_triplets, indent=2)}")

            # IMPORTANTE: operator.add funziona solo per nodi paralleli, non sequenziali!
            # Dobbiamo manualmente combinare con le augmented_triplets esistenti
            existing_augmented = augmented_triplets  # Già lette all'inizio del nodo
            combined = existing_augmented + iot_triplets

            if self.logger:
                self.logger.console.print(f"[blue]📊 Combining triplets:[/blue]")
                self.logger.console.print(f"[blue]  - Existing augmented (text): {len(existing_augmented)}[/blue]")
                self.logger.console.print(f"[blue]  - New IoT: {len(iot_triplets)}[/blue]")
                self.logger.console.print(f"[blue]  - Total combined: {len(combined)}[/blue]")

            return {"augmented_triplets": combined}

        except Exception as e:
            error_msg = f"Errore IoT triplet generation: {str(e)}"
            if self.logger:
                self.logger.log_error(error_msg, "IoT Generation Node")
            return {
                "error": error_msg,
                "augmented_triplets": []
            }

    def _validation_decide_node(self, state: GraphState) -> Dict[str, Any]:
        """
        Nodo 5a: Decide se eseguire validazione guardrail.

        Analizza tutte le triplette finali (estratte + augmented) e decide
        se serve validazione per consistenza e qualità.

        Args:
            state: Stato corrente del grafo

        Returns:
            Update dello stato con validation_should_run
        """
        extracted = state.get("triplets", [])
        augmented = state.get("augmented_triplets", [])
        all_triplets = extracted + augmented

        # Log inizio nodo
        if self.logger:
            self.logger.console.print(f"\n[bold blue]═══ Validation Decision Node ═══[/bold blue]")
            self.logger.console.print(f"[dim]Analyzing {len(all_triplets)} total triplets:[/dim]")
            self.logger.console.print(f"[dim]  - Extracted: {len(extracted)}[/dim]")
            self.logger.console.print(f"[dim]  - Augmented (text + IoT): {len(augmented)}[/dim]")
            self.logger.console.print(f"[dim]  - Total: {len(all_triplets)}[/dim]")

        # Se poche triplette, skippa validazione
        if len(all_triplets) < 5:
            if self.logger:
                self.logger.console.print(f"[yellow]Too few triplets ({len(all_triplets)}), skipping validation[/yellow]")
            return {
                "validation_should_run": False,
                "validation_iteration": 0,
                "validated_triplets": all_triplets
            }

        # Prepara prompt
        input_text = state.get("input_text", "")
        triplets_str = "\n".join([
            f"- {t.get('subject', {}).get('value', '')} {t.get('predicate', {}).get('value', '')} {t.get('object', {}).get('value', '')}"
            for t in all_triplets
        ])

        messages = self.prompt_manager.build_messages(
            'validation_decide',
            original_text=input_text[:500],  # Prime 500 char come context
            triplets=triplets_str,
            count=len(all_triplets)
        )

        lc_messages = []
        for msg in messages:
            role = msg.get('role')
            content = msg.get('content')
            if role == 'system':
                lc_messages.append(SystemMessage(content=content))
            else:
                lc_messages.append(HumanMessage(content=content))

        if self.logger:
            self.logger.log_llm_call(messages, "", {"model": self.llm.model_name, "provider": "Groq"})

        try:
            response = self.llm.invoke(lc_messages)
            # 🔹 Logging token per Triple Refinement / validation_decide
            usage = self._extract_usage_metadata(response)
            if usage:
                self._log_token_usage(
                    stage="triple_refinement",
                    node="validation_decide",
                    usage_metadata=usage,
                )
            decision = response.content.strip().upper()
            should_validate = any(kw in decision for kw in ["YES", "SI", "VALIDATE", "NEEDED", "INCONSISTEN"])

            if self.logger:
                emoji = "✅" if should_validate else "❌"
                self.logger.log_agent_response(f"{emoji} Decision: {'RUN validation' if should_validate else 'SKIP validation'}\n\n{response.content}")

            return {
                #"validation_should_run": should_validate,
                "validation_should_run": False,
                "validation_iteration": 0,
                "validated_triplets": all_triplets if not should_validate else [],
                "validation_reasoning": [f"Decision: {'Validate' if should_validate else 'Skip'}"]
            }

        except Exception as e:
            if self.logger:
                self.logger.log_error(f"Errore validation decision: {str(e)}", "Validation Decision")
            return {
                "validation_should_run": False,
                "validation_iteration": 0,
                "validated_triplets": all_triplets
            }

    def _validation_iterate_node(self, state: GraphState) -> Dict[str, Any]:
        """
        Nodo 5b: Esegue UNA iterazione di validazione guardrail.

        Il Guardrail Agent:
        1. Riceve testo originale + triplette correnti
        2. Identifica inconsistenze, ridondanze, errori semantici
        3. Rimuove/corregge triplette problematiche
        4. Decide se continuare validazione o finire

        Self-loop: può iterare fino a 3 volte per raffinare progressivamente.

        Args:
            state: Stato corrente del grafo

        Returns:
            Update con validated_triplets, removed_triplets, validation_iteration
        """
        iteration = state.get("validation_iteration", 0)
        current_triplets = state.get("validated_triplets", [])
        if not current_triplets:
            # Prima iterazione: prendi tutte
            current_triplets = state.get("triplets", []) + state.get("augmented_triplets", [])

        input_text = state.get("input_text", "")

        # Log
        if self.logger:
            self.logger.console.print(f"\n[bold magenta]🛡️  Validation Iteration {iteration + 1}/3[/bold magenta]")
            self.logger.console.print(f"[dim]Validating {len(current_triplets)} triplets against original text[/dim]")

        triplets_str = "\n".join([
            f"{i+1}. {t.get('subject', {}).get('value', '')} {t.get('predicate', {}).get('value', '')} {t.get('object', {}).get('value', '')}"
            for i, t in enumerate(current_triplets)
        ])

        messages = self.prompt_manager.build_messages(
            'validation_iterate',
            original_text=input_text,
            triplets=triplets_str,
            iteration=iteration + 1
        )

        lc_messages = []
        for msg in messages:
            role = msg.get('role')
            content = msg.get('content')
            if role == 'system':
                lc_messages.append(SystemMessage(content=content))
            else:
                lc_messages.append(HumanMessage(content=content))

        if self.logger and iteration == 0:
            self.logger.log_llm_call(messages, "", {"model": self.llm.model_name, "provider": "Groq"})

        try:
            import json

            # Structured output con Pydantic
            llm_with_structure = self.llm.with_structured_output(TripletList, method="json_mode")
            response = llm_with_structure.invoke(lc_messages)

            # 🔹 Logging token per Triple Refinement / validation_iterate
            usage = self._extract_usage_metadata(response)
            if usage:
                self._log_token_usage(
                    stage="triple_refinement",
                    node="validation_iterate",
                    usage_metadata=usage,
                )

            # Filtra triplette valide (matrice 2x3)
            valid = response.filter_valid()
            validated = [
                {
                    "subject": {"value": t.subject.value, "type": t.subject.type},
                    "predicate": {"value": t.predicate.value, "type": t.predicate.type},
                    "object": {"value": t.object.value, "type": t.object.type}
                }
                for t in valid
            ]

            self._save_validation_snapshot(
                iteration=iteration + 1,
                input_triplets=current_triplets,
                output_triplets=validated
            )

            # Identifica rimosse
            removed_count = len(current_triplets) - len(validated)

            # Reasoning: decide se continuare
            #should_continue = removed_count > 0 and iteration < 2  # Max 3 iterazioni

            # Solo per debug: Una sola iterazione
            should_continue = False

            # Log
            if self.logger:
                from rich.panel import Panel
                if removed_count > 0:
                    self.logger.console.print(f"[yellow]🗑️  Removed {removed_count} inconsistent triplets[/yellow]")
                else:
                    self.logger.console.print(f"[green]✅ All triplets validated, no changes needed[/green]")

                decision = "CONTINUE validation" if should_continue else "FINISH validation"
                self.logger.log_agent_response(f"Iteration {iteration + 1}: Validated {len(validated)}/{len(current_triplets)} triplets\n\nDecision: {decision}")

            removed_triplets = [
                t for t in current_triplets
                if self._triplet_key(t) not in {self._triplet_key(v) for v in validated}
            ]

            return {
                "validated_triplets": validated,
                "removed_triplets": removed_triplets,
                "validation_iteration": iteration + 1,
                "validation_reasoning": [f"Iteration {iteration + 1}: {removed_count} removed"],
                "validation_should_run": should_continue
            }

        except Exception as e:
            if self.logger:
                self.logger.log_error(f"Errore validation iteration: {str(e)}", "Validation Iterate")
            # In caso di errore, passa tutte le triplette senza modifiche
            return {
                "validated_triplets": current_triplets,
                "validation_iteration": iteration + 1,
                "validation_should_run": False
            }

    def _should_use_iot(self, state: GraphState) -> str:
        """
        Conditional edge: decide se iniziare l'esplorazione IoT.

        Args:
            state: Stato corrente del grafo

        Returns:
            "explore" se l'agent ha deciso di usare IoT, "skip" altrimenti
        """
        should_explore = state.get("iot_should_explore", False)

        if self.logger:
            self.logger.console.print(f"\n[bold cyan]→ Conditional Edge: _should_use_iot[/bold cyan]")
            self.logger.console.print(f"[dim]Decision: {'EXPLORE' if should_explore else 'SKIP'}[/dim]")

        if should_explore:
            if self.logger:
                self.logger.console.print(f"[green]→ Going to: iot_react (start ReAct loop)[/green]")
            return "explore"
        else:
            if self.logger:
                self.logger.console.print(f"[yellow]→ Going to: finalize (skip IoT)[/yellow]")
            return "skip"

    def _should_continue_react(self, state: GraphState) -> str:
        """
        Conditional edge: decide se continuare il ReAct loop.

        Controlla:
        - Se l'agent ha deciso di continuare (iot_should_explore)
        - Se non abbiamo raggiunto max_iterations (5)

        Args:
            state: Stato corrente del grafo

        Returns:
            "continue" per continuare il loop, "finish" per generare triplette
        """
        iteration = state.get("iot_react_iteration", 0)
        should_explore = state.get("iot_should_explore", True)
        max_iterations = 5

        if self.logger:
            self.logger.console.print(f"\n[bold cyan]→ Conditional Edge: _should_continue_react[/bold cyan]")
            self.logger.console.print(f"[dim]Iteration: {iteration}/{max_iterations}, Should explore: {should_explore}[/dim]")

        # Continua se: ancora vuole esplorare AND non ha raggiunto max iterations
        if should_explore and iteration < max_iterations:
            if self.logger:
                self.logger.console.print(f"[green]➡️  CONTINUE: More data needed[/green]")
            return "continue"
        else:
            reason = "max iterations reached" if iteration >= max_iterations else "agent decided to stop"
            if self.logger:
                self.logger.console.print(f"[yellow]✅ FINISH: {reason}[/yellow]")
            return "finish"

    def _should_validate(self, state: GraphState) -> str:
        """
        Conditional edge: decide se iniziare validazione guardrail.

        Args:
            state: Stato corrente del grafo

        Returns:
            "validate" per avviare validazione, "skip" per saltare
        """
        should_validate = state.get("validation_should_run", False)

        if self.logger:
            self.logger.console.print(f"\n[bold cyan]→ Conditional Edge: _should_validate[/bold cyan]")
            self.logger.console.print(f"[dim]Decision: {'VALIDATE' if should_validate else 'SKIP'}[/dim]")

        if should_validate:
            if self.logger:
                self.logger.console.print(f"[green]→ Going to: validation_iterate (start validation loop)[/green]")
            return "validate"
        else:
            if self.logger:
                self.logger.console.print(f"[yellow]→ Going to: finalize (skip validation)[/yellow]")
            return "skip"

    def _should_continue_validation(self, state: GraphState) -> str:
        """
        Conditional edge: decide se continuare il loop di validazione.

        Controlla:
        - Se l'agent ha deciso di continuare (validation_should_run)
        - Se non abbiamo raggiunto max_iterations (3)

        Args:
            state: Stato corrente del grafo

        Returns:
            "continue" per iterare ancora, "finish" per terminare
        """
        iteration = state.get("validation_iteration", 0)
        should_continue = state.get("validation_should_run", False)
        max_iterations = 3

        if self.logger:
            self.logger.console.print(f"\n[bold cyan]→ Conditional Edge: _should_continue_validation[/bold cyan]")
            self.logger.console.print(f"[dim]Iteration: {iteration}/{max_iterations}, Should continue: {should_continue}[/dim]")

        if should_continue and iteration < max_iterations:
            if self.logger:
                self.logger.console.print(f"[green]➡️  CONTINUE: More validation needed[/green]")
            return "continue"
        else:
            reason = "max iterations reached" if iteration >= max_iterations else "validation complete"
            if self.logger:
                self.logger.console.print(f"[yellow]✅ FINISH: {reason}[/yellow]")
            return "finish"

    def _finalize_node(self, state: GraphState) -> Dict[str, Any]:
        """
        Nodo 6: Finalizza con le triplette validate (se la validazione è stata eseguita).

        Args:
            state: Stato corrente del grafo

        Returns:
            Update dello stato con le triplette finali
        """
        # Se validazione eseguita, usa validated_triplets; altrimenti usa extracted + augmented
        validated = state.get("validated_triplets", [])

        if validated:
            final = validated
            validation_run = True
        else:
            extracted = state.get("triplets", [])
            augmented = state.get("augmented_triplets", [])
            final = extracted + augmented
            validation_run = False

        # Log finale
        if self.logger:
            self.logger.console.print(f"\n[bold white]═══ Finalize Node ═══[/bold white]")
            self.logger.console.print(f"[green]Final triplets: {len(final)}[/green]")

            if validation_run:
                validation_iterations = state.get("validation_iteration", 0)
                self.logger.console.print(f"[blue]  ✓ Validated through {validation_iterations} guardrail iteration(s)[/blue]")
            else:
                extracted = state.get("triplets", [])
                augmented = state.get("augmented_triplets", [])
                self.logger.console.print(f"  - Extracted from text: {len(extracted)}")
                self.logger.console.print(f"  - Augmented (text + IoT): {len(augmented)}")

            self.logger.log_summary()

        return {"final_triplets": final}

    def run(
        self,
        input_text: str,
        chunk_size: int = 1000,
        canonical_subject_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Esegue il grafo di estrazione triplette.

        Args:
            input_text: Testo da cui estrarre triplette
            chunk_size: Dimensione dei chunk in caratteri
            canonical_subject_name: Nome canonico del soggetto principale del dataset/profilo.
                Se valorizzato, sostituisce placeholder come "user" o "User" nel subject.value
                delle triplette estratte.

        Returns:
            Risultato finale con tutte le triplette estratte e augmented
        """

        if self.enable_logging:
            self._init_token_metrics()

        # Inizializza lo stato
        initial_state = {
            # Text processing
            "input_text": input_text,
            "chunk_size": chunk_size,
            "chunks": [],
            "current_chunk_index": 0,
            "previous_summary": None,
            "summary_history": [],

            # Triplets
            "triplets": [],
            "augmented_triplets": [],
            "final_triplets": [],

            # IoT ReAct Loop
            "iot_should_explore": False,
            "iot_react_iteration": 0,
            "iot_data_collected": {},
            "iot_reasoning_history": [],
            "iot_conversation": [],

            # Validation Guardrail
            "validation_should_run": False,
            "validation_iteration": 0,
            "validation_reasoning": [],
            "validated_triplets": [],
            "removed_triplets": [],

            # Error handling
            "error": None
        }

        # Esegui il grafo
        # Esegui il grafo con recursion limit aumentato per gestire molti chunk
        result = self.graph.invoke(initial_state, {"recursion_limit": 300})

        all_repairs = []

        if canonical_subject_name:
            buckets_to_repair = [
                "triplets",
                "augmented_triplets",
                "validated_triplets",
                "final_triplets"
            ]

            for bucket in buckets_to_repair:
                triplet_list = result.get(bucket, [])

                if isinstance(triplet_list, list) and triplet_list:
                    repaired_triplets, repairs = self._repair_subject_placeholders(
                        triplet_list,
                        canonical_subject_name
                    )

                    result[bucket] = repaired_triplets

                    for repair in repairs:
                        all_repairs.append({
                            "bucket": bucket,
                            **repair
                        })

            self._save_subject_repairs(canonical_subject_name, all_repairs)

            if self.logger and all_repairs:
                self.logger.console.print(
                    f"[yellow]Repaired {len(all_repairs)} triplets with subject placeholder -> {canonical_subject_name}[/yellow]"
                )

        if self.enable_logging:
            self._save_token_metrics(label="triple_pipeline")

        return result
