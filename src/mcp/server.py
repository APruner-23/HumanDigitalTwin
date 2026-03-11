from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import uvicorn
from collections import defaultdict
import re

class MCPServer:
    """Server MCP per esporre API che iniettano informazioni al modello."""

    def __init__(self, host: str = "localhost", port: int = 8000, kg_storage=None, neo4j_config: Optional[Dict[str, Any]] = None):
        """
        Inizializza il server MCP.

        Args:
            host: Host su cui far girare il server
            port: Porta su cui far girare il server
            kg_storage: Istanza del Knowledge Graph storage (Neo4jKnowledgeGraph o InMemoryKnowledgeGraph)
        """
        self.host = host
        self.port = port
        self.kg_storage = kg_storage
        self.neo4j_config = neo4j_config or {}
        self.app = FastAPI(
            title="MCP Server - Human Digital Twin",
            description="API per l'interazione autonoma con il modello LLM",
            version="1.0.0"
        )

        # Storage in-memory per dati IoT e contesto
        self.iot_data_store: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.external_data_store: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        # Setup delle routes
        self._setup_routes()

    def _setup_routes(self) -> None:
        """Configura le routes dell'API."""

        @self.app.get("/")
        async def root():
            """Endpoint di test."""
            return {"status": "ok", "message": "MCP Server is running"}

        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {"status": "healthy"}

        @self.app.post("/api/iot/data")
        async def receive_iot_data(data: IoTDataModel):
            """
            Riceve dati IoT dai dispositivi.

            Args:
                data: Dati IoT in formato JSON

            Returns:
                Conferma di ricezione con ID
            """
            # Salva i dati nello storage in-memory
            device_id = data.device_id
            data_dict = {
                "device_type": data.device_type,
                "device_id": device_id,
                "timestamp": data.timestamp,
                "data": data.data,
                "metadata": data.metadata
            }
            self.iot_data_store[device_id].append(data_dict)

            return {
                "status": "received",
                "device_id": device_id,
                "data_type": data.device_type,
                "timestamp": data.timestamp,
                "stored_count": len(self.iot_data_store[device_id])
            }

        @self.app.get("/api/iot/recent")
        async def get_recent_iot_data(
            device_id: str = Query(..., description="ID del dispositivo"),
            limit: int = Query(10, description="Numero massimo di record da restituire")
        ):
            """
            Recupera i dati IoT più recenti di un dispositivo.

            Args:
                device_id: ID del dispositivo
                limit: Numero massimo di record

            Returns:
                Lista di dati IoT recenti
            """
            if device_id not in self.iot_data_store:
                return {
                    "device_id": device_id,
                    "data": [],
                    "message": "Nessun dato disponibile per questo dispositivo"
                }

            # Recupera gli ultimi N record
            recent_data = self.iot_data_store[device_id][-limit:]

            return {
                "device_id": device_id,
                "count": len(recent_data),
                "data": recent_data
            }

        @self.app.get("/api/iot/stats")
        async def get_iot_stats(
            device_id: str = Query(..., description="ID del dispositivo")
        ):
            """
            Calcola statistiche aggregate sui dati IoT di un dispositivo.

            Args:
                device_id: ID del dispositivo

            Returns:
                Statistiche aggregate
            """
            if device_id not in self.iot_data_store:
                raise HTTPException(404, f"Dispositivo {device_id} non trovato")

            data_list = self.iot_data_store[device_id]

            if not data_list:
                return {"device_id": device_id, "stats": {}, "message": "Nessun dato"}

            # Calcola statistiche base
            stats = {
                "total_records": len(data_list),
                "first_timestamp": data_list[0]["timestamp"],
                "last_timestamp": data_list[-1]["timestamp"],
                "device_type": data_list[0]["device_type"]
            }

            # Calcola medie per campi numerici
            numeric_fields = {}
            for record in data_list:
                for key, value in record["data"].items():
                    if isinstance(value, (int, float)):
                        if key not in numeric_fields:
                            numeric_fields[key] = []
                        numeric_fields[key].append(value)

            averages = {}
            for field, values in numeric_fields.items():
                averages[field] = {
                    "avg": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "count": len(values)
                }

            stats["metrics"] = averages

            return {
                "device_id": device_id,
                "stats": stats
            }

        @self.app.post("/api/external/gmail")
        async def receive_gmail_data(data: ExternalDataModel):
            """
            Riceve dati da Gmail o altri servizi esterni.

            Args:
                data: Dati esterni in formato JSON

            Returns:
                Conferma di ricezione
            """
            # Salva i dati esterni
            source = data.source
            data_dict = {
                "source": source,
                "data_id": data.data_id,
                "timestamp": data.timestamp,
                "content": data.content,
                "metadata": data.metadata
            }
            self.external_data_store[source].append(data_dict)

            return {
                "status": "received",
                "source": source,
                "data_id": data.data_id,
                "stored_count": len(self.external_data_store[source])
            }

        @self.app.get("/api/context")
        async def get_context():
            """
            Returns a lightweight summary of the current context.
            Does NOT return actual data, only metadata and counts.
            Use specific query tools to retrieve actual data.

            Returns:
                Context summary from various sources
            """
            # Aggregate metadata only (no actual data)
            context = {
                "iot_devices": list(self.iot_data_store.keys()),
                "external_sources": list(self.external_data_store.keys()),
                "total_iot_records": sum(len(v) for v in self.iot_data_store.values()),
                "total_external_records": sum(len(v) for v in self.external_data_store.values())
            }

            # Add lightweight device summary (metadata only, no data)
            iot_summary = {}
            for device_id, data_list in self.iot_data_store.items():
                if data_list:
                    latest = data_list[-1]
                    iot_summary[device_id] = {
                        "device_type": latest["device_type"],
                        "last_update": latest["timestamp"],
                        "available_fields": list(latest["data"].keys()),  # Field names only
                        "record_count": len(data_list)
                    }

            context["iot_summary"] = iot_summary

            return {
                "context": context,
                "sources": ["iot", "external"],
                "note": "This is a summary only. Use get_data_schema, query_iot_field, or aggregate_iot_field for actual data."
            }

        @self.app.get("/api/devices")
        async def list_devices():
            """
            Elenca tutti i dispositivi IoT registrati.

            Returns:
                Lista di device_id con informazioni base
            """
            devices = []
            for device_id, data_list in self.iot_data_store.items():
                if data_list:
                    devices.append({
                        "device_id": device_id,
                        "device_type": data_list[0]["device_type"],
                        "record_count": len(data_list),
                        "last_update": data_list[-1]["timestamp"]
                    })

            return {
                "count": len(devices),
                "devices": devices
            }

        @self.app.get("/api/schema/{device_id}")
        async def get_data_schema(device_id: str):
            """
            Returns the data schema for a specific device.
            Shows available fields and their types without returning actual data.

            Args:
                device_id: ID of the device

            Returns:
                Schema information with field names and types
            """
            if device_id not in self.iot_data_store:
                raise HTTPException(404, f"Device {device_id} not found")

            data_list = self.iot_data_store[device_id]
            if not data_list:
                return {"device_id": device_id, "schema": {}, "message": "No data available"}

            # Extract schema from the latest record
            latest_record = data_list[-1]
            schema = {
                "device_type": latest_record["device_type"],
                "fields": {}
            }

            # Analyze fields from multiple records to get accurate types
            for record in data_list[-10:]:  # Check last 10 records
                for field_name, value in record["data"].items():
                    field_type = type(value).__name__
                    if field_name not in schema["fields"]:
                        schema["fields"][field_name] = {
                            "type": field_type,
                            "sample_value": value
                        }

            return {
                "device_id": device_id,
                "schema": schema,
                "total_records": len(data_list)
            }

        @self.app.get("/api/iot/field")
        async def query_iot_field(
            device_id: str = Query(..., description="Device ID"),
            field_name: str = Query(..., description="Field name to query"),
            limit: int = Query(10, description="Maximum number of records")
        ):
            """
            Query a specific field from IoT data, returning only that field.

            Args:
                device_id: ID of the device
                field_name: Name of the field to retrieve
                limit: Maximum number of records

            Returns:
                List of values for the specified field
            """
            if device_id not in self.iot_data_store:
                raise HTTPException(404, f"Device {device_id} not found")

            data_list = self.iot_data_store[device_id]
            if not data_list:
                return {"device_id": device_id, "field": field_name, "values": [], "message": "No data"}

            # Extract only the requested field
            values = []
            for record in data_list[-limit:]:
                if field_name in record["data"]:
                    values.append({
                        "timestamp": record["timestamp"],
                        "value": record["data"][field_name]
                    })

            return {
                "device_id": device_id,
                "field": field_name,
                "count": len(values),
                "values": values
            }

        @self.app.get("/api/iot/latest")
        async def get_latest_value(
            device_id: str = Query(..., description="Device ID"),
            field_name: str = Query(..., description="Field name")
        ):
            """
            Get the latest value for a specific field.

            Args:
                device_id: ID of the device
                field_name: Name of the field

            Returns:
                Latest value with timestamp
            """
            if device_id not in self.iot_data_store:
                raise HTTPException(404, f"Device {device_id} not found")

            data_list = self.iot_data_store[device_id]
            if not data_list:
                raise HTTPException(404, f"No data available for device {device_id}")

            # Get latest record
            latest_record = data_list[-1]
            if field_name not in latest_record["data"]:
                raise HTTPException(404, f"Field {field_name} not found in device data")

            return {
                "device_id": device_id,
                "field": field_name,
                "value": latest_record["data"][field_name],
                "timestamp": latest_record["timestamp"]
            }

        @self.app.get("/api/iot/aggregate")
        async def aggregate_iot_field(
            device_id: str = Query(..., description="Device ID"),
            field_name: str = Query(..., description="Field name to aggregate"),
            operation: str = Query(..., description="Operation: avg, min, max, sum, count")
        ):
            """
            Compute server-side aggregation on a field without returning raw data.

            Args:
                device_id: ID of the device
                field_name: Name of the field to aggregate
                operation: Aggregation operation (avg, min, max, sum, count)

            Returns:
                Aggregated value
            """
            if device_id not in self.iot_data_store:
                raise HTTPException(404, f"Device {device_id} not found")

            data_list = self.iot_data_store[device_id]
            if not data_list:
                raise HTTPException(404, f"No data available for device {device_id}")

            # Extract field values
            values = []
            for record in data_list:
                if field_name in record["data"]:
                    value = record["data"][field_name]
                    if isinstance(value, (int, float)):
                        values.append(value)

            if not values:
                raise HTTPException(404, f"No numeric values found for field {field_name}")

            # Compute aggregation
            result = None
            if operation == "avg":
                result = sum(values) / len(values)
            elif operation == "min":
                result = min(values)
            elif operation == "max":
                result = max(values)
            elif operation == "sum":
                result = sum(values)
            elif operation == "count":
                result = len(values)
            else:
                raise HTTPException(400, f"Invalid operation: {operation}. Use: avg, min, max, sum, count")

            return {
                "device_id": device_id,
                "field": field_name,
                "operation": operation,
                "result": result,
                "sample_size": len(values)
            }

        # ==================== KNOWLEDGE GRAPH ENDPOINTS ====================

        @self.app.get("/api/kg/profile")
        async def get_kg_profile():
            """
            Mostra quale profilo Person sta usando il server.

            Returns:
                Info sul profilo corrente
            """
            if not self.kg_storage:
                return {
                    "person_id": None,
                    "person_name": None,
                    "storage_type": "Neo4j",
                    "status": "no_active_profile"
                }

            if hasattr(self.kg_storage, 'person_id'):
                return {
                    "person_id": self.kg_storage.person_id,
                    "person_name": getattr(self.kg_storage, 'person_name', 'Unknown'),
                    "storage_type": "Neo4j" if hasattr(self.kg_storage, 'driver') else "InMemory"
                }
            else:
                return {
                    "person_id": None,
                    "person_name": None,
                    "storage_type": "InMemory"
                }
            
        @self.app.post("/api/kg/switch_profile")
        async def switch_kg_profile_endpoint(data: SwitchProfileModel):
            """
            Cambia il profilo KG attivo usato dal server MCP.
            """
            if not self.neo4j_config:
                raise HTTPException(500, "Neo4j config not available in MCP server")

            try:
                return self.switch_kg_profile(
                    person_id=data.person_id,
                    person_name=data.person_name
                )
            except Exception as e:
                raise HTTPException(500, f"Error switching profile: {str(e)}")

        @self.app.get("/api/kg/topics")
        async def get_kg_topics():
            """
            Recupera tutti i topic (broader e narrower) dal Knowledge Graph.

            Returns:
                Dict con {broader_topic: [narrower_topic1, ...]}
            """
            if not self.kg_storage:
                raise HTTPException(503, "Knowledge Graph storage not configured")

            try:
                topics = self.kg_storage.get_all_topics()
                return {
                    "topics": topics,
                    "count": {
                        "broader_topics": len(topics),
                        "narrower_topics": sum(len(narrowers) for narrowers in topics.values())
                    }
                }
            except Exception as e:
                raise HTTPException(500, f"Error retrieving topics: {str(e)}")

        @self.app.get("/api/kg/stats")
        async def get_kg_stats():
            """
            Recupera statistiche sul Knowledge Graph.

            Returns:
                Statistiche sul KG (numero di broader/narrower topics, triplette)
            """
            if not self.kg_storage:
                raise HTTPException(503, "Knowledge Graph storage not configured")

            try:
                stats = self.kg_storage.get_stats()
                return {"stats": stats}
            except Exception as e:
                raise HTTPException(500, f"Error retrieving stats: {str(e)}")

        @self.app.get("/api/kg/query/topic")
        async def query_by_topic(
            broader_topic: Optional[str] = Query(None, description="Broader topic da filtrare"),
            narrower_topic: Optional[str] = Query(None, description="Narrower topic da filtrare")
        ):
            """
            Interroga il Knowledge Graph per topic specifici.

            Args:
                broader_topic: Filtra per broader topic (opzionale)
                narrower_topic: Filtra per narrower topic (opzionale)

            Returns:
                Relazioni e informazioni filtrate per topic
            """
            if not self.kg_storage:
                raise HTTPException(503, "Knowledge Graph storage not configured")

            try:
                # Se Neo4j, fai query Cypher
                if hasattr(self.kg_storage, 'driver'):
                    return self._query_neo4j_by_topic(broader_topic, narrower_topic)
                else:
                    # InMemory storage
                    return self._query_inmemory_by_topic(broader_topic, narrower_topic)
            except Exception as e:
                raise HTTPException(500, f"Error querying by topic: {str(e)}")

        @self.app.get("/api/kg/query/entity")
        async def query_by_entity(
            entity_name: str = Query(..., description="Nome dell'entità da cercare"),
            relationship_type: Optional[str] = Query(None, description="Tipo di relazione (opzionale)")
        ):
            """
            Cerca informazioni su un'entità specifica nel Knowledge Graph.

            Args:
                entity_name: Nome dell'entità (es. "Mario", "Milano", "Pizza")
                relationship_type: Tipo di relazione opzionale per filtrare

            Returns:
                Tutte le relazioni che coinvolgono l'entità
            """
            if not self.kg_storage:
                raise HTTPException(503, "Knowledge Graph storage not configured")

            try:
                # Se Neo4j, fai query Cypher
                if hasattr(self.kg_storage, 'driver'):
                    return self._query_neo4j_by_entity(entity_name, relationship_type)
                else:
                    raise HTTPException(501, "Entity search not implemented for in-memory storage")
            except Exception as e:
                raise HTTPException(500, f"Error querying by entity: {str(e)}")

        @self.app.get("/api/kg/search")
        async def search_kg(
            query: str = Query(..., description="Testo libero da cercare nel KG"),
            limit: int = Query(10, description="Numero massimo di risultati")
        ):
            """
            Ricerca full-text nel Knowledge Graph.

            Args:
                query: Testo da cercare
                limit: Numero massimo di risultati

            Returns:
                Risultati della ricerca con rilevanza
            """
            if not self.kg_storage:
                raise HTTPException(503, "Knowledge Graph storage not configured")

            try:
                # Se Neo4j, fai query con CONTAINS
                if hasattr(self.kg_storage, 'driver'):
                    return self._search_neo4j(query, limit)
                else:
                    raise HTTPException(501, "Search not implemented for in-memory storage")
            except Exception as e:
                raise HTTPException(500, f"Error searching KG: {str(e)}")

    def _sanitize_rel_type(self, rel_type: str) -> str:
        """
        Neo4j relationship types must match: [A-Za-z_][A-Za-z0-9_]*
        We'll convert spaces/dashes to underscores and drop invalid chars.
        """
        if not rel_type:
            return ""

        # Trim + replace spaces/dashes with underscore
        cleaned = rel_type.strip()
        cleaned = re.sub(r"[\s\-]+", "_", cleaned)

        # Remove any remaining invalid chars
        cleaned = re.sub(r"[^A-Za-z0-9_]", "", cleaned)

        # Relationship type cannot start with a digit
        if cleaned and cleaned[0].isdigit():
            cleaned = "_" + cleaned

        return cleaned
    
    def switch_kg_profile(self, person_id: str, person_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Cambia il profilo Neo4j attivo usato dal server MCP.
        """
        if not self.neo4j_config:
            raise ValueError("Neo4j config not available in MCP server")

        from src.agents.knowledge_graph_builder import Neo4jKnowledgeGraph

        new_storage = Neo4jKnowledgeGraph(
            uri=self.neo4j_config["uri"],
            username=self.neo4j_config["username"],
            password=self.neo4j_config["password"],
            database=self.neo4j_config["database"],
            person_id=person_id,
            person_name=person_name or person_id
        )

        try:
            if self.kg_storage and hasattr(self.kg_storage, "close"):
                self.kg_storage.close()
        except Exception:
            pass

        self.kg_storage = new_storage

        return {
            "status": "ok",
            "person_id": self.kg_storage.person_id,
            "person_name": self.kg_storage.person_name,
            "storage_type": "Neo4j" if hasattr(self.kg_storage, "driver") else "InMemory"
        }

    def _query_neo4j_by_topic(self, broader_topic: Optional[str], narrower_topic: Optional[str]) -> Dict[str, Any]:
        """Query Neo4j per topic."""
        with self.kg_storage.driver.session(database=self.kg_storage.database) as session:
            if broader_topic and narrower_topic:
                # Filtra per entrambi
                result = session.run("""
                    MATCH (person:Person {id: $person_id})-[:KNOWS]->(subj)-[r]->(obj)
                    WHERE r.person_id = $person_id
                      AND r.broader_topic = $broader
                      AND r.narrower_topic = $narrower
                    RETURN
                        labels(subj)[0] AS subject_type,
                        subj.name AS subject,
                        type(r) AS predicate,
                        labels(obj)[0] AS object_type,
                        obj.name AS object,
                        r.broader_topic AS broader_topic,
                        r.narrower_topic AS narrower_topic,
                        r.reasoning AS reasoning
                    LIMIT 50
                """, person_id=self.kg_storage.person_id, broader=broader_topic, narrower=narrower_topic)
            elif broader_topic:
                # Solo broader
                result = session.run("""
                    MATCH (person:Person {id: $person_id})-[:KNOWS]->(subj)-[r]->(obj)
                    WHERE r.person_id = $person_id AND r.broader_topic = $broader
                    RETURN
                        labels(subj)[0] AS subject_type,
                        subj.name AS subject,
                        type(r) AS predicate,
                        labels(obj)[0] AS object_type,
                        obj.name AS object,
                        r.broader_topic AS broader_topic,
                        r.narrower_topic AS narrower_topic,
                        r.reasoning AS reasoning
                    LIMIT 50
                """, person_id=self.kg_storage.person_id, broader=broader_topic)
            else:
                # Tutti i topic
                result = session.run("""
                    MATCH (person:Person {id: $person_id})-[:KNOWS]->(subj)-[r]->(obj)
                    WHERE r.person_id = $person_id
                    RETURN
                        labels(subj)[0] AS subject_type,
                        subj.name AS subject,
                        type(r) AS predicate,
                        labels(obj)[0] AS object_type,
                        obj.name AS object,
                        r.broader_topic AS broader_topic,
                        r.narrower_topic AS narrower_topic,
                        r.reasoning AS reasoning
                    LIMIT 50
                """, person_id=self.kg_storage.person_id)

            relationships = [dict(record) for record in result]
            return {
                "count": len(relationships),
                "relationships": relationships,
                "filters": {
                    "broader_topic": broader_topic,
                    "narrower_topic": narrower_topic
                }
            }

    def _query_inmemory_by_topic(self, broader_topic: Optional[str], narrower_topic: Optional[str]) -> Dict[str, Any]:
        """Query in-memory storage per topic."""
        all_topics = self.kg_storage.get_all_topics()

        if broader_topic and narrower_topic:
            if broader_topic in all_topics and narrower_topic in all_topics[broader_topic]:
                return {
                    "broader_topic": broader_topic,
                    "narrower_topics": [narrower_topic]
                }
        elif broader_topic:
            if broader_topic in all_topics:
                return {
                    "broader_topic": broader_topic,
                    "narrower_topics": all_topics[broader_topic]
                }
        else:
            return {"topics": all_topics}

        return {"message": "No matching topics found"}

    def _query_neo4j_by_entity(self, entity_name: str, relationship_type: Optional[str]) -> Dict[str, Any]:
        """Query Neo4j per entità."""
        with self.kg_storage.driver.session(database=self.kg_storage.database) as session:
            if relationship_type:
                safe_rel = self._sanitize_rel_type(relationship_type)

                # Se dopo la sanitizzazione è vuoto, ignora il filtro (fallback)
                if not safe_rel:
                    relationship_type = None
                else:
                    relationship_type = safe_rel

                
            if relationship_type:
                query = f"""
                    MATCH (person:Person {{id: $person_id}})-[:KNOWS]->(subj)-[r:{relationship_type}]->(obj)
                    WHERE r.person_id = $person_id
                    AND (
                        (subj.name IS NOT NULL AND toLower(subj.name) CONTAINS toLower($entity))
                        OR (subj.id IS NOT NULL AND toLower(subj.id) CONTAINS toLower($entity))
                        OR (obj.name IS NOT NULL AND toLower(obj.name) CONTAINS toLower($entity))
                        OR (obj.id IS NOT NULL AND toLower(obj.id) CONTAINS toLower($entity))
                    )
                    RETURN
                        labels(subj)[0] AS subject_type,
                        coalesce(subj.name, subj.id) AS subject,
                        type(r) AS predicate,
                        labels(obj)[0] AS object_type,
                        coalesce(obj.name, obj.id) AS object,
                        r.broader_topic AS broader_topic,
                        r.narrower_topic AS narrower_topic,
                        r.reasoning AS reasoning
                    LIMIT 20
                """
            else:
                query = """
                    MATCH (person:Person {id: $person_id})-[:KNOWS]->(subj)-[r]->(obj)
                    WHERE r.person_id = $person_id
                    AND (
                        (subj.name IS NOT NULL AND toLower(subj.name) CONTAINS toLower($entity))
                        OR (subj.id IS NOT NULL AND toLower(subj.id) CONTAINS toLower($entity))
                        OR (obj.name IS NOT NULL AND toLower(obj.name) CONTAINS toLower($entity))
                        OR (obj.id IS NOT NULL AND toLower(obj.id) CONTAINS toLower($entity))
                    )
                    RETURN
                        labels(subj)[0] AS subject_type,
                        coalesce(subj.name, subj.id) AS subject,
                        type(r) AS predicate,
                        labels(obj)[0] AS object_type,
                        coalesce(obj.name, obj.id) AS object,
                        r.broader_topic AS broader_topic,
                        r.narrower_topic AS narrower_topic,
                        r.reasoning AS reasoning
                    LIMIT 20
                """

            result = session.run(query, person_id=self.kg_storage.person_id, entity=entity_name)
            relationships = [dict(record) for record in result]

            return {
                "entity": entity_name,
                "count": len(relationships),
                "relationships": relationships
            }

    def _search_neo4j(self, search_query: str, limit: int) -> Dict[str, Any]:
        """Ricerca full-text in Neo4j."""
        with self.kg_storage.driver.session(database=self.kg_storage.database) as session:
            result = session.run("""
                // Caso 1: il profilo Person stesso è il subject della relazione
                MATCH (person:Person {id: $person_id})-[r]->(obj)
                WHERE r.person_id = $person_id
                AND type(r) <> 'KNOWS'
                AND (
                    toLower(coalesce(person.name, "")) CONTAINS toLower($search_query)
                    OR toLower(coalesce(obj.name, "")) CONTAINS toLower($search_query)
                    OR toLower(coalesce(r.broader_topic, "")) CONTAINS toLower($search_query)
                    OR toLower(coalesce(r.narrower_topic, "")) CONTAINS toLower($search_query)
                )
                RETURN
                    'Person' AS subject_type,
                    person.name AS subject,
                    type(r) AS predicate,
                    labels(obj)[0] AS object_type,
                    obj.name AS object,
                    r.broader_topic AS broader_topic,
                    r.narrower_topic AS narrower_topic,
                    r.reasoning AS reasoning

                UNION

                // Caso 2: un nodo conosciuto è il subject
                MATCH (person:Person {id: $person_id})-[:KNOWS]->(subj)-[r]->(obj)
                WHERE r.person_id = $person_id
                AND (
                    toLower(coalesce(subj.name, "")) CONTAINS toLower($search_query)
                    OR toLower(coalesce(obj.name, "")) CONTAINS toLower($search_query)
                    OR toLower(coalesce(r.broader_topic, "")) CONTAINS toLower($search_query)
                    OR toLower(coalesce(r.narrower_topic, "")) CONTAINS toLower($search_query)
                )
                RETURN
                    labels(subj)[0] AS subject_type,
                    subj.name AS subject,
                    type(r) AS predicate,
                    labels(obj)[0] AS object_type,
                    obj.name AS object,
                    r.broader_topic AS broader_topic,
                    r.narrower_topic AS narrower_topic,
                    r.reasoning AS reasoning

                LIMIT $limit
            """, person_id=self.kg_storage.person_id, search_query=search_query, limit=int(limit))

            relationships = [dict(record) for record in result]

            return {
                "query": search_query,
                "count": len(relationships),
                "relationships": relationships
            }

    def run(self, debug: bool = False) -> None:
        """
        Avvia il server MCP.

        Args:
            debug: Se True, abilita il modo debug
        """
        uvicorn.run(
            self.app,
            host=self.host,
            port=self.port,
            log_level="debug" if debug else "info"
        )

    def get_app(self) -> FastAPI:
        """
        Restituisce l'app FastAPI per testing o deployment.

        Returns:
            L'istanza FastAPI
        """
        return self.app


# Modelli Pydantic per la validazione dei dati

class IoTDataModel(BaseModel):
    """Modello per i dati IoT."""
    device_type: str
    device_id: str
    timestamp: str
    data: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None


class ExternalDataModel(BaseModel):
    """Modello per i dati da servizi esterni."""
    source: str  # gmail, calendar, etc.
    data_id: str
    timestamp: str
    content: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None


class SwitchProfileModel(BaseModel):
    person_id: str
    person_name: Optional[str] = None
