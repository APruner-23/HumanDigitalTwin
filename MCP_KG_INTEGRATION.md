# Integrazione Knowledge Graph nell'MCP Server

Questa integrazione permette di esporre il Knowledge Graph tramite l'MCP Server, rendendo disponibili le conoscenze apprese dall'utente agli agenti LLM attraverso tools LangChain.

## 🎯 Obiettivo

Integrare il Knowledge Graph (Neo4j o InMemory) nell'MCP Server esistente per permettere query e retrieval delle conoscenze apprese dall'utente.

## 🏗️ Architettura

```
┌─────────────────────────────────────────────────────────┐
│                    LLM Agent                            │
│  (con tools LangChain per IoT e Knowledge Graph)        │
└────────────────┬────────────────────────────────────────┘
                 │
                 │ HTTP Requests
                 ▼
┌─────────────────────────────────────────────────────────┐
│                   MCP Server (FastAPI)                   │
│ ┌─────────────────────┐  ┌───────────────────────────┐ │
│ │   IoT Endpoints     │  │   KG Endpoints            │ │
│ │  /api/iot/*         │  │  /api/kg/*                │ │
│ └─────────────────────┘  └───────────────────────────┘ │
└──────────────┬──────────────────────┬───────────────────┘
               │                      │
               │                      ▼
               │           ┌──────────────────────┐
               │           │  Knowledge Graph     │
               │           │  Storage             │
               │           │  (Neo4j/InMemory)    │
               │           └──────────────────────┘
               ▼
    ┌──────────────────────┐
    │  IoT Data Store      │
    │  (In-Memory)         │
    └──────────────────────┘
```

## 📋 Nuovi Endpoint API

### 1. **GET /api/kg/topics**
Recupera tutti i topic (broader e narrower) dal Knowledge Graph.

**Response:**
```json
{
  "topics": {
    "Health": ["Heart Rate", "Sleep"],
    "Social": ["Friends", "Family"],
    "Work": ["Projects", "Meetings"]
  },
  "count": {
    "broader_topics": 3,
    "narrower_topics": 6
  }
}
```

### 2. **GET /api/kg/stats**
Statistiche sul Knowledge Graph.

**Response:**
```json
{
  "stats": {
    "num_broader_topics": 5,
    "num_narrower_topics": 15,
    "num_triplets": 42
  }
}
```

### 3. **GET /api/kg/query/topic**
Query per topic specifici.

**Query Parameters:**
- `broader_topic` (optional): Filtra per broader topic
- `narrower_topic` (optional): Filtra per narrower topic

**Response:**
```json
{
  "count": 2,
  "relationships": [
    {
      "subject_type": "Person",
      "subject": "Mario",
      "predicate": "friend_of",
      "object_type": "Person",
      "object": "Luca",
      "broader_topic": "Social",
      "narrower_topic": "Friends",
      "reasoning": "Social relationship"
    }
  ],
  "filters": {
    "broader_topic": "Social",
    "narrower_topic": null
  }
}
```

### 4. **GET /api/kg/query/entity**
Cerca informazioni su un'entità specifica.

**Query Parameters:**
- `entity_name` (required): Nome dell'entità
- `relationship_type` (optional): Tipo di relazione per filtrare

**Response:**
```json
{
  "entity": "Mario",
  "count": 4,
  "relationships": [
    {
      "subject": "Mario",
      "predicate": "lives_in",
      "object": "Milano",
      "broader_topic": "Personal",
      "narrower_topic": "Location"
    }
  ]
}
```

### 5. **GET /api/kg/search**
Ricerca full-text nel Knowledge Graph.

**Query Parameters:**
- `query` (required): Testo da cercare
- `limit` (optional, default=10): Numero massimo di risultati

**Response:**
```json
{
  "query": "Pizza",
  "count": 1,
  "relationships": [
    {
      "subject": "Mario",
      "predicate": "likes",
      "object": "Pizza",
      "broader_topic": "Lifestyle",
      "narrower_topic": "Food Preferences"
    }
  ]
}
```

## 🛠️ Tools LangChain

Nuovi tools disponibili per gli agenti:

### 1. `get_kg_topics()`
Recupera tutti i topic dal KG.

### 2. `get_kg_stats()`
Statistiche sul KG.

### 3. `query_kg_by_topic(broader_topic: str, narrower_topic: str)`
Query per topic specifici.

### 4. `query_kg_by_entity(entity_name: str, relationship_type: str)`
Cerca informazioni su un'entità.

### 5. `search_kg(query: str, limit: int)`
Ricerca full-text nel KG.

## 🚀 Utilizzo

### Setup Base

```python
from src.agents.knowledge_graph_builder import Neo4jKnowledgeGraph
from src.mcp.server import MCPServer
from src.config import ConfigManager

# 1. Crea il KG storage
config = ConfigManager()
neo4j_config = config.get_neo4j_config()

kg = Neo4jKnowledgeGraph(
    uri=neo4j_config['uri'],
    username=neo4j_config['username'],
    password=neo4j_config['password'],
    person_id="user_123",
    person_name="Mario Rossi"
)

# 2. Popola il KG (esempio)
triplet = {
    "subject": {"value": "Mario", "type": "Person"},
    "predicate": {"value": "lives_in", "type": "Relation"},
    "object": {"value": "Milano", "type": "Place"}
}
kg.add_triplet(triplet, "Personal", "Location")

# 3. Avvia MCP server con KG integrato
server = MCPServer(host="localhost", port=8000, kg_storage=kg)
server.run()
```

### Uso con Agente LLM

```python
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from src.mcp.mcp_tools import get_mcp_tools

# 1. Setup LLM
llm = ChatGroq(
    groq_api_key="your-api-key",
    model_name="llama-3.3-70b-versatile"
)

# 2. Ottieni i tools (include IoT + KG)
tools = get_mcp_tools(mcp_base_url="http://localhost:8000")

# 3. Crea agente ReAct
agent = create_react_agent(llm, tools)

# 4. Query l'agente
result = agent.invoke({
    "messages": [("user", "Cosa sai sugli interessi di Mario?")]
})

print(result["messages"][-1].content)
```

## 🧪 Testing

Usa lo script di test fornito:

```bash
python test_mcp_kg_integration.py
```

Lo script:
1. Crea un KG con dati di esempio
2. Avvia l'MCP server
3. Testa tutti gli endpoint KG
4. Lascia il server in esecuzione per test manuali

## 📝 Note Implementative

### Storage Supportati

1. **Neo4j** (consigliato per produzione)
   - Query Cypher efficienti
   - Supporto per entity search
   - Full-text search
   - Persistenza dati

2. **InMemory** (per test e sviluppo)
   - Veloce per piccoli dataset
   - Nessuna persistenza
   - Limitato per query complesse

### Prossimi Step (Non Implementati)

Come discusso, l'agentic framework (ReAct node) sarà implementato dopo:
1. Test dell'integrazione base
2. Discussione sull'architettura del retrieval
3. Design del prompt per il ReAct agent

## 🔗 File Modificati

- `src/mcp/server.py`: Aggiunto parametro `kg_storage` e 5 nuovi endpoint KG
- `src/mcp/mcp_tools.py`: Aggiunti 5 nuovi tools LangChain per KG
- `test_mcp_kg_integration.py`: Script di test standalone

## 📚 Esempi di Query

### Domande che l'agente può rispondere con i KG tools:

1. "Cosa sai su Mario?"
   → Usa `query_kg_by_entity(entity_name="Mario")`

2. "Quali sono gli interessi dell'utente?"
   → Usa `query_kg_by_topic(broader_topic="Lifestyle")`

3. "Cosa ha imparato il sistema sulla salute dell'utente?"
   → Usa `query_kg_by_topic(broader_topic="Health")`

4. "Cerca informazioni su Pizza"
   → Usa `search_kg(query="Pizza")`

5. "Quante conoscenze hai appreso?"
   → Usa `get_kg_stats()`

## ⚡ Performance

- **Neo4j**: Query ottimizzate con indici su `person_id`
- **Limit**: Query limitate a 20-50 risultati per evitare overhead
- **Caching**: MCP tools usano JSON serialization lightweight
- **Async**: FastAPI endpoints sono async-ready

## 🔐 Sicurezza

- Tutti gli endpoint verificano che `kg_storage` sia configurato
- Query Neo4j usano parametri prepared per evitare injection
- Person isolation: ogni query filtra per `person_id`
