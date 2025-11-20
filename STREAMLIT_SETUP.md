# Come Usare l'Integrazione KG con Streamlit

Guida rapida per usare l'MCP Server + Knowledge Graph con l'app Streamlit.

## 🚀 Quick Start

### 1. Avvia l'MCP Server (Terminale 1)

```bash
# Con Neo4j (raccomandato - richiede Neo4j in esecuzione)
python start_mcp_server.py --storage neo4j --person-name "Mario Rossi"

# Oppure con storage in-memory (per test rapidi)
python start_mcp_server.py --storage memory
```

**Output atteso:**
```
============================================================
🚀 MCP SERVER + KNOWLEDGE GRAPH
============================================================

🔗 Inizializzazione Neo4j Knowledge Graph...
✅ Neo4j connesso: bolt://localhost:7687
   Profilo: Mario Rossi (ID: main_person)
   KG Stats: {'num_broader_topics': 0, 'num_narrower_topics': 0, 'num_triplets': 0}

🌐 Avvio MCP Server su http://localhost:8000
   API IoT: /api/iot/*
   API KG:  /api/kg/*
   Docs:    http://localhost:8000/docs

============================================================
✅ Server in esecuzione. Premi Ctrl+C per terminare.
============================================================
```

### 2. Verifica che il Server Funziona

Apri nel browser: http://localhost:8000/docs

Dovresti vedere la documentazione FastAPI con tutti gli endpoint IoT e KG.

### 3. Avvia Streamlit (Terminale 2)

```bash
streamlit run app.py
```

L'app si aprirà automaticamente su http://localhost:8501

### 4. Workflow Completo

#### Step 1: Estrai Triplette (Tab 1)

1. Vai alla **Tab "Estrazione Triplette"**
2. Inserisci del testo, esempio:
   ```
   Mario vive a Milano e lavora come Software Engineer.
   Gli piace molto la pizza e va spesso al parco.
   Il suo migliore amico si chiama Luca.
   ```
3. Clicca **"Estrai Triplette con LangGraph"**
4. Le triplette vengono estratte e salvate in session_state

#### Step 2: Valida con Ontologia (Tab 2) [Opzionale]

1. Vai alla **Tab "Ontology Validation"**
2. Seleziona "Sessione corrente (in memoria)"
3. Clicca **"Valida Triplette con Schema.org"**
4. Le triplette vengono validate con Schema.org

#### Step 3: Costruisci il Knowledge Graph (Tab 3)

1. Vai alla **Tab "Knowledge Graph Builder"**
2. Seleziona la sorgente:
   - "Sessione corrente (dopo estrazione)" per triplette non validate
   - "Sessione corrente (dopo validazione)" per triplette validate
3. Clicca **"Costruisci Knowledge Graph"**
4. Il sistema:
   - Classifica ogni tripletta in broader/narrower topics con LLM
   - Fa semantic matching con topics esistenti
   - Salva tutto nel Neo4j (o InMemory)
5. Visualizza il grafo interattivo!

#### Step 4: Chatta con l'Agent (Tab 6)

1. Vai alla **Tab "Chat Agent con Accesso MCP"**
2. L'agent ha accesso a **TUTTI i tools**, inclusi quelli del KG:
   - `get_kg_topics()` - Scopri cosa sa l'AI
   - `query_kg_by_topic()` - Cerca per categoria
   - `query_kg_by_entity()` - Cerca per entità
   - `search_kg()` - Ricerca libera

3. Prova queste domande:

```
📝 Esempi di Chat:

"Cosa sai su di me?"
→ L'agent chiama get_kg_stats() e get_kg_topics()

"Quali sono i miei interessi?"
→ L'agent chiama query_kg_by_topic(broader_topic="Lifestyle")

"Dove vivo?"
→ L'agent chiama query_kg_by_entity(entity_name="Milano")
   o search_kg(query="Milano")

"Chi sono i miei amici?"
→ L'agent chiama query_kg_by_topic(broader_topic="Social", narrower_topic="Friends")

"Cerca informazioni su Pizza"
→ L'agent chiama search_kg(query="Pizza")
```

## 🛠️ Opzioni Avanzate

### Cambiare Profilo Person (Solo Neo4j)

Nella **Tab 3 (Knowledge Graph Builder)**, espandi la sezione **"Gestione Profili"**:

- **Crea nuovo profilo**: Crea un nuovo Person node nel grafo
- **Cambia profilo attivo**: Passa da un profilo all'altro
- **Elimina profilo**: Rimuovi un profilo (⚠️ cancella tutti i dati!)

Ogni profilo ha il suo KG separato.

### Argomenti CLI per start_mcp_server.py

```bash
python start_mcp_server.py --help

Opzioni:
  --host HOST              Host del server (default: localhost)
  --port PORT              Porta del server (default: 8000)
  --storage {neo4j,memory} Tipo di storage (default: neo4j)
  --person-id ID           ID del profilo Person (default: main_person)
  --person-name NAME       Nome del profilo (default: User)
```

### Esempio Multi-Profilo

```bash
# Profilo 1: Mario
python start_mcp_server.py --storage neo4j --person-id mario_rossi --person-name "Mario Rossi" --port 8000

# Profilo 2: Luca (in un altro terminale)
python start_mcp_server.py --storage neo4j --person-id luca_bianchi --person-name "Luca Bianchi" --port 8001
```

## 🔍 Debugging

### Verifica Endpoint KG

Test manuale via curl:

```bash
# Stats
curl http://localhost:8000/api/kg/stats

# Topics
curl http://localhost:8000/api/kg/topics

# Query by topic
curl "http://localhost:8000/api/kg/query/topic?broader_topic=Social"

# Query by entity
curl "http://localhost:8000/api/kg/query/entity?entity_name=Mario"

# Search
curl "http://localhost:8000/api/kg/search?query=Pizza&limit=5"
```

### Logs

I logs dell'MCP server appaiono nel terminale 1.
I logs di Streamlit appaiono nel terminale 2.

### Problemi Comuni

**Errore: "Knowledge Graph storage not configured"**
→ Il server MCP è stato avviato senza `--storage`. Riavvialo con `--storage neo4j` o `--storage memory`.

**Errore: "Impossibile connettersi al server MCP"**
→ Assicurati che `start_mcp_server.py` sia in esecuzione nel terminale 1.

**Errore Neo4j: "Connection refused"**
→ Neo4j non è in esecuzione. Avvialo con `neo4j start` (Docker o locale).

## 📊 Visualizzazione Grafo

Nella **Tab 3**, dopo aver costruito il KG, vedrai:

- **Statistiche**: Broader/Narrower topics, numero di triplette
- **Struttura testuale**: Tree view dei topic
- **Grafo interattivo Plotly**: Visualizzazione del grafo entità-relazioni
  - Nodi Person (blu)
  - Nodi entità (colorati per tipo)
  - Relazioni con predicato e narrower topic

Puoi zoomare, spostare e cliccare sui nodi!

## 🎯 Prossimi Step

Dopo aver testato l'integrazione base, possiamo discutere e implementare:
1. **LangGraph ReAct Agent** per retrieval intelligente
2. Strategie di query optimization
3. Caching e performance tuning
