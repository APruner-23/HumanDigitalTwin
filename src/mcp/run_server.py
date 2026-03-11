"""Script per avviare il server MCP."""

import sys
from pathlib import Path

# Aggiungi il path del progetto al PYTHONPATH
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config import ConfigManager
from src.mcp import MCPServer

from src.agents.knowledge_graph_builder import Neo4jKnowledgeGraph

def main():
    """Avvia il server MCP."""
    # Carica la configurazione
    config = ConfigManager()
    mcp_config = config.get_mcp_config()

    host = mcp_config.get('host', 'localhost')
    port = mcp_config.get('port', 8000)

    print(f"Starting MCP Server on {host}:{port}...")

    # Configurazione di Neo4J per tool MCP
    neo4j_cfg = config.get_neo4j_config() 

    uri = neo4j_cfg["uri"]  # Es. "bolt://localhost:7687"
    database = neo4j_cfg["database"]
    username = neo4j_cfg["username"]
    password = neo4j_cfg["password"]

    if not username or not password:
        raise ValueError("Password o Username mancante per Neo4j. Controlla la configurazione.")
    
    kg_storage = None

    server = MCPServer(
        host=host,
        port=port,
        kg_storage=kg_storage,
        neo4j_config=neo4j_cfg
    )
    server.run(debug=True)


if __name__ == "__main__":
    main()
