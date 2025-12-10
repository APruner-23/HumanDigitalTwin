"""
Script per avviare l'MCP Server con Knowledge Graph integrato.

Questo script:
1. Inizializza il Knowledge Graph storage (Neo4j o InMemory)
2. Avvia l'MCP server con il KG integrato
3. Espone API per IoT + Knowledge Graph

Usage:
    python start_mcp_server.py
"""

import sys
from pathlib import Path
import argparse

# Setup path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.config import ConfigManager
from src.agents.knowledge_graph_builder import Neo4jKnowledgeGraph, InMemoryKnowledgeGraph
from src.mcp.server import MCPServer


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Avvia MCP Server con Knowledge Graph")
    parser.add_argument("--host", default="localhost", help="Host del server")
    parser.add_argument("--port", type=int, default=8000, help="Porta del server")
    parser.add_argument("--storage", choices=["neo4j", "memory"], default="neo4j",
                       help="Tipo di storage per il KG")
    parser.add_argument("--person-id", default="main_person", help="ID del profilo Person")
    parser.add_argument("--person-name", default="User", help="Nome del profilo Person")

    args = parser.parse_args()

    print("\n" + "="*60)
    print("🚀 MCP SERVER + KNOWLEDGE GRAPH")
    print("="*60 + "\n")

    # Carica configurazione
    config = ConfigManager()

    # Inizializza Knowledge Graph storage
    kg_storage = None

    if args.storage == "neo4j":
        print("🔗 Inizializzazione Neo4j Knowledge Graph...")

        try:
            neo4j_config = config.get_neo4j_config()

            kg_storage = Neo4jKnowledgeGraph(
                uri=neo4j_config['uri'],
                username=neo4j_config['username'],
                password=neo4j_config['password'],
                database=neo4j_config.get('database', 'neo4j'),
                person_id=args.person_id,
                person_name=args.person_name
            )

            stats = kg_storage.get_stats()
            print(f"✅ Neo4j connesso: {neo4j_config['uri']}")
            print(f"   Profilo: {args.person_name} (ID: {args.person_id})")
            print(f"   KG Stats: {stats}\n")

        except Exception as e:
            print(f"❌ Errore connessione Neo4j: {e}")
            print("   Fallback a storage in-memory...\n")
            kg_storage = InMemoryKnowledgeGraph()
    else:
        print("💾 Inizializzazione InMemory Knowledge Graph...")
        kg_storage = InMemoryKnowledgeGraph()
        print("✅ Storage in-memory creato\n")

    # Avvia MCP server
    print(f"🌐 Avvio MCP Server su http://{args.host}:{args.port}")
    print(f"   API IoT: /api/iot/*")
    print(f"   API KG:  /api/kg/*")
    print(f"   Docs:    http://{args.host}:{args.port}/docs")
    print("\n" + "="*60)
    print("✅ Server in esecuzione. Premi Ctrl+C per terminare.")
    print("="*60 + "\n")

    server = MCPServer(host=args.host, port=args.port, kg_storage=kg_storage)

    try:
        server.run(debug=False)
    except KeyboardInterrupt:
        print("\n\n👋 Arresto del server...")
    finally:
        if kg_storage and hasattr(kg_storage, 'close'):
            print("🔒 Chiusura connessione KG...")
            kg_storage.close()


if __name__ == "__main__":
    main()
