"""
Test script per l'integrazione Knowledge Graph nell'MCP Server.

Questo script:
1. Crea un KG storage (Neo4j o InMemory)
2. Popola il KG con alcune triplette di esempio
3. Avvia l'MCP server con il KG integrato
4. Testa gli endpoint KG tramite richieste HTTP

Usage:
    python test_mcp_kg_integration.py
"""

import sys
from pathlib import Path
import time
import threading
import requests

# Setup path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.config import ConfigManager
from src.agents.knowledge_graph_builder import Neo4jKnowledgeGraph, InMemoryKnowledgeGraph
from src.mcp.server import MCPServer


def create_kg_with_sample_data(use_neo4j: bool = True):
    """
    Crea un KG storage e lo popola con dati di esempio.

    Args:
        use_neo4j: Se True usa Neo4j, altrimenti usa InMemory

    Returns:
        Istanza del KG storage
    """
    if use_neo4j:
        print("🔗 Creazione Neo4j Knowledge Graph...")
        config = ConfigManager()
        neo4j_config = config.get_neo4j_config()

        kg = Neo4jKnowledgeGraph(
            uri=neo4j_config['uri'],
            username=neo4j_config['username'],
            password=neo4j_config['password'],
            database=neo4j_config.get('database', 'neo4j'),
            person_id="test_user",
            person_name="Test User"
        )
    else:
        print("💾 Creazione InMemory Knowledge Graph...")
        kg = InMemoryKnowledgeGraph()

    # Popola con dati di esempio
    print("📝 Popolamento KG con dati di esempio...")

    sample_triplets = [
        {
            "subject": {"value": "Mario", "type": "Person"},
            "predicate": {"value": "lives_in", "type": "Relation"},
            "object": {"value": "Milano", "type": "Place"},
            "classification_reasoning": "Geographic information about residence"
        },
        {
            "subject": {"value": "Mario", "type": "Person"},
            "predicate": {"value": "works_as", "type": "Relation"},
            "object": {"value": "Software Engineer", "type": "Occupation"},
            "classification_reasoning": "Professional occupation"
        },
        {
            "subject": {"value": "Mario", "type": "Person"},
            "predicate": {"value": "likes", "type": "Relation"},
            "object": {"value": "Pizza", "type": "Food"},
            "classification_reasoning": "Food preference"
        },
        {
            "subject": {"value": "Mario", "type": "Person"},
            "predicate": {"value": "friend_of", "type": "Relation"},
            "object": {"value": "Luca", "type": "Person"},
            "classification_reasoning": "Social relationship"
        },
    ]

    topics = [
        ("Personal", "Location"),
        ("Work", "Occupation"),
        ("Lifestyle", "Food Preferences"),
        ("Social", "Friends"),
    ]

    for triplet, (broader, narrower) in zip(sample_triplets, topics):
        kg.add_triplet(triplet, broader, narrower)
        print(f"  ✓ Added: {triplet['subject']['value']} → {triplet['predicate']['value']} → {triplet['object']['value']}")

    stats = kg.get_stats()
    print(f"\n📊 KG Stats: {stats}\n")

    return kg


def start_mcp_server_thread(kg_storage, port=8000):
    """
    Avvia l'MCP server in un thread separato.

    Args:
        kg_storage: Istanza del KG storage
        port: Porta su cui avviare il server
    """
    server = MCPServer(host="localhost", port=port, kg_storage=kg_storage)

    def run_server():
        print(f"🚀 Avvio MCP Server su http://localhost:{port}...")
        server.run(debug=False)

    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()

    # Wait per il server
    print("⏳ Attendo che il server si avvii...")
    time.sleep(3)

    return server


def test_kg_endpoints(base_url="http://localhost:8000"):
    """
    Testa gli endpoint del Knowledge Graph.

    Args:
        base_url: URL base del server MCP
    """
    print("\n" + "="*60)
    print("🧪 TEST DEGLI ENDPOINT KNOWLEDGE GRAPH")
    print("="*60 + "\n")

    # Test 1: Get KG Stats
    print("1️⃣ Test: GET /api/kg/stats")
    try:
        response = requests.get(f"{base_url}/api/kg/stats")
        response.raise_for_status()
        print(f"   ✓ Status: {response.status_code}")
        print(f"   Response: {response.json()}\n")
    except Exception as e:
        print(f"   ✗ Error: {e}\n")

    # Test 2: Get KG Topics
    print("2️⃣ Test: GET /api/kg/topics")
    try:
        response = requests.get(f"{base_url}/api/kg/topics")
        response.raise_for_status()
        print(f"   ✓ Status: {response.status_code}")
        data = response.json()
        print(f"   Topics: {data['topics']}")
        print(f"   Count: {data['count']}\n")
    except Exception as e:
        print(f"   ✗ Error: {e}\n")

    # Test 3: Query by Topic
    print("3️⃣ Test: GET /api/kg/query/topic?broader_topic=Social")
    try:
        response = requests.get(f"{base_url}/api/kg/query/topic", params={"broader_topic": "Social"})
        response.raise_for_status()
        print(f"   ✓ Status: {response.status_code}")
        data = response.json()
        print(f"   Found {data.get('count', 0)} relationships")
        if data.get('relationships'):
            for rel in data['relationships'][:3]:  # Mostra solo i primi 3
                print(f"     - {rel['subject']} → {rel['predicate']} → {rel['object']}")
        print()
    except Exception as e:
        print(f"   ✗ Error: {e}\n")

    # Test 4: Query by Entity
    print("4️⃣ Test: GET /api/kg/query/entity?entity_name=Mario")
    try:
        response = requests.get(f"{base_url}/api/kg/query/entity", params={"entity_name": "Mario"})
        response.raise_for_status()
        print(f"   ✓ Status: {response.status_code}")
        data = response.json()
        print(f"   Entity: {data.get('entity')}")
        print(f"   Found {data.get('count', 0)} relationships")
        if data.get('relationships'):
            for rel in data['relationships']:
                print(f"     - {rel['subject']} → {rel['predicate']} → {rel['object']} [{rel['narrower_topic']}]")
        print()
    except Exception as e:
        print(f"   ✗ Error: {e}\n")

    # Test 5: Search KG
    print("5️⃣ Test: GET /api/kg/search?query=Pizza")
    try:
        response = requests.get(f"{base_url}/api/kg/search", params={"query": "Pizza", "limit": 5})
        response.raise_for_status()
        print(f"   ✓ Status: {response.status_code}")
        data = response.json()
        print(f"   Query: {data.get('query')}")
        print(f"   Found {data.get('count', 0)} results")
        if data.get('relationships'):
            for rel in data['relationships']:
                print(f"     - {rel['subject']} → {rel['predicate']} → {rel['object']}")
        print()
    except Exception as e:
        print(f"   ✗ Error: {e}\n")

    print("="*60)
    print("✅ Test completati!")
    print("="*60 + "\n")


def main():
    """Main function."""
    print("\n" + "="*60)
    print("🧪 TEST INTEGRAZIONE MCP + KNOWLEDGE GRAPH")
    print("="*60 + "\n")

    # Ask user quale storage usare
    print("Scegli il tipo di storage:")
    print("1. Neo4j (richiede Neo4j in esecuzione)")
    print("2. InMemory (per test rapidi)")
    choice = input("\nScelta (1/2) [default=2]: ").strip() or "2"

    use_neo4j = choice == "1"

    try:
        # Crea KG con dati di esempio
        kg = create_kg_with_sample_data(use_neo4j=use_neo4j)

        # Avvia MCP server
        server = start_mcp_server_thread(kg, port=8000)

        # Testa gli endpoint
        test_kg_endpoints()

        print("\n💡 Il server continua a girare. Premi Ctrl+C per terminare.")
        print("   Puoi testare manualmente gli endpoint a http://localhost:8000/docs\n")

        # Keep alive
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n\n👋 Arresto del server...")

    except Exception as e:
        print(f"\n❌ Errore: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if use_neo4j and 'kg' in locals():
            print("🔒 Chiusura connessione Neo4j...")
            kg.close()


if __name__ == "__main__":
    main()
