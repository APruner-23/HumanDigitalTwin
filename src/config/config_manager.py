import yaml
import os
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv


class ConfigManager:
    """Gestisce la configurazione del progetto da file YAML e variabili d'ambiente."""

    def __init__(self, config_path: str = None):
        """
        Inizializza il ConfigManager.

        Args:
            config_path: Percorso al file config.yaml. Se None, usa il path di default.
        """
        if config_path is None:
            # Percorso default: root del progetto
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "config.yaml"

        self.config_path = Path(config_path)
        self._config: Dict[str, Any] = {}

        # Carica le variabili d'ambiente
        load_dotenv(override=True)

        # Carica la configurazione
        self._load_config()

    def _load_config(self) -> None:
        """Carica la configurazione dal file YAML."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"File di configurazione non trovato: {self.config_path}")

        with open(self.config_path, 'r', encoding='utf-8') as f:
            self._config = yaml.safe_load(f)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Recupera un valore di configurazione usando la notazione dot.

        Args:
            key: Chiave in formato 'section.subsection.key'
            default: Valore di default se la chiave non esiste

        Returns:
            Il valore della configurazione o il default
        """
        keys = key.split('.')
        value = self._config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def get_env(self, key: str, default: str = None) -> str:
        """
        Recupera una variabile d'ambiente.

        Args:
            key: Nome della variabile d'ambiente
            default: Valore di default se non esiste

        Returns:
            Il valore della variabile d'ambiente o il default
        """
        return os.getenv(key, default)

    def get_llm_config(self) -> Dict[str, Any]:
        """Recupera la configurazione completa dell'LLM."""
        return self._config.get('llm', {})

    def get_mcp_config(self) -> Dict[str, Any]:
        """Recupera la configurazione del server MCP."""
        return self._config.get('mcp_server', {})

    def get_streamlit_config(self) -> Dict[str, Any]:
        """Recupera la configurazione di Streamlit."""
        return self._config.get('streamlit', {})

    def get_ontology_config(self) -> Dict[str, Any]:
        """Recupera la configurazione dell'Ontology."""
        return self._config.get('ontology', {})

    def get_neo4j_config(self) -> Dict[str, Any]:
        """
        Recupera la configurazione di Neo4j.

        Returns:
            Dict con uri, username, password, database da env vars o config.yaml
        """
        # Priorità: env vars > config.yaml
        kg_config = self._config.get('knowledge_graph', {})
        neo4j_config = kg_config.get('neo4j', {})

        return {
            'uri': self.get_env('NEO4J_URI', neo4j_config.get('uri', 'bolt://localhost:7687')),
            'username': self.get_env('NEO4J_USERNAME', neo4j_config.get('username', 'neo4j')),
            'password': self.get_env('NEO4J_PASSWORD', neo4j_config.get('password', '')),
            'database': self.get_env('NEO4J_DATABASE', neo4j_config.get('database', 'neo4j'))
        }

    def reload(self) -> None:
        """Ricarica la configurazione dal file."""
        self._load_config()
