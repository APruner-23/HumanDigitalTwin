import yaml
import os
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv


class ConfigManager:
    """Manages project configuration from YAML files and environment variables."""

    def __init__(self, config_path: str = None):
        """
        Initialize ConfigManager.

        Args:
            config_path: Path to config.yaml. If None, uses default path.
        """
        if config_path is None:
            # Percorso default: root del progetto
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "config.yaml"

        self.config_path = Path(config_path)
        self._config: Dict[str, Any] = {}

        # Load environment variables
        load_dotenv(override=True)

        # Load configuration
        self._load_config()

    def _load_config(self) -> None:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"File di configurazione non trovato: {self.config_path}")

        with open(self.config_path, 'r', encoding='utf-8') as f:
            self._config = yaml.safe_load(f)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Retrieve a configuration value using dot notation.

        Args:
            key: Key in 'section.subsection.key' format
            default: Default value if key does not exist

        Returns:
            Configuration value or default
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
        Retrieve an environment variable.

        Args:
            key: Name of the environment variable
            default: Default value if it does not exist

        Returns:
            Value of the environment variable or default
        """
        return os.getenv(key, default)

    def get_llm_config(self) -> Dict[str, Any]:
        """Retrieve full LLM configuration."""
        return self._config.get('llm', {})

    def get_mcp_config(self) -> Dict[str, Any]:
        """Retrieve MCP server configuration."""
        return self._config.get('mcp_server', {})

    def get_streamlit_config(self) -> Dict[str, Any]:
        """Retrieve Streamlit configuration."""
        return self._config.get('streamlit', {})

    def get_ontology_config(self) -> Dict[str, Any]:
        """Retrieve Ontology configuration."""
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
        """Reload configuration from file."""
        self._load_config()
