import yaml
from pathlib import Path
from typing import Dict, Any, Optional


class PromptManager:
    """Manages system prompts from YAML files."""

    def __init__(self, prompts_path: str = None):
        """
        Initialize PromptManager.

        Args:
            prompts_path: Path to prompts.yaml. If None, uses default path.
        """
        if prompts_path is None:
            # Percorso default: stesso directory di questo file
            prompts_path = Path(__file__).parent / "prompts.yaml"

        self.prompts_path = Path(prompts_path)
        self._prompts: Dict[str, Any] = {}

        # Load prompts
        self._load_prompts()

    def _load_prompts(self) -> None:
        """Load prompts from YAML file."""
        if not self.prompts_path.exists():
            raise FileNotFoundError(f"File prompts non trovato: {self.prompts_path}")

        with open(self.prompts_path, 'r', encoding='utf-8') as f:
            self._prompts = yaml.safe_load(f)

    def get_prompt(self, prompt_name: str) -> Optional[Dict[str, str]]:
        """
        Retrieve a full prompt (system + user_template).

        Args:
            prompt_name: Prompt name

        Returns:
            Dictionary with 'system' and 'user_template' or None if not found
        """
        return self._prompts.get(prompt_name)

    def get_system_prompt(self, prompt_name: str) -> Optional[str]:
        """
        Retrieve only the system prompt.

        Args:
            prompt_name: Prompt name

        Returns:
            The system prompt or None if not found
        """
        prompt = self._prompts.get(prompt_name)
        return prompt.get('system') if prompt else None

    def get_user_template(self, prompt_name: str) -> Optional[str]:
        """
        Retrieve only the user prompt template.

        Args:
            prompt_name: Prompt name

        Returns:
            The user prompt template or None if not found
        """
        prompt = self._prompts.get(prompt_name)
        return prompt.get('user_template') if prompt else None

    def format_prompt(self, prompt_name: str, **kwargs) -> Optional[str]:
        """
        Format a prompt with provided parameters.

        Args:
            prompt_name: Prompt name
            **kwargs: Parameters to substitute in the template

        Returns:
            Formatted prompt or None if not found
        """
        template = self.get_user_template(prompt_name)
        if template:
            return template.format(**kwargs)
        return None

    def build_messages(self, prompt_name: str, **kwargs) -> list:
        """
        Build a list of messages for the LLM.

        Args:
            prompt_name: Prompt name
            **kwargs: Parameters to substitute in the template

        Returns:
            List of messages in format [{"role": "system/user", "content": "..."}]
        """
        prompt = self.get_prompt(prompt_name)
        if not prompt:
            return []

        messages = []

        # Aggiungi system prompt se presente
        if 'system' in prompt and prompt['system']:
            messages.append({
                'role': 'system',
                'content': prompt['system'].strip()
            })

        # Aggiungi user prompt formattato
        if 'user_template' in prompt:
            formatted = prompt['user_template'].format(**kwargs)
            messages.append({
                'role': 'user',
                'content': formatted.strip()
            })

        return messages

    def list_prompts(self) -> list:
        """
        Return list of all available prompts.

        Returns:
            List of prompt names
        """
        return list(self._prompts.keys())

    def reload(self) -> None:
        """Reload prompts from file."""
        self._load_prompts()
