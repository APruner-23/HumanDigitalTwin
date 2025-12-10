"""
Logging system with Rich to track LLM and MCP interactions.
"""

from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import json
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich import box


class AgentLogger:
    """Logger to track agent, LLM, and MCP interactions with Rich and file logging."""

    def __init__(self, log_dir: str = "logs"):
        """
        Initialize the logger.

        Args:
            log_dir: Directory for log files
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        # Create unique run ID
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"run_{self.run_id}.log"
        self.html_file = self.log_dir / f"run_{self.run_id}.html"

        # Console Rich per stdout
        self.console = Console()

        # Console Rich per file HTML
        self.html_console = Console(
            record=True,
            width=120,
            force_terminal=True,
            force_interactive=False
        )

        # Contatori
        self.llm_calls = 0
        self.tool_calls = 0
        self.mcp_requests = 0

        # Write header to log file
        self._write_to_file(self._make_header())

        # Show banner
        self._show_banner()

    def _make_header(self):
        """Create structured header for the log file."""
        header = []
        header.append("╔" + "═"*78 + "╗")
        header.append("║" + " "*78 + "║")
        header.append("║" + "    🤖 HUMAN DIGITAL TWIN - AGENT LOG".center(78) + "║")
        header.append("║" + " "*78 + "║")
        header.append("║" + f"    Run ID: {self.run_id}".ljust(78) + "║")
        header.append("║" + f"    Timestamp: {datetime.now().isoformat()}".ljust(78) + "║")
        header.append("║" + " "*78 + "║")
        header.append("╚" + "═"*78 + "╝")
        return "\n".join(header) + "\n\n"

    def _show_banner(self):
        """Show initial banner."""
        banner = Text()
        banner.append("Human Digital Twin Agent\n", style="bold cyan")
        banner.append(f"Run ID: {self.run_id}\n", style="dim")
        banner.append(f"Log file: {self.log_file}\n", style="dim")
        banner.append(f"HTML file: {self.html_file}", style="dim")

        self.console.print(Panel(banner, box=box.DOUBLE, border_style="cyan"))
        self.console.print()

    def _write_to_file(self, content: str):
        """Write to log file."""
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(content)

    def log_user_message(self, message: str):
        """
        Log a user message.

        Args:
            message: User's message
        """
        timestamp = datetime.now().strftime("%H:%M:%S")

        # Console
        self.console.print(f"[bold blue][{timestamp}] User:[/bold blue]")
        self.console.print(Panel(message, border_style="blue", box=box.ROUNDED))
        self.console.print()

        # File
        self._write_to_file(f"\n[{timestamp}] USER MESSAGE:\n")
        self._write_to_file(f"{message}\n")
        self._write_to_file("-" * 80 + "\n")

    def log_llm_call(self, messages: list, response: str, model_info: Optional[Dict] = None):
        """
        Log an LLM call.

        Args:
            messages: Messages sent to LLM
            response: LLM response
            model_info: Model info
        """
        self.llm_calls += 1
        timestamp = datetime.now().strftime("%H:%M:%S")

        # Console con Box
        title = f"🤖 LLM CALL #{self.llm_calls}"
        if model_info:
            title += f" | {model_info.get('model', 'N/A')}"
            if model_info.get('with_tools'):
                title += f" [cyan](with {model_info.get('with_tools')} tools)[/cyan]"

        self.console.print(f"\n[bold yellow]{'='*80}[/bold yellow]")
        self.console.print(f"[bold yellow]{title}[/bold yellow]")
        self.console.print(f"[dim]{timestamp}[/dim]")
        self.console.print(f"[bold yellow]{'='*80}[/bold yellow]\n")

        # Mostra PROMPT (system + user messages)
        self.console.print(Panel.fit(
            "[bold cyan]📤 PROMPT[/bold cyan]",
            border_style="cyan"
        ))

        for msg in messages:
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')

            if role == 'system':
                self.console.print(Panel(
                    content[:500] + ("..." if len(content) > 500 else ""),
                    title="[magenta]SYSTEM MESSAGE[/magenta]",
                    border_style="magenta",
                    box=box.ROUNDED
                ))
            elif role == 'user':
                self.console.print(Panel(
                    content[:500] + ("..." if len(content) > 500 else ""),
                    title="[blue]USER MESSAGE[/blue]",
                    border_style="blue",
                    box=box.ROUNDED
                ))

        self.console.print()

        # Aggiungi anche a html_console per HTML export
        self.html_console.print(f"\n[bold yellow]{'='*80}[/bold yellow]")
        self.html_console.print(f"[bold yellow]{title}[/bold yellow]")
        self.html_console.print(f"[dim]{timestamp}[/dim]")
        self.html_console.print(f"[bold yellow]{'='*80}[/bold yellow]\n")

        for msg in messages:
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
            if role == 'system':
                self.html_console.print(Panel(
                    content[:500] + ("..." if len(content) > 500 else ""),
                    title="[magenta]SYSTEM MESSAGE[/magenta]",
                    border_style="magenta",
                    box=box.ROUNDED
                ))
            elif role == 'user':
                self.html_console.print(Panel(
                    content[:500] + ("..." if len(content) > 500 else ""),
                    title="[blue]USER MESSAGE[/blue]",
                    border_style="blue",
                    box=box.ROUNDED
                ))

        # File .log con formato strutturato ASCII
        log_content = []
        log_content.append("\n" + "╔" + "═"*78 + "╗")
        log_content.append("║" + f" 🤖 LLM CALL #{self.llm_calls}".ljust(78) + "║")
        if model_info:
            model_str = f"{model_info.get('model', 'N/A')}"
            if model_info.get('with_tools'):
                model_str += f" (with {model_info.get('with_tools')} tools)"
            log_content.append("║" + f" Model: {model_str}".ljust(78) + "║")
        log_content.append("║" + f" Time: {timestamp}".ljust(78) + "║")
        log_content.append("╚" + "═"*78 + "╝\n")

        # PROMPT section
        log_content.append("┌─ 📤 PROMPT " + "─"*64 + "┐")
        for msg in messages:
            role = msg.get('role', 'unknown').upper()
            content = msg.get('content', '')
            log_content.append(f"│ [{role}]")
            # Wrappa il contenuto
            for line in content.split('\n')[:20]:  # Max 20 righe
                if len(line) > 74:
                    log_content.append(f"│ {line[:74]}...")
                else:
                    log_content.append(f"│ {line.ljust(76)}│")
            log_content.append("│" + " "*76 + "│")
        log_content.append("└" + "─"*78 + "┘\n")

        self._write_to_file("\n".join(log_content))

    def log_tool_call(self, tool_name: str, tool_args: Dict[str, Any], tool_result: str):
        """
        Log a tool call.

        Args:
            tool_name: Tool name
            tool_args: Arguments passed to the tool
            tool_result: Tool result
        """
        self.tool_calls += 1
        timestamp = datetime.now().strftime("%H:%M:%S")

        # Console con separatore chiaro
        self.console.print(f"\n[bold magenta]{'─'*80}[/bold magenta]")
        self.console.print(f"[bold magenta]🔧 TOOL CALL #{self.tool_calls} | {tool_name}[/bold magenta]")
        self.console.print(f"[dim]{timestamp}[/dim]")
        self.console.print(f"[bold magenta]{'─'*80}[/bold magenta]\n")

        # Mostra argomenti se presenti
        if tool_args:
            table = Table(title="Arguments", box=box.SIMPLE, show_header=True)
            table.add_column("Parameter", style="cyan")
            table.add_column("Value", style="white")

            for key, value in tool_args.items():
                table.add_row(key, str(value))

            self.console.print(table)
        else:
            self.console.print("[dim]No arguments[/dim]")

        # Mostra risultato
        self.console.print("\n[bold cyan]Result:[/bold cyan]")
        try:
            # Prova a fare pretty print del JSON
            result_json = json.loads(tool_result)
            syntax = Syntax(json.dumps(result_json, indent=2), "json", theme="monokai", line_numbers=False)
            self.console.print(syntax)
        except:
            # Se non è JSON, mostra come testo
            self.console.print(Panel(tool_result[:500], border_style="magenta", box=box.ROUNDED))

        self.console.print()

        # File
        self._write_to_file(f"\n{'='*80}\n")
        self._write_to_file(f"[{timestamp}] TOOL CALL #{self.tool_calls}:\n")
        self._write_to_file(f"Tool: {tool_name}\n")
        self._write_to_file(f"Arguments: {json.dumps(tool_args, indent=2)}\n")
        self._write_to_file(f"Result:\n{tool_result}\n")
        self._write_to_file(f"{'='*80}\n")

    def log_mcp_request(self, method: str, endpoint: str, params: Optional[Dict] = None, response: Any = None):
        """
        Log an MCP server request.

        Args:
            method: HTTP method
            endpoint: Endpoint called
            params: Request parameters
            response: Server response
        """
        self.mcp_requests += 1
        timestamp = datetime.now().strftime("%H:%M:%S")

        # Console
        self.console.print(f"[bold green][{timestamp}] MCP Request #{self.mcp_requests}:[/bold green]")
        self.console.print(f"  [cyan]{method}[/cyan] {endpoint}")

        if params:
            self.console.print(f"  [dim]Params: {params}[/dim]")

        if response:
            self.console.print("[dim]Response:[/dim]")
            try:
                syntax = Syntax(json.dumps(response, indent=2), "json", theme="monokai", line_numbers=False)
                self.console.print(syntax)
            except:
                self.console.print(f"  {str(response)[:200]}")

        self.console.print()

        # File
        self._write_to_file(f"\n[{timestamp}] MCP REQUEST #{self.mcp_requests}:\n")
        self._write_to_file(f"{method} {endpoint}\n")
        if params:
            self._write_to_file(f"Params: {json.dumps(params, indent=2)}\n")
        if response:
            self._write_to_file(f"Response: {json.dumps(response, indent=2)}\n")
        self._write_to_file("-" * 80 + "\n")

    def log_agent_response(self, response: str):
        """
        Log the final agent response.

        Args:
            response: Agent response
        """
        timestamp = datetime.now().strftime("%H:%M:%S")

        # Console con separatore chiaro
        self.console.print(f"\n[bold green]{'─'*80}[/bold green]")
        self.console.print(Panel.fit(
            "[bold green]📥 RESPONSE[/bold green]",
            border_style="green"
        ))
        self.console.print(Panel(
            response,
            title=f"[green]Agent Response | {timestamp}[/green]",
            border_style="green",
            box=box.DOUBLE
        ))
        self.console.print(f"[bold green]{'─'*80}[/bold green]\n")

        # Aggiungi a html_console
        self.html_console.print(f"\n[bold green]{'─'*80}[/bold green]")
        self.html_console.print(Panel.fit(
            "[bold green]📥 RESPONSE[/bold green]",
            border_style="green"
        ))
        self.html_console.print(Panel(
            response,
            title=f"[green]Agent Response | {timestamp}[/green]",
            border_style="green",
            box=box.DOUBLE
        ))
        self.html_console.print(f"[bold green]{'─'*80}[/bold green]\n")

        # File .log strutturato
        log_content = []
        log_content.append("\n" + "┌─ 📥 RESPONSE " + "─"*63 + "┐")
        log_content.append("│" + f" Time: {timestamp}".ljust(76) + "│")
        log_content.append("├" + "─"*78 + "┤")

        # Wrappa la risposta
        for line in response.split('\n')[:50]:  # Max 50 righe
            if len(line) > 74:
                # Spezza linee lunghe
                while len(line) > 74:
                    log_content.append("│ " + line[:74])
                    line = line[74:]
                if line:
                    log_content.append("│ " + line.ljust(76) + "│")
            else:
                log_content.append("│ " + line.ljust(76) + "│")

        log_content.append("└" + "─"*78 + "┘\n")
        self._write_to_file("\n".join(log_content))

    def log_error(self, error: str, context: Optional[str] = None):
        """
        Log an error.

        Args:
            error: Error message
            context: Error context
        """
        timestamp = datetime.now().strftime("%H:%M:%S")

        # Console
        self.console.print(f"[bold red][{timestamp}] ERROR:[/bold red]")
        if context:
            self.console.print(f"[dim]Context: {context}[/dim]")
        self.console.print(Panel(error, border_style="red", box=box.HEAVY))
        self.console.print()

        # File
        self._write_to_file(f"\n[{timestamp}] ERROR:\n")
        if context:
            self._write_to_file(f"Context: {context}\n")
        self._write_to_file(f"{error}\n")
        self._write_to_file("-" * 80 + "\n")

    def log_summary(self):
        """Show session summary and save HTML."""
        timestamp = datetime.now().strftime("%H:%M:%S")

        # Crea tabella riassuntiva
        table = Table(title="Session Summary", box=box.DOUBLE_EDGE)
        table.add_column("Metric", style="cyan")
        table.add_column("Count", style="yellow", justify="right")

        table.add_row("LLM Calls", str(self.llm_calls))
        table.add_row("Tool Calls", str(self.tool_calls))
        table.add_row("MCP Requests", str(self.mcp_requests))

        self.console.print()
        self.console.print(table)
        self.console.print(f"\n[dim]Log file: {self.log_file}[/dim]")
        self.console.print(f"[dim]HTML file: {self.html_file}[/dim]\n")

        # Aggiungi a html_console
        self.html_console.print()
        self.html_console.print(table)

        # File .log
        log_content = []
        log_content.append("\n" + "╔" + "═"*78 + "╗")
        log_content.append("║" + " SESSION SUMMARY".center(78) + "║")
        log_content.append("╠" + "═"*78 + "╣")
        log_content.append("║" + f" LLM Calls: {self.llm_calls}".ljust(78) + "║")
        log_content.append("║" + f" Tool Calls: {self.tool_calls}".ljust(78) + "║")
        log_content.append("║" + f" MCP Requests: {self.mcp_requests}".ljust(78) + "║")
        log_content.append("║" + f" Time: {timestamp}".ljust(78) + "║")
        log_content.append("╚" + "═"*78 + "╝\n")
        self._write_to_file("\n".join(log_content))

        # Salva HTML con colori Rich
        self._save_html()

    def _save_html(self):
        """Save log in HTML format with Rich colors."""
        try:
            from rich.terminal_theme import MONOKAI

            html_output = self.html_console.export_html(
                theme=MONOKAI,
                inline_styles=True,
                clear=False
            )

            # Aggiungi un titolo custom
            html_with_title = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Agent Log - {self.run_id}</title>
    <style>
        body {{
            background-color: #1e1e1e;
            margin: 20px;
            font-family: 'Consolas', 'Monaco', monospace;
        }}
        h1 {{
            color: #00d9ff;
            text-align: center;
            padding: 20px;
            border-bottom: 2px solid #00d9ff;
        }}
    </style>
</head>
<body>
    <h1>🤖 Human Digital Twin - Agent Log</h1>
    <p style="text-align: center; color: #888;">Run ID: {self.run_id} | {datetime.now().isoformat()}</p>
    {html_output}
</body>
</html>"""

            with open(self.html_file, 'w', encoding='utf-8') as f:
                f.write(html_with_title)

        except Exception as e:
            self.console.print(f"[yellow]Warning: Could not save HTML log: {str(e)}[/yellow]")


# Istanza globale del logger
_global_logger: Optional[AgentLogger] = None


def get_logger(log_dir: str = "logs") -> AgentLogger:
    """
    Get the global logger instance (singleton).

    Args:
        log_dir: Log directory

    Returns:
        Logger instance
    """
    global _global_logger
    if _global_logger is None:
        _global_logger = AgentLogger(log_dir)
    return _global_logger
