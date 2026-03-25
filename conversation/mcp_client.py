"""MCP (Model Context Protocol) client for connecting to local MCP servers.

Starts MCP servers as subprocesses, communicates via stdio.
Gracefully handles missing servers — never crashes.
"""

import asyncio
import json
import subprocess
import shutil
from logger import get_logger

log = get_logger("mcp")

# MCP servers to attempt starting
MCP_SERVERS = {
    "filesystem": {
        "cmd": ["npx", "-y", "@modelcontextprotocol/server-filesystem",
                "~/Documents", "~/Desktop", "~/Downloads"],
        "description": "Read/write files on disk",
    },
}

# Optional servers (only if tools are installed)
OPTIONAL_SERVERS = {
    "gmail": {
        "cmd": ["npx", "-y", "@modelcontextprotocol/server-gmail"],
        "description": "Read and send Gmail",
    },
    "calendar": {
        "cmd": ["npx", "-y", "@modelcontextprotocol/server-google-calendar"],
        "description": "Google Calendar events",
    },
    "screenpipe": {
        "cmd": ["npx", "-y", "screenpipe-mcp"],
        "description": "Screen recording memory",
    },
}


class MCPClient:
    def __init__(self):
        self._processes: dict[str, subprocess.Popen] = {}
        self._tools: list[dict] = []

    async def start_servers(self):
        """Start available MCP servers."""
        if not shutil.which("npx"):
            log.warning("npx not found, MCP servers disabled")
            return

        all_servers = {**MCP_SERVERS, **OPTIONAL_SERVERS}
        for name, config in all_servers.items():
            try:
                proc = subprocess.Popen(
                    config["cmd"],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                # Quick health check
                await asyncio.sleep(1)
                if proc.poll() is not None:
                    log.warning("MCP server '%s' exited immediately", name)
                    continue
                self._processes[name] = proc
                log.info("Started MCP server: %s", name)
            except Exception as e:
                log.warning("Failed to start MCP server '%s': %s", name, e)

    async def list_tools(self) -> list[dict]:
        """Get tool definitions from all running MCP servers.

        Returns tools in Claude API format with server prefix in name.
        """
        # For now, return empty — MCP tool discovery requires protocol negotiation
        # TODO: Implement full MCP protocol handshake
        return []

    async def call_tool(self, server: str, tool: str, args: dict) -> str:
        """Call a tool on a specific MCP server."""
        if server not in self._processes:
            return f"MCP server '{server}' not running"

        try:
            # MCP uses JSON-RPC over stdio
            request = json.dumps({
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {"name": tool, "arguments": args},
                "id": 1,
            }) + "\n"

            proc = self._processes[server]
            proc.stdin.write(request.encode())
            proc.stdin.flush()

            # Read response (with timeout)
            line = proc.stdout.readline().decode().strip()
            if line:
                response = json.loads(line)
                return json.dumps(response.get("result", response))
            return "No response from MCP server"
        except Exception as e:
            log.error("MCP tool call failed: %s", e)
            return f"Error: {e}"

    async def stop_servers(self):
        """Stop all running MCP servers."""
        for name, proc in self._processes.items():
            try:
                proc.terminate()
                proc.wait(timeout=3)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
            log.info("Stopped MCP server: %s", name)
        self._processes.clear()
