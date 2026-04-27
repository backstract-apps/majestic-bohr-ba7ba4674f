import json
from fastapi import WebSocket
from typing import Dict, List, Optional, MutableMapping, Any


class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        origin = websocket.headers.get("origin")
        # Accept from any origin to avoid CORS rejection on WebSocket handshake
        await websocket.accept(headers=[(b"access-control-allow-origin", (origin or "*").encode())])
        self.active_connections[client_id] = websocket

    def disconnect(self, client_id: str):
        self.active_connections.pop(client_id, None)

    async def send_personal_message(self, message: str, client_id: str):
        websocket = self.active_connections.get(client_id)
        if websocket:
            await websocket.send_text(message)

    async def broadcast(
        self,
        message: Any,
        send_method: str = "send_text",
        filter_clients: Optional[List[str]] = None,
        filter_type: Optional[str] = None,
    ):
        for cid, ws in self.active_connections.items():
            if filter_clients and filter_type:
                if filter_type == "exclude" and cid in filter_clients:
                    continue
                if filter_type == "include" and cid not in filter_clients:
                    continue
            if send_method == "send_json":
                data = message if isinstance(message, (dict, list)) else json.loads(message)
                await ws.send_json(data)
            else:
                await ws.send_text(str(message))

    async def receive_text(self, client_id: str) -> str:
        websocket = self.active_connections.get(client_id)
        if not websocket:
            raise ValueError("Client not connected")

        return await websocket.receive_text()

# Singleton instance
ws_connection_manager = ConnectionManager()