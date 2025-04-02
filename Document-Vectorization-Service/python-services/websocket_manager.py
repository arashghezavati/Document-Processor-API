"""
WebSocket manager for real-time document updates
"""
from typing import List, Dict, Any
from fastapi import WebSocket
from starlette.websockets import WebSocketDisconnect
import logging
import json
import asyncio

logger = logging.getLogger(__name__)

class ConnectionManager:
    """
    Manages WebSocket connections and broadcasts messages to connected clients
    """
    def __init__(self):
        # Store active connections by user_id
        self.active_connections: Dict[str, List[WebSocket]] = {}
        
    async def connect(self, websocket: WebSocket, user_id: str):
        """
        Accept a new WebSocket connection and store it
        """
        await websocket.accept()
        if user_id not in self.active_connections:
            self.active_connections[user_id] = []
        self.active_connections[user_id].append(websocket)
        logger.info(f"New WebSocket connection for user {user_id}. Total connections: {self.connection_count()}")
    
    def disconnect(self, websocket: WebSocket, user_id: str):
        """
        Remove a WebSocket connection
        """
        if user_id in self.active_connections:
            if websocket in self.active_connections[user_id]:
                self.active_connections[user_id].remove(websocket)
            # Clean up empty lists
            if not self.active_connections[user_id]:
                del self.active_connections[user_id]
        logger.info(f"WebSocket disconnected for user {user_id}. Total connections: {self.connection_count()}")
    
    def connection_count(self):
        """
        Count total number of active connections
        """
        return sum(len(connections) for connections in self.active_connections.values())
    
    async def send_personal_message(self, message: Dict[str, Any], user_id: str):
        """
        Send a message to a specific user's connections
        """
        if user_id in self.active_connections:
            dead_connections = []
            for connection in self.active_connections[user_id]:
                try:
                    await connection.send_json(message)
                except Exception as e:
                    logger.error(f"Error sending message to user {user_id}: {str(e)}")
                    dead_connections.append(connection)
            
            # Clean up dead connections
            for dead_connection in dead_connections:
                self.active_connections[user_id].remove(dead_connection)
            
            if dead_connections:
                logger.info(f"Removed {len(dead_connections)} dead connections for user {user_id}")
    
    async def broadcast(self, message: Dict[str, Any]):
        """
        Broadcast a message to all connected clients
        """
        for user_id in list(self.active_connections.keys()):
            await self.send_personal_message(message, user_id)

# Create a global connection manager instance
manager = ConnectionManager()

# Event types
class EventTypes:
    DOCUMENT_ADDED = "document_added"
    DOCUMENT_DELETED = "document_deleted"
    DOCUMENT_UPDATED = "document_updated"
    FOLDER_ADDED = "folder_added"
    FOLDER_DELETED = "folder_deleted"

async def notify_document_change(user_id: str, event_type: str, data: Dict[str, Any]):
    """
    Notify clients about document changes
    """
    message = {
        "type": event_type,
        "data": data
    }
    await manager.send_personal_message(message, user_id)
