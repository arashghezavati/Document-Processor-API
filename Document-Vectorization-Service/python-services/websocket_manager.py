"""
WebSocket manager for real-time document updates
"""
from enum import Enum
from typing import List, Dict, Any
from fastapi import WebSocket
import logging

logger = logging.getLogger(__name__)

class EventTypes(str, Enum):
    """Event types for WebSocket notifications."""
    DOCUMENT_ADDED = "document_added"
    DOCUMENT_DELETED = "document_deleted"
    FOLDER_ADDED = "folder_added"
    FOLDER_DELETED = "folder_deleted"


class ConnectionManager:
    """Manage WebSocket connections."""
    
    def __init__(self):
        # Store active connections by username
        self.active_connections: Dict[str, List[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, username: str):
        """Connect a user to the WebSocket."""
        await websocket.accept()
        
        if username not in self.active_connections:
            self.active_connections[username] = []
            
        self.active_connections[username].append(websocket)
        logger.info(f"WebSocket connected for user {username}. Total connections: {len(self.active_connections[username])}")
    
    def disconnect(self, websocket: WebSocket, username: str):
        """Disconnect a user from the WebSocket."""
        if username in self.active_connections:
            try:
                self.active_connections[username].remove(websocket)
                logger.info(f"WebSocket disconnected for user {username}. Remaining connections: {len(self.active_connections[username])}")
                
                # Clean up if no connections remain for this user
                if not self.active_connections[username]:
                    del self.active_connections[username]
                    logger.info(f"Removed user {username} from active connections")
            except ValueError:
                pass
    
    async def send_personal_message(self, message: dict, username: str):
        """Send a message to a specific user."""
        if username in self.active_connections:
            disconnected_sockets = []
            
            for websocket in self.active_connections[username]:
                try:
                    await websocket.send_json(message)
                except Exception as e:
                    logger.error(f"Error sending message to user {username}: {str(e)}")
                    disconnected_sockets.append(websocket)
            
            # Clean up any disconnected websockets
            for websocket in disconnected_sockets:
                self.disconnect(websocket, username)
    
    async def broadcast(self, message: dict):
        """Broadcast a message to all connected clients."""
        for username in list(self.active_connections.keys()):
            await self.send_personal_message(message, username)


# Create a singleton manager
manager = ConnectionManager()


async def notify_document_change(username: str, event_type: EventTypes, data: Dict[str, Any]):
    """Notify a user about document changes."""
    await manager.send_personal_message(
        {
            "type": event_type,
            "data": data
        },
        username
    )
