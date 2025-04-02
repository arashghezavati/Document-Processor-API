"""
WebSocket manager for real-time document updates
"""
from enum import Enum
from typing import List, Dict, Any
from fastapi import WebSocket
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EventTypes(str, Enum):
    """Event types for WebSocket notifications"""
    DOCUMENT_ADDED = "document_added"
    DOCUMENT_DELETED = "document_deleted"
    FOLDER_ADDED = "folder_added"
    FOLDER_DELETED = "folder_deleted"
    PROCESSING_COMPLETE = "processing_complete"
    PROCESSING_ERROR = "processing_error"

class ConnectionManager:
    """Manages WebSocket connections"""
    
    def __init__(self):
        # Store connections by username
        self.active_connections: Dict[str, List[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, username: str):
        """Connect a new client"""
        await websocket.accept()
        
        # Add connection to the user's list
        if username not in self.active_connections:
            self.active_connections[username] = []
        
        self.active_connections[username].append(websocket)
        logger.info(f"New WebSocket connection for user {username}. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket, username: str):
        """Disconnect a client"""
        if username in self.active_connections:
            # Remove this specific connection
            try:
                self.active_connections[username].remove(websocket)
                
                # If no more connections for this user, remove the entry
                if not self.active_connections[username]:
                    del self.active_connections[username]
            except ValueError:
                pass  # Connection might already be removed
    
    async def send_personal_message(self, message: str, username: str):
        """Send a text message to a specific user"""
        if username in self.active_connections:
            disconnected = []
            
            for connection in self.active_connections[username]:
                try:
                    await connection.send_text(message)
                except Exception:
                    # Connection might be closed
                    disconnected.append(connection)
            
            # Clean up disconnected connections
            for conn in disconnected:
                self.disconnect(conn, username)
    
    async def send_personal_json(self, data: Dict[str, Any], username: str):
        """Send JSON data to a specific user"""
        if username in self.active_connections:
            disconnected = []
            
            for connection in self.active_connections[username]:
                try:
                    await connection.send_json(data)
                except Exception:
                    # Connection might be closed
                    disconnected.append(connection)
            
            # Clean up disconnected connections
            for conn in disconnected:
                self.disconnect(conn, username)

# Create a singleton manager instance
manager = ConnectionManager()

async def notify_document_change(username: str, event_type: EventTypes, document_info: Dict[str, Any]):
    """
    Notify a user about document changes
    
    Args:
        username: The username to notify
        event_type: Type of event (added, deleted, etc.)
        document_info: Information about the document
    """
    await manager.send_personal_json({
        "type": event_type,
        "data": document_info
    }, username)
