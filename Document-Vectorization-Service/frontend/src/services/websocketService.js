/**
 * WebSocket service for real-time updates
 */

class WebSocketService {
  constructor() {
    this.socket = null;
    this.isConnected = false;
    this.reconnectAttempts = 0;
    this.maxReconnectAttempts = 5;
    this.reconnectTimeout = null;
    this.eventListeners = {
      document_added: [],
      document_deleted: [],
      document_updated: [],
      folder_added: [],
      folder_deleted: [],
      connection_established: [],
      connection_error: [],
      connection_closed: []
    };
  }

  /**
   * Connect to the WebSocket server
   * @param {string} token - Authentication token
   */
  connect(token) {
    if (this.socket) {
      this.disconnect();
    }

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host;
    const wsUrl = `${protocol}//${host}/ws?token=${token}`;

    console.log('Connecting to WebSocket server...');
    this.socket = new WebSocket(wsUrl);

    this.socket.onopen = () => {
      console.log('WebSocket connection established');
      this.isConnected = true;
      this.reconnectAttempts = 0;
      this._notifyListeners('connection_established', { connected: true });
    };

    this.socket.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        console.log('WebSocket message received:', data);
        
        if (data.type && this.eventListeners[data.type]) {
          this._notifyListeners(data.type, data.data);
        }
      } catch (error) {
        console.error('Error parsing WebSocket message:', error);
      }
    };

    this.socket.onclose = (event) => {
      this.isConnected = false;
      console.log(`WebSocket connection closed: ${event.code} - ${event.reason}`);
      this._notifyListeners('connection_closed', { 
        code: event.code, 
        reason: event.reason 
      });

      // Attempt to reconnect if not closed intentionally
      if (event.code !== 1000 && this.reconnectAttempts < this.maxReconnectAttempts) {
        this._scheduleReconnect();
      }
    };

    this.socket.onerror = (error) => {
      console.error('WebSocket error:', error);
      this._notifyListeners('connection_error', { error });
    };
  }

  /**
   * Disconnect from the WebSocket server
   */
  disconnect() {
    if (this.socket) {
      this.socket.close(1000, 'Client disconnected');
      this.socket = null;
      this.isConnected = false;
      
      // Clear any pending reconnect
      if (this.reconnectTimeout) {
        clearTimeout(this.reconnectTimeout);
        this.reconnectTimeout = null;
      }
    }
  }

  /**
   * Schedule a reconnection attempt
   * @private
   */
  _scheduleReconnect() {
    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout);
    }

    const delay = Math.min(1000 * Math.pow(2, this.reconnectAttempts), 30000);
    console.log(`Scheduling reconnect attempt in ${delay}ms`);
    
    this.reconnectTimeout = setTimeout(() => {
      console.log(`Reconnect attempt ${this.reconnectAttempts + 1}/${this.maxReconnectAttempts}`);
      this.reconnectAttempts++;
      
      // Get token from localStorage
      const token = localStorage.getItem('token');
      if (token) {
        this.connect(token);
      } else {
        console.error('Cannot reconnect: No authentication token available');
      }
    }, delay);
  }

  /**
   * Add an event listener
   * @param {string} event - Event type to listen for
   * @param {Function} callback - Callback function
   */
  addEventListener(event, callback) {
    if (this.eventListeners[event]) {
      this.eventListeners[event].push(callback);
    } else {
      console.warn(`Unknown event type: ${event}`);
    }
  }

  /**
   * Remove an event listener
   * @param {string} event - Event type
   * @param {Function} callback - Callback function to remove
   */
  removeEventListener(event, callback) {
    if (this.eventListeners[event]) {
      this.eventListeners[event] = this.eventListeners[event].filter(
        (cb) => cb !== callback
      );
    }
  }

  /**
   * Notify all listeners of an event
   * @param {string} event - Event type
   * @param {Object} data - Event data
   * @private
   */
  _notifyListeners(event, data) {
    if (this.eventListeners[event]) {
      this.eventListeners[event].forEach((callback) => {
        try {
          callback(data);
        } catch (error) {
          console.error(`Error in ${event} event listener:`, error);
        }
      });
    }
  }
}

// Create a singleton instance
const websocketService = new WebSocketService();
export default websocketService;
