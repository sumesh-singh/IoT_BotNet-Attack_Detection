"""
WebSocket Handler for Enhanced IoT BotScan
Author: Kotiwale Sumesh Singh (160124862043)

Handles real-time WebSocket connections for live updates and notifications.
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Set, Optional
from datetime import datetime
import websockets
from websockets.server import WebSocketServerProtocol
from dataclasses import dataclass, asdict
import uuid

logger = logging.getLogger(__name__)


@dataclass
class WebSocketMessage:
    """WebSocket message data structure."""
    type: str
    data: Dict[str, Any]
    timestamp: datetime
    message_id: str


@dataclass
class ClientConnection:
    """Client connection information."""
    websocket: WebSocketServerProtocol
    client_id: str
    subscribed_channels: Set[str]
    connected_at: datetime
    last_activity: datetime


class WebSocketManager:
    """Manages WebSocket connections and real-time updates."""

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize WebSocket manager with configuration."""

        self.config = config or {}
        self.host = self.config.get('host', 'localhost')
        self.port = self.config.get('port', 8000)
        self.clients: Dict[str, ClientConnection] = {}
        self.channels: Dict[str, Set[str]] = {
            'threats': set(),
            'models': set(),
            'drift': set(),
            'alerts': set(),
            'system': set()
        }

        # Message queues for different channels
        self.message_queues: Dict[str, List[WebSocketMessage]] = {
            'threats': [],
            'models': [],
            'drift': [],
            'alerts': [],
            'system': []
        }

        # Statistics
        self.stats = {
            'total_connections': 0,
            'active_connections': 0,
            'messages_sent': 0,
            'messages_received': 0
        }

    async def start_server(self):
        """Start the WebSocket server."""

        logger.info(f"Starting WebSocket server on {self.host}:{self.port}")

        async with websockets.serve(
            self.handle_client,
            self.host,
            self.port,
            ping_interval=30,
            ping_timeout=10
        ):
            logger.info("WebSocket server started successfully")

            # Start background tasks
            await asyncio.gather(
                self.cleanup_inactive_connections(),
                self.process_message_queues(),
                self.send_heartbeat()
            )

    async def handle_client(self, websocket: WebSocketServerProtocol, path: str):
        """Handle new client connection."""

        client_id = str(uuid.uuid4())
        client = ClientConnection(
            websocket=websocket,
            client_id=client_id,
            subscribed_channels=set(),
            connected_at=datetime.now(),
            last_activity=datetime.now()
        )

        self.clients[client_id] = client
        self.stats['total_connections'] += 1
        self.stats['active_connections'] += 1

        logger.info(
            f"Client {client_id} connected from {websocket.remote_address}")

        try:
            # Send welcome message
            await self.send_to_client(client_id, {
                'type': 'connection_established',
                'data': {
                    'client_id': client_id,
                    'server_time': datetime.now().isoformat(),
                    'available_channels': list(self.channels.keys())
                }
            })

            # Handle incoming messages
            async for message in websocket:
                await self.handle_message(client_id, message)

        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client {client_id} disconnected")
        except Exception as e:
            logger.error(f"Error handling client {client_id}: {e}")
        finally:
            await self.disconnect_client(client_id)

    async def handle_message(self, client_id: str, message: str):
        """Handle incoming message from client."""

        try:
            data = json.loads(message)
            self.stats['messages_received'] += 1

            client = self.clients.get(client_id)
            if not client:
                return

            client.last_activity = datetime.now()

            message_type = data.get('type')

            if message_type == 'subscribe':
                await self.handle_subscribe(client_id, data.get('channels', []))
            elif message_type == 'unsubscribe':
                await self.handle_unsubscribe(client_id, data.get('channels', []))
            elif message_type == 'ping':
                await self.handle_ping(client_id)
            elif message_type == 'get_stats':
                await self.handle_get_stats(client_id)
            else:
                logger.warning(f"Unknown message type: {message_type}")

        except json.JSONDecodeError:
            logger.error(f"Invalid JSON message from client {client_id}")
        except Exception as e:
            logger.error(
                f"Error processing message from client {client_id}: {e}")

    async def handle_subscribe(self, client_id: str, channels: List[str]):
        """Handle client subscription to channels."""

        client = self.clients.get(client_id)
        if not client:
            return

        for channel in channels:
            if channel in self.channels:
                client.subscribed_channels.add(channel)
                self.channels[channel].add(client_id)
                logger.info(f"Client {client_id} subscribed to {channel}")

        await self.send_to_client(client_id, {
            'type': 'subscription_confirmed',
            'data': {
                'subscribed_channels': list(client.subscribed_channels)
            }
        })

    async def handle_unsubscribe(self, client_id: str, channels: List[str]):
        """Handle client unsubscription from channels."""

        client = self.clients.get(client_id)
        if not client:
            return

        for channel in channels:
            if channel in self.channels:
                client.subscribed_channels.discard(channel)
                self.channels[channel].discard(client_id)
                logger.info(f"Client {client_id} unsubscribed from {channel}")

        await self.send_to_client(client_id, {
            'type': 'unsubscription_confirmed',
            'data': {
                'subscribed_channels': list(client.subscribed_channels)
            }
        })

    async def handle_ping(self, client_id: str):
        """Handle ping message from client."""

        await self.send_to_client(client_id, {
            'type': 'pong',
            'data': {
                'server_time': datetime.now().isoformat()
            }
        })

    async def handle_get_stats(self, client_id: str):
        """Handle stats request from client."""

        await self.send_to_client(client_id, {
            'type': 'stats',
            'data': {
                'server_stats': self.stats,
                'channel_stats': {
                    channel: len(subscribers)
                    for channel, subscribers in self.channels.items()
                }
            }
        })

    async def send_to_client(self, client_id: str, message_data: Dict[str, Any]):
        """Send message to specific client."""

        client = self.clients.get(client_id)
        if not client:
            return

        try:
            message = WebSocketMessage(
                type=message_data['type'],
                data=message_data['data'],
                timestamp=datetime.now(),
                message_id=str(uuid.uuid4())
            )

            await client.websocket.send(json.dumps(asdict(message), default=str))
            self.stats['messages_sent'] += 1

        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client {client_id} connection closed")
            await self.disconnect_client(client_id)
        except Exception as e:
            logger.error(f"Error sending message to client {client_id}: {e}")

    async def broadcast_to_channel(self, channel: str, message_data: Dict[str, Any]):
        """Broadcast message to all clients subscribed to a channel."""

        if channel not in self.channels:
            logger.warning(f"Unknown channel: {channel}")
            return

        message = WebSocketMessage(
            type=message_data['type'],
            data=message_data['data'],
            timestamp=datetime.now(),
            message_id=str(uuid.uuid4())
        )

        # Send to all subscribers
        disconnected_clients = []
        for client_id in self.channels[channel]:
            try:
                client = self.clients.get(client_id)
                if client:
                    await client.websocket.send(json.dumps(asdict(message), default=str))
                    self.stats['messages_sent'] += 1
                else:
                    disconnected_clients.append(client_id)
            except websockets.exceptions.ConnectionClosed:
                disconnected_clients.append(client_id)
            except Exception as e:
                logger.error(f"Error broadcasting to client {client_id}: {e}")
                disconnected_clients.append(client_id)

        # Clean up disconnected clients
        for client_id in disconnected_clients:
            await self.disconnect_client(client_id)

    async def disconnect_client(self, client_id: str):
        """Disconnect client and clean up resources."""

        client = self.clients.get(client_id)
        if not client:
            return

        # Remove from all channels
        for channel in client.subscribed_channels:
            self.channels[channel].discard(client_id)

        # Remove client
        del self.clients[client_id]
        self.stats['active_connections'] -= 1

        logger.info(f"Client {client_id} disconnected")

    async def cleanup_inactive_connections(self):
        """Clean up inactive connections."""

        while True:
            try:
                current_time = datetime.now()
                inactive_clients = []

                for client_id, client in self.clients.items():
                    # Disconnect clients inactive for more than 5 minutes
                    if (current_time - client.last_activity).total_seconds() > 300:
                        inactive_clients.append(client_id)

                for client_id in inactive_clients:
                    logger.info(f"Disconnecting inactive client {client_id}")
                    await self.disconnect_client(client_id)

                await asyncio.sleep(60)  # Check every minute

            except Exception as e:
                logger.error(f"Error in cleanup task: {e}")
                await asyncio.sleep(60)

    async def process_message_queues(self):
        """Process message queues for each channel."""

        while True:
            try:
                for channel, queue in self.message_queues.items():
                    if queue:
                        # Process oldest message
                        message = queue.pop(0)
                        await self.broadcast_to_channel(channel, {
                            'type': message.type,
                            'data': message.data
                        })

                await asyncio.sleep(0.1)  # Small delay to prevent busy waiting

            except Exception as e:
                logger.error(f"Error processing message queues: {e}")
                await asyncio.sleep(1)

    async def send_heartbeat(self):
        """Send heartbeat to all connected clients."""

        while True:
            try:
                heartbeat_data = {
                    'type': 'heartbeat',
                    'data': {
                        'server_time': datetime.now().isoformat(),
                        'active_connections': self.stats['active_connections']
                    }
                }

                # Send to all connected clients
                disconnected_clients = []
                for client_id, client in self.clients.items():
                    try:
                        await client.websocket.send(json.dumps(heartbeat_data))
                    except websockets.exceptions.ConnectionClosed:
                        disconnected_clients.append(client_id)
                    except Exception as e:
                        logger.error(
                            f"Error sending heartbeat to client {client_id}: {e}")
                        disconnected_clients.append(client_id)

                # Clean up disconnected clients
                for client_id in disconnected_clients:
                    await self.disconnect_client(client_id)

                await asyncio.sleep(30)  # Send heartbeat every 30 seconds

            except Exception as e:
                logger.error(f"Error in heartbeat task: {e}")
                await asyncio.sleep(30)

    # Public methods for external systems to send updates
    async def send_threat_detection(self, threat_data: Dict[str, Any]):
        """Send threat detection update."""

        await self.broadcast_to_channel('threats', {
            'type': 'threat_detected',
            'data': threat_data
        })

    async def send_model_update(self, model_data: Dict[str, Any]):
        """Send model performance update."""

        await self.broadcast_to_channel('models', {
            'type': 'model_performance_updated',
            'data': model_data
        })

    async def send_drift_detection(self, drift_data: Dict[str, Any]):
        """Send concept drift detection update."""

        await self.broadcast_to_channel('drift', {
            'type': 'concept_drift_detected',
            'data': drift_data
        })

    async def send_alert(self, alert_data: Dict[str, Any]):
        """Send alert update."""

        await self.broadcast_to_channel('alerts', {
            'type': 'alert_created',
            'data': alert_data
        })

    async def send_system_metrics(self, metrics_data: Dict[str, Any]):
        """Send system metrics update."""

        await self.broadcast_to_channel('system', {
            'type': 'system_metrics_updated',
            'data': metrics_data
        })

    def get_stats(self) -> Dict[str, Any]:
        """Get WebSocket server statistics."""

        return {
            'server_stats': self.stats,
            'channel_stats': {
                channel: len(subscribers)
                for channel, subscribers in self.channels.items()
            },
            'client_stats': {
                'total_clients': len(self.clients),
                'clients_by_channel': {
                    channel: len(subscribers)
                    for channel, subscribers in self.channels.items()
                }
            }
        }


# Example usage and testing
async def main():
    """Main function for testing WebSocket server."""

    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Create WebSocket manager
    ws_manager = WebSocketManager({
        'host': 'localhost',
        'port': 8000
    })

    # Start server
    await ws_manager.start_server()


if __name__ == '__main__':
    asyncio.run(main())
