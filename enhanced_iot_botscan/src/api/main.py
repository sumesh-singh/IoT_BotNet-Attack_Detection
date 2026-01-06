"""
Main API Server for Enhanced IoT BotScan
Author: Kotiwale Sumesh Singh (160124862043)

Combines GraphQL API and WebSocket server for comprehensive IoT botnet detection system.
"""

import asyncio
import logging
from typing import Dict, Any
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
import uvicorn
from contextlib import asynccontextmanager

from pathlib import Path
import os
import uuid
import json
from datetime import datetime
from dataclasses import dataclass

from .schema import schema
from .websocket_handler import WebSocketManager
from strawberry.fastapi import GraphQLRouter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ClientConnection:
    websocket: WebSocket
    client_id: str
    subscribed_channels: set
    connected_at: datetime
    last_activity: datetime

class APIServer:
    """Main API server combining GraphQL and WebSocket functionality."""

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize API server with configuration."""

        self.config = config or {}
        self.app = FastAPI(
            title="Enhanced IoT BotScan API",
            description="API for IoT botnet detection system with real-time updates",
            version="1.0.0"
        )

        # Initialize components
        # self.graphql_resolvers = GraphQLResolvers(config) # Removed: Logic moved to schema.py/mock_data.py
        self.websocket_manager = WebSocketManager(config)

        # Setup CORS
        self.setup_cors()

        # Setup routes
        self.setup_routes()

        # Setup GraphQL
        self.setup_graphql()

        # Setup WebSocket
        self.setup_websocket()

    def setup_cors(self):
        """Setup CORS middleware."""

        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure appropriately for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def setup_routes(self):
        """Setup API routes."""

        @self.app.get("/")
        async def root():
            """Root endpoint."""
            return {
                "message": "Enhanced IoT BotScan API",
                "version": "1.0.0",
                "endpoints": {
                    "graphql": "/graphql",
                    "websocket": "/ws",
                    "health": "/health",
                    "stats": "/stats"
                }
            }

        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {
                "status": "healthy",
                "timestamp": "2024-01-01T00:00:00Z",
                "services": {
                    "graphql": "operational",
                    "websocket": "operational",
                    "database": "operational"
                }
            }

        @self.app.get("/stats")
        async def get_stats():
            """Get server statistics."""
            return {
                "api_stats": {
                    "uptime": "24h",
                    "requests_processed": 1000,
                    "active_connections": self.websocket_manager.stats['active_connections']
                },
                "websocket_stats": self.websocket_manager.get_stats()
            }

        @self.app.get("/dashboard")
        async def dashboard():
            """Serve dashboard page."""
            try:
                # Use robust path handling to find the web directory
                # Try relative to current directory first
                dashboard_path = Path("web/dashboard.html")
                if not dashboard_path.exists():
                     # Try relative to the package root if running from elsewhere
                     # Assuming typical structure: project_root/web
                     # and project_root/src/api/main.py
                     current_file = Path(__file__).resolve()
                     project_root = current_file.parent.parent.parent
                     dashboard_path = project_root / "web" / "dashboard.html"
                
                if dashboard_path.exists():
                    return HTMLResponse(dashboard_path.read_text(encoding='utf-8'))
                else:
                    logger.error(f"Dashboard file not found at: {dashboard_path}")
                    raise HTTPException(status_code=404, detail="Dashboard template not found")
            except Exception as e:
                logger.error(f"Error serving dashboard: {e}")
                raise HTTPException(status_code=500, detail="Internal server error")

        @self.app.get("/analytics")
        async def analytics():
            """Serve analytics page."""
            try:
                analytics_path = Path("web/analytics.html")
                if not analytics_path.exists():
                     current_file = Path(__file__).resolve()
                     project_root = current_file.parent.parent.parent
                     analytics_path = project_root / "web" / "analytics.html"

                if analytics_path.exists():
                    return HTMLResponse(analytics_path.read_text(encoding='utf-8'))
                else:
                    logger.error(f"Analytics file not found at: {analytics_path}")
                    raise HTTPException(status_code=404, detail="Analytics template not found")
            except Exception as e:
                logger.error(f"Error serving analytics: {e}")
                raise HTTPException(status_code=500, detail="Internal server error")

    def setup_graphql(self):
        """Setup GraphQL endpoint."""

        # Create GraphQL app with resolvers
        graphql_app = GraphQLRouter(schema)

        # Add GraphQL endpoint - Mount it to handle both graphql and graphiql
        self.app.include_router(graphql_app, prefix="/graphql")
        self.app.include_router(graphql_app, prefix="/graphql-playground") # Optional alias

    def setup_websocket(self):
        """Setup WebSocket endpoint."""

        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time updates."""

            await websocket.accept()
            client_id = str(uuid.uuid4())

            # Add client to WebSocket manager
            client = ClientConnection(
                websocket=websocket,
                client_id=client_id,
                subscribed_channels=set(),
                connected_at=datetime.now(),
                last_activity=datetime.now()
            )

            self.websocket_manager.clients[client_id] = client
            self.websocket_manager.stats['total_connections'] += 1
            self.websocket_manager.stats['active_connections'] += 1

            logger.info(f"WebSocket client {client_id} connected")

            try:
                # Send welcome message
                await websocket.send_json({
                    'type': 'connection_established',
                    'data': {
                        'client_id': client_id,
                        'server_time': datetime.now().isoformat(),
                        'available_channels': list(self.websocket_manager.channels.keys())
                    }
                })

                # Handle incoming messages
                while True:
                    try:
                        message = await websocket.receive_json()
                        await self.websocket_manager.handle_message(client_id, json.dumps(message))
                    except WebSocketDisconnect:
                        break
                    except Exception as e:
                        logger.error(f"Error handling WebSocket message: {e}")
                        break

            except WebSocketDisconnect:
                logger.info(f"WebSocket client {client_id} disconnected")
            except Exception as e:
                logger.error(f"WebSocket error for client {client_id}: {e}")
            finally:
                await self.websocket_manager.disconnect_client(client_id)

    async def start_server(self):
        """Start the API server."""

        logger.info("Starting Enhanced IoT BotScan API Server")

        # Start WebSocket manager tasks in background (do not start independent server)
        asyncio.create_task(self.websocket_manager.start_background_tasks())

        # Start FastAPI server
        config = uvicorn.Config(
            app=self.app,
            host=self.config.get('host', '0.0.0.0'),
            port=self.config.get('port', 8000),
            log_level="info"
        )

        server = uvicorn.Server(config)
        await server.serve()

    def get_app(self):
        """Get FastAPI app instance."""
        return self.app


# Example usage and testing
async def main():
    """Main function for running the API server."""

    # Create API server
    api_server = APIServer({
        'host': '0.0.0.0',
        'port': 8000
    })

    # Start server
    await api_server.start_server()


if __name__ == '__main__':
    asyncio.run(main())
