"""
Enhanced IoT BotScan - Main API
Migrated to use Strawberry GraphQL
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from strawberry.fastapi import GraphQLRouter
import uvicorn

# Import your GraphQL schema
try:
    from api.schema import schema
    HAS_GRAPHQL = True
except ImportError:
    HAS_GRAPHQL = False
    print("WARNING: GraphQL schema not found. Create api/schema.py")

# ============================================
# FastAPI App
# ============================================

app = FastAPI(
    title="Enhanced IoT BotScan API",
    description="ML-powered IoT botnet detection system",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# GraphQL Endpoint
# ============================================

if HAS_GRAPHQL:
    graphql_app = GraphQLRouter(schema)
    app.include_router(graphql_app, prefix="/graphql")
    print("✓ GraphQL endpoint mounted at /graphql")
else:
    print("✗ GraphQL endpoint not available")

# ============================================
# REST Endpoints
# ============================================

@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "name": "Enhanced IoT BotScan API",
        "version": "2.0.0",
        "status": "operational",
        "endpoints": {
            "graphql": "/graphql" if HAS_GRAPHQL else "unavailable",
            "docs": "/docs",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": "2025-12-21",
        "graphql_available": HAS_GRAPHQL
    }

@app.get("/api/devices")
async def get_devices():
    """REST endpoint for devices (legacy support)"""
    # TODO: Implement actual logic
    return {"devices": []}

@app.post("/api/scan")
async def trigger_scan(device_id: str):
    """REST endpoint to trigger device scan"""
    # TODO: Implement actual logic
    return {"status": "scan_initiated", "device_id": device_id}

# ============================================
# Startup
# ============================================

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    print("🚀 Enhanced IoT BotScan API starting...")
    print(f"📡 GraphQL: {'Enabled' if HAS_GRAPHQL else 'Disabled'}")
    print("✓ Ready to accept connections")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("👋 Shutting down Enhanced IoT BotScan API")

# ============================================
# Run
# ============================================

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
