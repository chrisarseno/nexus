"""
Cog-Eng REST API Server
FastAPI server for standalone deployment and integration
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging
import asyncio
import json

from .client import CognitiveEngine, CogEngConfig, CogEngResponse

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Cog-Eng API",
    description="Cognitive Engine - Standalone AGI Core System API",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global engine instance
engine: Optional[CognitiveEngine] = None

# Request/Response Models
class ProcessRequest(BaseModel):
    """Request model for processing tasks."""
    task: str = Field(..., description="The task to process")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")
    priority: str = Field("normal", description="Priority: low, normal, high, critical")
    require_verification: bool = Field(True, description="Whether to verify results")

class ProcessResponse(BaseModel):
    """Response model for processed tasks."""
    response: str
    confidence: float
    knowledge_nodes_added: int
    agents_involved: List[str]
    processing_time: float
    consciousness_state: Optional[Dict[str, Any]] = None
    learning_insights: Optional[Dict[str, Any]] = None
    safety_evaluation: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    timestamp: str

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    initialized: bool
    components: Dict[str, bool]
    timestamp: str

class SystemStateResponse(BaseModel):
    """System state response."""
    initialized: bool
    config: Dict[str, Any]
    processing_stats: Dict[str, Any]
    consciousness: Optional[Dict[str, Any]] = None
    timestamp: str

class ConfigUpdateRequest(BaseModel):
    """Request to update configuration."""
    enable_consciousness: Optional[bool] = None
    enable_learning: Optional[bool] = None
    enable_agents: Optional[bool] = None
    enable_routing: Optional[bool] = None
    safety_threshold: Optional[float] = None
    confidence_threshold: Optional[float] = None
    routing_strategy: Optional[str] = None

# Startup/Shutdown Events
@app.on_event("startup")
async def startup_event():
    """Initialize the Cognitive Engine on startup."""
    global engine

    logger.info("Starting Cog-Eng API Server...")

    try:
        # Create engine with default config
        engine = CognitiveEngine(
            enable_consciousness=True,
            enable_learning=True,
            enable_agents=True,
            enable_routing=True
        )

        # Initialize asynchronously
        await engine.initialize()

        logger.info("✅ Cog-Eng API Server started successfully")

    except Exception as e:
        logger.error(f"Failed to start Cog-Eng API Server: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    global engine

    logger.info("Shutting down Cog-Eng API Server...")

    if engine:
        await engine.shutdown()

    logger.info("Cog-Eng API Server stopped")

# API Endpoints
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Cog-Eng API",
        "version": "0.1.0",
        "description": "Cognitive Engine - Standalone AGI Core System",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    components = {
        "consciousness": engine.consciousness_core is not None,
        "learning": engine.learning_loop is not None,
        "agents": engine.agent_orchestrator is not None,
        "routing": engine.adaptive_router is not None
    }

    return HealthResponse(
        status="healthy" if engine.initialized else "initializing",
        version="0.1.0",
        initialized=engine.initialized,
        components=components,
        timestamp=datetime.now().isoformat()
    )

@app.post("/process", response_model=ProcessResponse)
async def process_task(request: ProcessRequest):
    """
    Process a task through the Cognitive Engine.

    This is the main endpoint for task processing.
    """
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    try:
        result = await engine.process(
            task=request.task,
            context=request.context,
            priority=request.priority,
            require_verification=request.require_verification
        )

        return ProcessResponse(
            response=result.response,
            confidence=result.confidence,
            knowledge_nodes_added=result.knowledge_nodes_added,
            agents_involved=result.agents_involved,
            processing_time=result.processing_time,
            consciousness_state=result.consciousness_state,
            learning_insights=result.learning_insights,
            safety_evaluation=result.safety_evaluation,
            metadata=result.metadata,
            timestamp=result.timestamp.isoformat()
        )

    except Exception as e:
        logger.error(f"Error processing task: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/state", response_model=SystemStateResponse)
async def get_system_state():
    """Get comprehensive system state."""
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    state = engine.get_system_state()

    return SystemStateResponse(**state)

@app.get("/consciousness/state")
async def get_consciousness_state():
    """Get detailed consciousness state."""
    if not engine or not engine.consciousness_core:
        raise HTTPException(status_code=503, detail="Consciousness not enabled")

    return engine.consciousness_core.get_system_state()

@app.post("/consciousness/pause")
async def pause_consciousness():
    """Pause consciousness system."""
    if not engine or not engine.consciousness_core:
        raise HTTPException(status_code=503, detail="Consciousness not enabled")

    success = engine.consciousness_core.pause_system()

    return {"success": success, "status": "paused"}

@app.post("/consciousness/resume")
async def resume_consciousness():
    """Resume consciousness system."""
    if not engine or not engine.consciousness_core:
        raise HTTPException(status_code=503, detail="Consciousness not enabled")

    success = engine.consciousness_core.resume_system()

    return {"success": success, "status": "active"}

@app.post("/consciousness/emergency_stop")
async def emergency_stop():
    """Emergency stop all systems."""
    if not engine or not engine.consciousness_core:
        raise HTTPException(status_code=503, detail="Consciousness not enabled")

    success = engine.consciousness_core.emergency_stop()

    return {"success": success, "status": "emergency_stop"}

@app.get("/consciousness/stream")
async def stream_consciousness():
    """Stream real-time consciousness updates via Server-Sent Events."""
    if not engine or not engine.consciousness_core:
        raise HTTPException(status_code=503, detail="Consciousness not enabled")

    async def event_stream():
        """Generate Server-Sent Events for consciousness updates."""
        queue = asyncio.Queue()

        def callback(state):
            asyncio.create_task(queue.put(state))

        # Add callback
        engine.add_consciousness_callback(callback)

        try:
            while True:
                state = await queue.get()
                yield f"data: {json.dumps(state)}\n\n"
                await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            pass

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream"
    )

@app.get("/stats")
async def get_statistics():
    """Get processing statistics."""
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    return engine.processing_stats

@app.post("/config", response_model=Dict[str, Any])
async def update_config(request: ConfigUpdateRequest):
    """Update engine configuration."""
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    # Update config
    updates = {}
    for field, value in request.dict(exclude_none=True).items():
        if hasattr(engine.config, field):
            setattr(engine.config, field, value)
            updates[field] = value

    return {
        "updated": updates,
        "current_config": engine.config.__dict__
    }

@app.post("/reinitialize")
async def reinitialize_engine():
    """Reinitialize the engine with current configuration."""
    global engine

    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    try:
        # Shutdown current engine
        await engine.shutdown()

        # Reinitialize
        await engine.initialize()

        return {"success": True, "message": "Engine reinitialized successfully"}

    except Exception as e:
        logger.error(f"Error reinitializing engine: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Run server
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=3737,
        reload=True,
        log_level="info"
    )
