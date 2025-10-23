"""
FastAPI REST API for RAG Chatbot
Provides HTTP endpoints with full OpenTelemetry instrumentation
"""

import os
import time
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# OpenTelemetry FastAPI instrumentation
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

# Import RAG engine
from rag_engine import ObservableRAG, RAGConfig
from otel_config import initialize_observability


# Request/Response Models
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000, description="Question to ask the RAG system")
    top_k: int = Field(default=4, ge=1, le=10, description="Number of documents to retrieve")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="LLM temperature")

class SourceDocument(BaseModel):
    content: str
    metadata: dict

class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceDocument]
    duration_seconds: float
    num_sources: int
    query: str

class IngestRequest(BaseModel):
    file_paths: List[str] = Field(..., description="List of file paths to ingest")

class IngestResponse(BaseModel):
    num_chunks: int
    duration_seconds: float
    status: str

class HealthResponse(BaseModel):
    status: str
    version: str
    environment: str
    otel_endpoint: str


# Global RAG instance
rag_instance = None
otel_config = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    global rag_instance, otel_config
    
    # Startup: Initialize observability and RAG
    print("🚀 Starting RAG Chatbot API...")
    
    # Initialize OpenTelemetry
    tracer, rag_metrics, otel_config = initialize_observability()
    print("✅ OpenTelemetry initialized")
    
    # Initialize RAG system
    config = RAGConfig()
    rag_instance = ObservableRAG(config)
    
    # Load existing vector store if available
    if os.path.exists(config.vector_db_path):
        from langchain_community.vectorstores import Chroma
        rag_instance.vectorstore = Chroma(
            persist_directory=config.vector_db_path,
            embedding_function=rag_instance.embeddings
        )
        rag_instance.setup_qa_chain()
        print("✅ Loaded existing vector database")
    else:
        print("⚠️  No vector database found. Use /ingest endpoint to add documents.")
    
    print("🎉 RAG Chatbot API ready!")
    
    yield
    
    # Shutdown: Clean up resources
    print("🛑 Shutting down...")
    if otel_config:
        otel_config.shutdown()


# Create FastAPI app
app = FastAPI(
    title="RAG Chatbot API",
    description="Production-ready RAG system with OpenTelemetry observability",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instrument FastAPI with OpenTelemetry
FastAPIInstrumentor.instrument_app(app)


@app.get("/", response_model=dict)
async def root():
    """Root endpoint"""
    return {
        "message": "RAG Chatbot API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        environment=os.getenv("ENVIRONMENT", "development"),
        otel_endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "not configured")
    )


@app.get("/ready")
async def readiness_check():
    """Readiness check endpoint"""
    if rag_instance is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    if rag_instance.vectorstore is None:
        raise HTTPException(status_code=503, detail="Vector database not loaded")
    
    return {"status": "ready"}


@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """
    Query the RAG system
    
    - **question**: The question to ask
    - **top_k**: Number of relevant documents to retrieve (1-10)
    - **temperature**: LLM temperature for response generation (0.0-2.0)
    """
    if rag_instance is None or rag_instance.vectorstore is None:
        raise HTTPException(
            status_code=503,
            detail="RAG system not initialized. Please ingest documents first using /ingest endpoint."
        )
    
    try:
        # Update config temporarily
        rag_instance.config.top_k_results = request.top_k
        rag_instance.config.temperature = request.temperature
        rag_instance.llm.temperature = request.temperature
        
        # Query the RAG system
        result = rag_instance.query(request.question)
        
        # Format response
        sources = [
            SourceDocument(
                content=doc.page_content[:500],  # Truncate for API response
                metadata=doc.metadata
            )
            for doc in result["source_documents"]
        ]
        
        return QueryResponse(
            answer=result["answer"],
            sources=sources,
            duration_seconds=result["duration_seconds"],
            num_sources=result["num_sources"],
            query=request.question
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.post("/ingest", response_model=IngestResponse)
async def ingest_documents(request: IngestRequest):
    """
    Ingest documents into the vector database
    
    - **file_paths**: List of file paths to ingest (supports .txt, .pdf, .md)
    """
    if rag_instance is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        start_time = time.time()
        
        # Validate file paths exist
        missing_files = [fp for fp in request.file_paths if not os.path.exists(fp)]
        if missing_files:
            raise HTTPException(
                status_code=400,
                detail=f"Files not found: {missing_files}"
            )
        
        # Ingest documents
        num_chunks = rag_instance.ingest_documents(request.file_paths)
        
        # Setup QA chain
        rag_instance.setup_qa_chain()
        
        duration = time.time() - start_time
        
        return IngestResponse(
            num_chunks=num_chunks,
            duration_seconds=duration,
            status="success"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@app.get("/stats")
async def get_stats():
    """Get vector database statistics"""
    if rag_instance is None or rag_instance.vectorstore is None:
        raise HTTPException(status_code=503, detail="Vector database not loaded")
    
    try:
        # Get collection stats from ChromaDB
        collection = rag_instance.vectorstore._collection
        count = collection.count()
        
        return {
            "total_documents": count,
            "vector_db_path": rag_instance.config.vector_db_path,
            "embedding_model": rag_instance.config.embedding_deployment,
            "llm_model": rag_instance.config.azure_openai_deployment
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler with telemetry"""
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "path": str(request.url)
        }
    )


if __name__ == "__main__":
    import uvicorn
    
    # Run the server
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=os.getenv("ENVIRONMENT") == "development",
        log_level="info"
    )