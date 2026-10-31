from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.router import api_router


app = FastAPI(
    title="Agentic RAG Dashboard API",
    description=(
        "Production-ready backend API for the Agentic RAG Dashboard. "
        "Powers authentication, normalized RBAC, knowledge base retrieval, and agent workflows."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)


# Allows frontend development servers (e.g. Vite on localhost:5173 or Next.js on localhost:3000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production domain restrictions
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)


@app.get("/health", tags=["Health"], summary="API Health Check")
async def health_check() -> dict[str, str]:
    return {"status": "healthy", "service": "agentic-rag-api"}
