

from fastapi import APIRouter
from src.api.v1.auth import router as auth_router
from src.api.v1.document import router as document_router
from src.api.v1.audit import router as audit_router
from src.api.v1.chat import router as chat_router

api_router = APIRouter(prefix="/api/v1")

api_router.include_router(auth_router)
api_router.include_router(document_router)
api_router.include_router(audit_router)
api_router.include_router(chat_router)
