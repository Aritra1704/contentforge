"""Router package exports for the stateless generation service."""

from app.routers.generate import router as generate_router
from app.routers.system import router as system_router

__all__ = ["generate_router", "system_router"]
