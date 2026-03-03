"""Router package exports for the standalone LLM comparator."""

from app.routers.generate import router as generate_router
from app.routers.results import router as results_router

__all__ = ["generate_router", "results_router"]
