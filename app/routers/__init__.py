"""Router package exports for the stateless generation service."""

from app.routers.generate import legacy_router as generation_router
from app.routers.generate import router as generate_router
from app.routers.judge import router as judge_router
from app.routers.quality import router as quality_router
from app.routers.system import router as system_router

__all__ = ["generate_router", "generation_router", "judge_router", "quality_router", "system_router"]
