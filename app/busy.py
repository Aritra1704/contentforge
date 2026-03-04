"""In-memory concurrency control for generation requests."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
import asyncio


@dataclass(slots=True)
class BusySnapshot:
    """Current limiter state exposed to routes and health checks."""

    active_jobs: int
    queued_jobs: int
    max_concurrent_jobs: int
    max_queue: int

    @property
    def busy(self) -> bool:
        """Return whether the service is saturated or has queued work."""

        return self.active_jobs >= self.max_concurrent_jobs or self.queued_jobs > 0


class BusyError(Exception):
    """Raised when a generation request cannot be admitted."""

    def __init__(self, retry_after_ms: int) -> None:
        super().__init__("busy")
        self.retry_after_ms = retry_after_ms


class BusyManager:
    """Simple in-memory limiter with optional bounded queueing."""

    def __init__(self, *, max_concurrent_jobs: int, max_queue: int, retry_after_ms: int) -> None:
        self.max_concurrent_jobs = max_concurrent_jobs
        self.max_queue = max_queue
        self.retry_after_ms = retry_after_ms
        self._active_jobs = 0
        self._queued_jobs = 0
        self._condition = asyncio.Condition()

    @asynccontextmanager
    async def slot(self) -> AsyncIterator[None]:
        """Acquire a job slot or raise BusyError if admission is denied."""

        granted = await self.acquire()
        if not granted:
            raise BusyError(self.retry_after_ms)

        try:
            yield
        finally:
            await self.release()

    async def acquire(self) -> bool:
        """Acquire capacity immediately or join the bounded queue."""

        async with self._condition:
            if self._active_jobs < self.max_concurrent_jobs:
                self._active_jobs += 1
                return True

            if self._queued_jobs >= self.max_queue:
                return False

            self._queued_jobs += 1
            try:
                while self._active_jobs >= self.max_concurrent_jobs:
                    await self._condition.wait()
                self._queued_jobs -= 1
                self._active_jobs += 1
                return True
            except BaseException:
                self._queued_jobs -= 1
                self._condition.notify(1)
                raise

    async def release(self) -> None:
        """Release one active slot and wake one queued request."""

        async with self._condition:
            if self._active_jobs == 0:
                return

            self._active_jobs -= 1
            self._condition.notify(1)

    async def snapshot(self) -> BusySnapshot:
        """Return a point-in-time snapshot of limiter state."""

        async with self._condition:
            return BusySnapshot(
                active_jobs=self._active_jobs,
                queued_jobs=self._queued_jobs,
                max_concurrent_jobs=self.max_concurrent_jobs,
                max_queue=self.max_queue,
            )
