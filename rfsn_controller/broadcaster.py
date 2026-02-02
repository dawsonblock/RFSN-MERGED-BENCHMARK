"""RFSN Progress Broadcaster.
from __future__ import annotations

Sends real-time events to the local dashboard via HTTP.
Fire-and-forget architecture to avoid blocking the controller.
"""

import queue
import threading
from dataclasses import asdict, dataclass
from typing import Any

import httpx

# Default dashboard URL
DASHBOARD_URL = "http://localhost:8000/api/events"


@dataclass
class Event:
    type: str
    data: dict[str, Any]
    run_id: str | None = None


class ProgressBroadcaster:
    """Async event broadcaster for the dashboard."""

    def __init__(self, run_id: str | None = None):
        self.run_id = run_id
        self._queue: queue.Queue[Event] = queue.Queue()
        self._stop_event = threading.Event()
        self._worker_thread = threading.Thread(target=self._worker, daemon=True)
        self._worker_thread.start()
        self.enabled = True

    def log(self, message: str, level: str = "info") -> None:
        """Broadcast a log message."""
        self._enqueue("log", {"message": message, "level": level})

    def status(
        self, phase: str, step: int | None = None, max_steps: int | None = None
    ) -> None:
        """Broadcast status update."""
        data: dict[str, Any] = {"phase": phase}
        if step is not None:
            data["step"] = step
        if max_steps is not None:
            data["max_steps"] = max_steps
        self._enqueue("status", data)

    def metric(
        self,
        patches_tried: int,
        success_rate: float,
        cost_est: float,
        tokens: int = 0,
    ) -> None:
        """Broadcast metrics."""
        self._enqueue(
            "metric",
            {
                "patches_tried": patches_tried,
                "success_rate": round(success_rate, 1),
                "cost_est": cost_est,
                "tokens": tokens,
            },
        )

    def tool(
        self, name: str, description: str, args: dict[str, Any] | None = None
    ) -> None:
        """Broadcast tool execution event."""
        self._enqueue(
            "tool", {"name": name, "description": description, "args": args or {}}
        )

    def thinking(self, active: bool, thought: str | None = None) -> None:
        """Broadcast AI thinking state."""
        data: dict[str, Any] = {"active": active}
        if thought:
            data["thought"] = thought
        self._enqueue("thinking", data)

    def step(self, step_num: int, summary: str, tool: str | None = None) -> None:
        """Broadcast step completion."""
        self._enqueue("step", {"step": step_num, "summary": summary, "tool": tool})

    def _enqueue(self, event_type: str, data: dict[str, Any]) -> None:
        if not self.enabled:
            return
        self._queue.put(Event(type=event_type, data=data, run_id=self.run_id))

    def _worker(self) -> None:
        """Background worker to send requests."""
        # Create a persistent httpx client for connection pooling
        with httpx.Client(timeout=0.2) as client:
            while not self._stop_event.is_set():
                try:
                    event = self._queue.get(timeout=0.5)
                    try:
                        client.post(
                            DASHBOARD_URL,
                            json=asdict(event),
                        )
                    except (httpx.HTTPError, httpx.TimeoutException):
                        # Dashboard probably not running, ignore
                        pass
                    finally:
                        self._queue.task_done()
                except queue.Empty:
                    continue

    def close(self) -> None:
        """Stop the worker."""
        self._stop_event.set()
        if self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.0)
