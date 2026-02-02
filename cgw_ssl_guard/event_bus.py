"""A minimal publish/subscribe event bus.

The event bus is used throughout the CGW/SSL guard runtime to emit
events at the points of truth: when a winner is selected from the
thalamic gate and when a CGW state is committed.  Subscribers can
register callbacks for specific event names via ``on()``.  When an
event is emitted via ``emit()``, all registered callbacks for that
event name are invoked synchronously with the event payload.

This implementation is deliberately simple and synchronous—callbacks
run in the order they were registered and any exceptions will
propagate to the caller.  It is the caller's responsibility to handle
errors and long‑running work.
"""

from typing import Any, Callable, Dict, List


class SimpleEventBus:
    """A simple synchronous event bus.

    Subscribers register callbacks on a named channel via ``on()``.
    Publishers emit events via ``emit()``.  Callbacks receive the
    payload passed to ``emit`` and are invoked in registration order.
    """

    def __init__(self) -> None:
        # Map of event name to list of callbacks
        self._subscribers: Dict[str, List[Callable[[Any], None]]] = {}

    def on(self, event_name: str, callback: Callable[[Any], None]) -> None:
        """Register a callback for ``event_name``.

        Args:
            event_name: Name of the event channel to subscribe to.
            callback: Callable invoked with the event payload when
                ``emit()`` is called with the same event name.
        """
        self._subscribers.setdefault(event_name, []).append(callback)

    def emit(self, event_name: str, payload: Any) -> None:
        """Emit an event on ``event_name``.

        All callbacks registered via ``on()`` for this event name are
        invoked synchronously with the payload.  Exceptions propagate
        to the caller.

        Args:
            event_name: The name of the event to emit.
            payload: Arbitrary data passed to subscribers.
        """
        for callback in self._subscribers.get(event_name, []):
            callback(payload)