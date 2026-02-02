"""CGW Persistent Event Store.

SQLite-backed event storage for CGW decisions, enabling:
- Persistent event logging across sessions
- Replay and debugging
- Query API for analysis
- Export to JSON/Parquet

All CGW events (GATE_SELECTION, CGW_COMMIT, EXECUTION_COMPLETE) are
stored with full metadata for replay and learning.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class StoredEvent:
    """A stored CGW event."""
    
    event_id: int
    session_id: str
    event_type: str
    cycle_id: int
    timestamp: float
    payload: Dict[str, Any]
    
    def as_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class EventStoreConfig:
    """Configuration for the event store."""
    
    # Database path
    db_path: str = ""
    
    # Maximum events per session (for cleanup)
    max_events_per_session: int = 10000
    
    # Maximum session age in days
    max_session_age_days: int = 30
    
    # Enable WAL mode for better concurrency
    enable_wal: bool = True
    
    def __post_init__(self):
        if not self.db_path:
            home = os.path.expanduser("~")
            self.db_path = os.path.join(home, ".cgw", "events.db")


class CGWEventStore:
    """SQLite-backed persistent event store for CGW.
    
    Usage:
        store = CGWEventStore()
        
        # Store events
        store.record_event("session_123", "GATE_SELECTION", 1, {...})
        
        # Query events
        events = store.get_session_events("session_123")
        
        # Export
        store.export_session_json("session_123", "events.json")
    """
    
    def __init__(
        self,
        config: Optional[EventStoreConfig] = None,
    ):
        self.config = config or EventStoreConfig()
        self._db: Optional[sqlite3.Connection] = None
        self._init_db()
    
    def _init_db(self) -> None:
        """Initialize the SQLite database."""
        db_path = self.config.db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        self._db = sqlite3.connect(db_path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        
        if self.config.enable_wal:
            self._db.execute("PRAGMA journal_mode=WAL")
        
        self._db.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                goal TEXT,
                started_at REAL NOT NULL,
                ended_at REAL,
                status TEXT DEFAULT 'running',
                total_cycles INTEGER DEFAULT 0,
                metadata TEXT
            );
            
            CREATE TABLE IF NOT EXISTS events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                cycle_id INTEGER NOT NULL,
                timestamp REAL NOT NULL,
                payload TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            );
            
            CREATE INDEX IF NOT EXISTS idx_events_session 
                ON events(session_id);
            CREATE INDEX IF NOT EXISTS idx_events_type 
                ON events(event_type);
            CREATE INDEX IF NOT EXISTS idx_events_cycle 
                ON events(cycle_id);
            CREATE INDEX IF NOT EXISTS idx_sessions_started 
                ON sessions(started_at);
        """)
        self._db.commit()
    
    def start_session(
        self,
        session_id: str,
        goal: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Start a new session.
        
        Args:
            session_id: Unique session identifier.
            goal: Goal description for the session.
            metadata: Additional session metadata.
        """
        self._db.execute("""
            INSERT OR REPLACE INTO sessions 
            (session_id, goal, started_at, status, metadata)
            VALUES (?, ?, ?, 'running', ?)
        """, (
            session_id,
            goal,
            time.time(),
            json.dumps(metadata or {}),
        ))
        self._db.commit()
        logger.info(f"Started session: {session_id}")
    
    def end_session(
        self,
        session_id: str,
        status: str = "completed",
        total_cycles: int = 0,
    ) -> None:
        """End a session.
        
        Args:
            session_id: Session identifier.
            status: Final status (completed, aborted, failed).
            total_cycles: Total number of cycles executed.
        """
        self._db.execute("""
            UPDATE sessions SET 
                ended_at = ?,
                status = ?,
                total_cycles = ?
            WHERE session_id = ?
        """, (time.time(), status, total_cycles, session_id))
        self._db.commit()
        logger.info(f"Ended session: {session_id} ({status})")
    
    def record_event(
        self,
        session_id: str,
        event_type: str,
        cycle_id: int,
        payload: Dict[str, Any],
        timestamp: Optional[float] = None,
    ) -> int:
        """Record a CGW event.
        
        Args:
            session_id: Session this event belongs to.
            event_type: Event type (GATE_SELECTION, CGW_COMMIT, etc.)
            cycle_id: Cycle number.
            payload: Event payload data.
            timestamp: Event timestamp (defaults to now).
            
        Returns:
            Event ID of the stored event.
        """
        cursor = self._db.execute("""
            INSERT INTO events (session_id, event_type, cycle_id, timestamp, payload)
            VALUES (?, ?, ?, ?, ?)
        """, (
            session_id,
            event_type,
            cycle_id,
            timestamp or time.time(),
            json.dumps(payload, default=str),
        ))
        self._db.commit()
        return cursor.lastrowid
    
    def get_event(self, event_id: int) -> Optional[StoredEvent]:
        """Get a single event by ID."""
        cursor = self._db.execute(
            "SELECT * FROM events WHERE event_id = ?",
            (event_id,)
        )
        row = cursor.fetchone()
        if row:
            return self._row_to_event(row)
        return None
    
    def get_session_events(
        self,
        session_id: str,
        event_type: Optional[str] = None,
        start_cycle: Optional[int] = None,
        end_cycle: Optional[int] = None,
    ) -> List[StoredEvent]:
        """Get all events for a session.
        
        Args:
            session_id: Session identifier.
            event_type: Optional filter by event type.
            start_cycle: Optional start cycle filter.
            end_cycle: Optional end cycle filter.
            
        Returns:
            List of StoredEvent objects.
        """
        query = "SELECT * FROM events WHERE session_id = ?"
        params: List[Any] = [session_id]
        
        if event_type:
            query += " AND event_type = ?"
            params.append(event_type)
        
        if start_cycle is not None:
            query += " AND cycle_id >= ?"
            params.append(start_cycle)
        
        if end_cycle is not None:
            query += " AND cycle_id <= ?"
            params.append(end_cycle)
        
        query += " ORDER BY cycle_id, timestamp"
        
        cursor = self._db.execute(query, params)
        return [self._row_to_event(row) for row in cursor]
    
    def get_recent_sessions(
        self,
        limit: int = 10,
        status: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get recent sessions.
        
        Args:
            limit: Maximum number of sessions to return.
            status: Optional filter by status.
            
        Returns:
            List of session dictionaries.
        """
        query = "SELECT * FROM sessions"
        params: List[Any] = []
        
        if status:
            query += " WHERE status = ?"
            params.append(status)
        
        query += " ORDER BY started_at DESC LIMIT ?"
        params.append(limit)
        
        cursor = self._db.execute(query, params)
        return [dict(row) for row in cursor]
    
    def count_events(
        self,
        session_id: Optional[str] = None,
        event_type: Optional[str] = None,
    ) -> int:
        """Count events with optional filters."""
        query = "SELECT COUNT(*) FROM events WHERE 1=1"
        params: List[Any] = []
        
        if session_id:
            query += " AND session_id = ?"
            params.append(session_id)
        
        if event_type:
            query += " AND event_type = ?"
            params.append(event_type)
        
        cursor = self._db.execute(query, params)
        return cursor.fetchone()[0]
    
    def iter_events(
        self,
        session_id: str,
        batch_size: int = 100,
    ) -> Iterator[StoredEvent]:
        """Iterate through session events efficiently.
        
        Args:
            session_id: Session identifier.
            batch_size: Number of events per batch.
            
        Yields:
            StoredEvent objects.
        """
        offset = 0
        while True:
            cursor = self._db.execute("""
                SELECT * FROM events 
                WHERE session_id = ?
                ORDER BY cycle_id, timestamp
                LIMIT ? OFFSET ?
            """, (session_id, batch_size, offset))
            
            rows = cursor.fetchall()
            if not rows:
                break
            
            for row in rows:
                yield self._row_to_event(row)
            
            offset += len(rows)
            if len(rows) < batch_size:
                break
    
    def export_session_json(
        self,
        session_id: str,
        output_path: str,
    ) -> int:
        """Export session events to JSON file.
        
        Args:
            session_id: Session to export.
            output_path: Output file path.
            
        Returns:
            Number of events exported.
        """
        events = self.get_session_events(session_id)
        
        output = {
            "session_id": session_id,
            "event_count": len(events),
            "events": [e.as_dict() for e in events],
        }
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        
        logger.info(f"Exported {len(events)} events to {output_path}")
        return len(events)
    
    def export_session_jsonl(
        self,
        session_id: str,
        output_path: str,
    ) -> int:
        """Export session events to JSONL (one JSON per line).
        
        Args:
            session_id: Session to export.
            output_path: Output file path.
            
        Returns:
            Number of events exported.
        """
        count = 0
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            for event in self.iter_events(session_id):
                f.write(json.dumps(event.as_dict(), default=str) + "\n")
                count += 1
        
        logger.info(f"Exported {count} events to {output_path}")
        return count
    
    def cleanup_old_sessions(
        self,
        max_age_days: Optional[int] = None,
    ) -> int:
        """Delete old sessions and their events.
        
        Args:
            max_age_days: Maximum age in days (uses config default if None).
            
        Returns:
            Number of sessions deleted.
        """
        max_age_days = max_age_days or self.config.max_session_age_days
        cutoff = time.time() - (max_age_days * 24 * 60 * 60)
        
        # Get sessions to delete
        cursor = self._db.execute(
            "SELECT session_id FROM sessions WHERE started_at < ?",
            (cutoff,)
        )
        session_ids = [row[0] for row in cursor]
        
        if not session_ids:
            return 0
        
        # Delete events
        placeholders = ",".join(["?"] * len(session_ids))
        self._db.execute(
            f"DELETE FROM events WHERE session_id IN ({placeholders})",
            session_ids
        )
        
        # Delete sessions
        self._db.execute(
            f"DELETE FROM sessions WHERE session_id IN ({placeholders})",
            session_ids
        )
        
        self._db.commit()
        logger.info(f"Cleaned up {len(session_ids)} old sessions")
        return len(session_ids)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get store statistics."""
        cursor = self._db.execute("SELECT COUNT(*) FROM sessions")
        total_sessions = cursor.fetchone()[0]
        
        cursor = self._db.execute("SELECT COUNT(*) FROM events")
        total_events = cursor.fetchone()[0]
        
        cursor = self._db.execute("""
            SELECT event_type, COUNT(*) 
            FROM events 
            GROUP BY event_type
        """)
        events_by_type = dict(cursor.fetchall())
        
        return {
            "total_sessions": total_sessions,
            "total_events": total_events,
            "events_by_type": events_by_type,
        }
    
    def _row_to_event(self, row: sqlite3.Row) -> StoredEvent:
        """Convert a database row to StoredEvent."""
        return StoredEvent(
            event_id=row["event_id"],
            session_id=row["session_id"],
            event_type=row["event_type"],
            cycle_id=row["cycle_id"],
            timestamp=row["timestamp"],
            payload=json.loads(row["payload"]),
        )
    
    def close(self) -> None:
        """Close database connection."""
        if self._db:
            self._db.close()
            self._db = None


# === Singleton Access ===

_store_instance: Optional[CGWEventStore] = None


def get_event_store(
    db_path: Optional[str] = None,
) -> CGWEventStore:
    """Get or create the global event store instance."""
    global _store_instance
    
    if _store_instance is None:
        config = EventStoreConfig()
        if db_path:
            config.db_path = db_path
        _store_instance = CGWEventStore(config=config)
    
    return _store_instance


def reset_event_store() -> None:
    """Reset the global store instance (for testing)."""
    global _store_instance
    if _store_instance:
        _store_instance.close()
    _store_instance = None


# === Event Bus Integration ===

class EventStoreSubscriber:
    """Event bus subscriber that persists events to the store.
    
    Usage:
        from cgw_ssl_guard import SimpleEventBus
        
        bus = SimpleEventBus()
        subscriber = EventStoreSubscriber(session_id="my_session")
        subscriber.subscribe(bus)
        
        # Events are now automatically stored
    """
    
    def __init__(
        self,
        session_id: str,
        store: Optional[CGWEventStore] = None,
    ):
        self.session_id = session_id
        self.store = store or get_event_store()
        self._cycle_id = 0
    
    def subscribe(self, event_bus: Any) -> None:
        """Subscribe to all CGW events on the bus."""
        event_types = [
            "GATE_SELECTION",
            "CGW_COMMIT",
            "CGW_CLEAR",
            "EXECUTION_START",
            "EXECUTION_COMPLETE",
            "CYCLE_START",
            "CYCLE_END",
            "FORCED_SIGNAL",
        ]
        
        for event_type in event_types:
            handler = self._make_handler(event_type)
            event_bus.on(event_type, handler)
    
    def _make_handler(self, event_type: str):
        """Create a handler for an event type."""
        def handler(payload: Any):
            if isinstance(payload, dict) and 'cycle_id' in payload:
                self._cycle_id = payload['cycle_id']
            
            self.store.record_event(
                session_id=self.session_id,
                event_type=event_type,
                cycle_id=self._cycle_id,
                payload=payload if isinstance(payload, dict) else {"data": str(payload)},
            )
        return handler
    
    def set_cycle(self, cycle_id: int) -> None:
        """Manually set the current cycle ID."""
        self._cycle_id = cycle_id
