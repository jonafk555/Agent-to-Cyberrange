from __future__ import annotations

import json
import hashlib
import os
import sqlite3
import time
from pathlib import Path
from typing import Any

from .models import Evidence, Host


class ObservationStore:
    """Durable observation cache keyed by the effective command identity."""

    def __init__(self, path: str | None = None, ttl_seconds: int | None = None):
        configured_path = path or os.getenv(
            "CYBERQA_OBSERVATION_DB", ".cyberqa/observations.sqlite3"
        )
        self.path = configured_path
        self.ttl_seconds = (
            int(os.getenv("CYBERQA_OBSERVATION_TTL_SECONDS", "0"))
            if ttl_seconds is None else ttl_seconds
        )
        if configured_path != ":memory:":
            Path(configured_path).expanduser().parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(
            str(Path(configured_path).expanduser()) if configured_path != ":memory:" else configured_path,
            timeout=10,
        )
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS observations (
                signature TEXT PRIMARY KEY,
                payload TEXT NOT NULL,
                stored_at REAL NOT NULL
            )
            """
        )
        self.connection.commit()

    @staticmethod
    def signature(tool: str, target: str, action: str, parameters: Any = None,
                  namespace: str = "") -> str:
        payload = json.dumps({"namespace": namespace, "tool": tool, "target": target,
                              "action": action, "parameters": parameters or {}},
                             sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()[:20]

    def get(self, signature: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT payload, stored_at FROM observations WHERE signature = ?", (signature,)
        ).fetchone()
        if not row:
            return None
        if self.ttl_seconds > 0 and time.time() - float(row[1]) > self.ttl_seconds:
            self.connection.execute("DELETE FROM observations WHERE signature = ?", (signature,))
            self.connection.commit()
            return None
        return json.loads(row[0])

    def put(self, signature: str, result: dict[str, Any]) -> None:
        self.connection.execute(
            "INSERT OR REPLACE INTO observations(signature, payload, stored_at) VALUES (?, ?, ?)",
            (signature, json.dumps(result, ensure_ascii=False, default=str), time.time()),
        )
        self.connection.commit()

    def clear(self, signature: str | None = None) -> int:
        """Delete cached observations and return the number of removed rows."""

        if signature:
            cursor = self.connection.execute(
                "DELETE FROM observations WHERE signature = ?", (signature,)
            )
        else:
            cursor = self.connection.execute("DELETE FROM observations")
        self.connection.commit()
        return int(cursor.rowcount)

    def close(self) -> None:
        self.connection.close()


class RedisMemory:
    def __init__(self, url: str | None = None):
        self.url, self.local = url, {}

    async def put(self, key: str, value: Any) -> None:
        if self.url:
            from redis.asyncio import Redis
            client = Redis.from_url(self.url)
            await client.set(key, json.dumps(value, default=str))
            await client.close()
        else:
            self.local[key] = value

    async def get(self, key: str) -> Any:
        if self.url:
            from redis.asyncio import Redis
            client = Redis.from_url(self.url)
            value = await client.get(key)
            await client.close()
            return json.loads(value) if value else None
        return self.local.get(key)


class KnowledgeGraphRepository:
    """Neo4j-compatible projection; Cypher is kept explicit for review/auditing."""
    def __init__(self, uri: str | None = None, user: str | None = None, password: str | None = None):
        self.driver = None
        if uri:
            from neo4j import AsyncGraphDatabase
            self.driver = AsyncGraphDatabase.driver(uri, auth=(user, password))

    async def upsert_observation(self, host: Host, evidence: Evidence) -> None:
        if not self.driver:
            return
        query = """
        MERGE (h:Host {name:$name}) SET h.platform=$platform, h.address=$address
        MERGE (s:Service {host:$name, protocol:$protocol, port:$port})
        MERGE (h)-[:EXPOSES]->(s)
        SET s.running=$running, s.reachable=$reachable, s.functional=$functional
        """
        async with self.driver.session() as session:
            for service in host.services:
                await session.run(query, name=host.name, platform=host.platform, address=host.address,
                                  protocol=service.protocol.value, port=service.port,
                                  running=service.running, reachable=service.reachable,
                                  functional=service.functional)
