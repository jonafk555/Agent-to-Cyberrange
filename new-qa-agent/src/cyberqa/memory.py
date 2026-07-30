from __future__ import annotations

import json
from typing import Any

from .models import Evidence, Host


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
