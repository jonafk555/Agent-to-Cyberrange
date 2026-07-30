from __future__ import annotations

import json
from typing import Awaitable, Callable

from .models import Event


EventHandler = Callable[[Event], Awaitable[None]]


class EventBus:
    """RabbitMQ-compatible publisher with an in-process mode for local runs."""
    def __init__(self, url: str | None = None):
        self.url, self.handlers = url, {}

    def subscribe(self, event_type: str, handler: EventHandler) -> None:
        self.handlers.setdefault(event_type, []).append(handler)

    async def publish(self, event: Event) -> None:
        for handler in self.handlers.get(event.type, []) + self.handlers.get("*", []):
            await handler(event)
        if self.url:
            try:
                import aio_pika
                connection = await aio_pika.connect_robust(self.url)
                async with connection:
                    channel = await connection.channel()
                    exchange = await channel.declare_exchange("cyberqa.events", aio_pika.ExchangeType.TOPIC)
                    await exchange.publish(aio_pika.Message(json.dumps(event.model_dump(mode="json")).encode()), routing_key=event.type)
            except Exception as exc:  # transport failures are facts for the run, not control decisions
                raise RuntimeError(f"RabbitMQ publish failed: {exc}") from exc
