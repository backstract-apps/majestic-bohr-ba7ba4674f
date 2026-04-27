"""
Base Queue Adapter - Abstract interface for message queue adapters
"""
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional


class QueueAdapter(ABC):
    """Abstract base class for message queue adapters"""

    @abstractmethod
    async def connect(self, credentials: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> None:
        """Connect to the message queue"""
        raise NotImplementedError

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the message queue"""
        raise NotImplementedError

    @abstractmethod
    async def subscribe(
        self, queue_name: str, handler: Callable[[Dict[str, Any]], None], config: Optional[Dict[str, Any]] = None
    ) -> None:
        """Subscribe to a queue and register a handler"""
        raise NotImplementedError

    @abstractmethod
    async def unsubscribe(self, queue_name: str) -> None:
        """Unsubscribe from a queue"""
        raise NotImplementedError

    @abstractmethod
    async def publish(
        self, queue_name: str, message: Dict[str, Any], config: Optional[Dict[str, Any]] = None
    ) -> None:
        """Publish a message to a queue"""
        raise NotImplementedError


def create_queue_adapter(queue_type: str) -> QueueAdapter:
    """
    Factory function to create the appropriate queue adapter.
    
    Currently supports:
    - rabbitmq: RabbitMQ message queue
    """
    queue_type_lower = queue_type.lower()
    
    if queue_type_lower == "rabbitmq":
        from workers.adapters.rabbitmq_adapter import RabbitMQAdapter
        return RabbitMQAdapter()
    else:
        raise ValueError(
            f"Unsupported queue type: {queue_type}. "
            f"Currently only 'rabbitmq' is supported."
        )