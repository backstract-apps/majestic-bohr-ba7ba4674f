"""
RabbitMQ Queue Adapter
"""
import json
from typing import Any, Callable, Dict, Optional

import aio_pika
from loguru import logger

from workers.adapters.base_adapter import QueueAdapter


class RabbitMQAdapter(QueueAdapter):
    """RabbitMQ implementation of QueueAdapter"""

    def __init__(self):
        self.connection = None
        self.channel = None
        self.consumers = {}  # queue_name -> consumer_tag
        self.handlers = {}  # queue_name -> handler

    async def connect(self, credentials: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> None:
        """Connect to RabbitMQ with SSL support"""
        try:
            import urllib.parse
            import ssl
            
            if credentials.get("url"):
                url = credentials["url"]
                use_ssl = url.startswith("amqps://")
            else:
                # Build URL from credentials
                host = credentials.get('host', 'localhost')
                port = credentials.get('port', 5672)
                username = credentials.get('username', 'guest')
                password = credentials.get('password', 'guest')
                vhost = credentials.get('vhost', '/')
                use_ssl = credentials.get('ssl', False)
                
                # Strip leading slash from vhost (CloudAMQP vhosts like '/pcrbjpql' should become 'pcrbjpql')
                vhost_clean = vhost.lstrip('/') if vhost else ''
                vhost_encoded = urllib.parse.quote(vhost_clean, safe='')
                protocol = "amqps" if use_ssl else "amqp"
                url = f"{protocol}://{username}:{password}@{host}:{port}/{vhost_encoded}"
                url_masked = f"{protocol}://{username}:***@{host}:{port}/{vhost_encoded}"
                logger.info(f"[WORKER_ADAPTER] Connecting to RabbitMQ: {url_masked}, ssl={use_ssl}")
            
            # Create SSL context for secure connections
            if use_ssl:
                ssl_context = ssl.create_default_context()
                logger.info(f"[WORKER_ADAPTER] Using SSL/TLS connection with certificate verification")
                self.connection = await aio_pika.connect_robust(url, ssl_context=ssl_context)
            else:
                self.connection = await aio_pika.connect_robust(url)
            
            self.channel = await self.connection.channel()
            
            if config and config.get("prefetch_count"):
                await self.channel.set_qos(prefetch_count=config["prefetch_count"])
            
            logger.info("[WORKER_ADAPTER] Connected to RabbitMQ successfully")
        except Exception as e:
            logger.error(f"[WORKER_ADAPTER] Failed to connect to RabbitMQ: {e}")
            raise

    async def disconnect(self) -> None:
        """Disconnect from RabbitMQ"""
        try:
            # Unsubscribe from all queues
            for queue_name in list(self.consumers.keys()):
                await self.unsubscribe(queue_name)
            
            if self.channel:
                await self.channel.close()
            if self.connection:
                await self.connection.close()
            
            logger.info("Disconnected from RabbitMQ")
        except Exception as e:
            logger.error(f"Error disconnecting from RabbitMQ: {e}")

    async def subscribe(
        self, queue_name: str, handler: Callable[[Dict[str, Any]], None], config: Optional[Dict[str, Any]] = None
    ) -> None:
        """Subscribe to a RabbitMQ queue"""
        try:
            # Declare queue
            queue = await self.channel.declare_queue(
                queue_name,
                durable=config.get("durable", True) if config else True
            )
            
            # Store handler
            self.handlers[queue_name] = handler
            
            # Create consumer
            async def message_handler(message: aio_pika.IncomingMessage):
                async with message.process():
                    try:
                        body = json.loads(message.body.decode())
                        await handler(body)
                    except Exception as e:
                        logger.error(f"Error processing message from {queue_name}: {e}")
                        # Message will be nacked and potentially requeued
            
            consumer_tag = await queue.consume(message_handler)
            self.consumers[queue_name] = consumer_tag
            
            logger.info(f"Subscribed to RabbitMQ queue: {queue_name}")
        except Exception as e:
            logger.error(f"Failed to subscribe to queue {queue_name}: {e}")
            raise

    async def unsubscribe(self, queue_name: str) -> None:
        """Unsubscribe from a RabbitMQ queue"""
        try:
            if queue_name in self.consumers:
                consumer_tag = self.consumers[queue_name]
                await self.channel.cancel(consumer_tag)
                del self.consumers[queue_name]
                del self.handlers[queue_name]
                logger.info(f"Unsubscribed from RabbitMQ queue: {queue_name}")
        except Exception as e:
            logger.error(f"Error unsubscribing from queue {queue_name}: {e}")

    async def publish(
        self, queue_name: str, message: Dict[str, Any], config: Optional[Dict[str, Any]] = None
    ) -> None:
        """Publish a message to RabbitMQ queue"""
        try:
            queue = await self.channel.declare_queue(
                queue_name,
                durable=config.get("durable", True) if config else True
            )
            
            await self.channel.default_exchange.publish(
                aio_pika.Message(
                    json.dumps(message).encode(),
                    delivery_mode=aio_pika.DeliveryMode.PERSISTENT if config.get("persistent", True) else aio_pika.DeliveryMode.NOT_PERSISTENT
                ),
                routing_key=queue_name
            )
            
            logger.debug(f"Published message to RabbitMQ queue: {queue_name}")
        except Exception as e:
            logger.error(f"Failed to publish message to queue {queue_name}: {e}")
            raise