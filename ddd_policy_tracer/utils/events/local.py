import asyncio
import logging
from collections import defaultdict
from typing import Any, NoReturn
from uuid import uuid4

from utils.logger import CustomLoggingAdapter

from .base import AbstractSubscriber, Topic


class LocalPublisher:
    def __init__(self):
        self._id = uuid4()
        self._latest_event: dict[str, Any] | None = None
        self.subscribers: dict[Topic, list["LocalSubscriber"]] = defaultdict(
            list
        )
        self._logger = CustomLoggingAdapter(
            logging.getLogger(__name__),
            {"ctx": self},
        )

    def __repr__(self):
        return f"pub-{self._id}"

    def register(self, subscriber: "LocalSubscriber", topic: Topic):
        self._logger.debug(f"Registering {subscriber} for {topic}")
        self.subscribers[topic].append(subscriber)

    def publish(self, event: dict[str, Any]):
        self._latest_event = event
        try:
            topic = event["topic"]
        except KeyError:
            raise ValueError("Event must have a topic")
        self._logger.debug(f"Publishing event: {event}")
        subscribers = self.subscribers.get(topic, [])
        if len(subscribers) == 0:
            self._logger.debug(f"No subscribers for topic: {topic}")
        for subscriber in self.subscribers[topic]:
            self._logger.debug(f"Notifying {subscriber}")
            subscriber.notify(event)


class LocalSubscriber(AbstractSubscriber):
    def __init__(self, publisher: LocalPublisher):
        self.publisher = publisher
        self.queue = asyncio.Queue[dict[str, Any]]()
        super().__init__()

    def notify(self, event: dict[str, Any]):
        self.queue.put_nowait(event)

    async def subscribe(self, topics: list[str]):
        """Subscribe to a topic.

        a) Register the subscriber with the broker
        b) Start the worker if it's not already running

        Args:
            topic (str): The topic to subscribe to
        """
        for topic in topics:
            if topic not in self._topics:
                self._logger.debug(f"Subscribing to {topic}")
                self.publisher.register(self, topic)
                self._topics.add(topic)
        if self._worker is None:
            self._logger.debug("Starting worker coro")
            self._worker = asyncio.create_task(self.get_message_loop())
            self._worker.add_done_callback(self._listening_task_done_callback)

    async def get_message_loop(self) -> NoReturn:
        """Loop that takes an item from the queue and handles it."""
        self._logger.debug("loop started")
        while True:
            event = await self.queue.get()
            self._logger.debug(f"Received event: {event}")
            try:
                self._logger.debug(f"Handling event: {event}")
                await self.handle(event)
            except asyncio.CancelledError:
                raise
            except BaseException as e:
                self._logger.exception(e)
            finally:
                self._logger.debug(f"Done handling event: {event}")
                self.queue.task_done()
