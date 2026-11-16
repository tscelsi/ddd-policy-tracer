"""Repository for talkers in the Hansard dataset."""

from collections.abc import Mapping
from typing import Any, Protocol

from pydantic import BaseModel
from pymongo import AsyncMongoClient, MongoClient


class Talker(BaseModel):
    """Represent one talker from the Hansard dataset.

    Looks like:
    <talker>
        <page.no>15</page.no>
        <time.stamp/>
        <name role="metadata">Plibersek, Tanya Joan MP</name>
        <name.id>83M</name.id>
        <electorate>Sydney</electorate>
        <party>ALP</party>
        <in.gov/>
        <first.speech/>
    </talker>
    """

    id: str | None
    name: str
    first_name: str
    last_name: str
    electorate: str | None
    party: str | None


class AbstractTalkerRepository(Protocol):
    """Protocol for a repository of talkers in the Hansard dataset."""

    async def list_talkers(self) -> list[Talker]:
        """List all talkers in the repository."""


class MongoTalkerRepository(AbstractTalkerRepository):
    """Repository implementation for talkers using MongoDB as the backend."""

    def __init__(
        self,
        mongo_client: AsyncMongoClient[Any],
        db_name: str = "hansard",
    ) -> None:
        """Initialize the repository with a MongoDB client and database name."""
        self.db = mongo_client[db_name]
        self.collection = self.db["talkers"]

    async def list_talkers(self) -> list[Talker]:
        """List all talkers in the repository."""
        talkers: list[Talker] = []
        async for doc in self.collection.find():
            talkers.append(_talker_from_document(doc))
        return talkers


class MongoTalkerCatalog:
    """Read the Hansard speaker catalog synchronously for completion-time resolution."""

    def __init__(
        self,
        mongo_client: MongoClient[dict[str, Any]],
        db_name: str = "hansard",
    ) -> None:
        """Bind the lazy catalog adapter to the configured Mongo collection."""
        self._collection = mongo_client[db_name]["talkers"]

    def list_talkers(self) -> list[Talker]:
        """Load all speaker records when a completion first requests name resolution."""
        return [_talker_from_document(document) for document in self._collection.find()]


def _talker_from_document(document: Mapping[str, Any]) -> Talker:
    """Translate a Mongo record into the Hansard talker contract."""
    data = {
        key: value
        for key, value in document.items()
        if key not in {"_id", "bill_divisiveness", "divisiveness"}
    }
    return Talker(**data)
