"""Repository for talkers in the Hansard dataset."""

from typing import Protocol

from pydantic import BaseModel
from pymongo import AsyncMongoClient


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

    async def list_talkers(self) -> list[Talker]:  # type:ignore[return]
        """List all talkers in the repository."""


class MongoTalkerRepository(AbstractTalkerRepository):
    """Repository implementation for talkers using MongoDB as the backend."""

    def __init__(
        self,
        mongo_client: AsyncMongoClient,
        db_name: str = "hansard",
    ) -> None:
        """Initialize the repository with a MongoDB client and database name."""
        self.db = mongo_client[db_name]
        self.collection = self.db["talkers"]

    async def list_talkers(self) -> list[Talker]:
        """List all talkers in the repository."""
        talkers: list[Talker] = []
        async for doc in self.collection.find():
            doc.pop("_id", None)  # drop _id field
            doc.pop("bill_divisiveness", None)
            doc.pop("divisiveness", None)
            talker = Talker(**doc)
            talkers.append(talker)
        return talkers
