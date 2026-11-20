"""Module for managing the repository of bills in the Hansard dataset."""

from typing import Protocol

from pydantic import BaseModel
from pymongo import AsyncMongoClient
from pymongo.operations import UpdateOne


class Bill(BaseModel):
    """<a href="r7349" type="Bill">
        <p class="HPS-SubDebate" style="direction:ltr;unicode-bidi:normal;">
            <span class="HPS-SubDebate">Commonwealth Workplace Protection Orders Bill 2025</span>
        </p>
    </a>
    """

    id: str
    name: str


class AbstractBillRepository(Protocol):
    """Port for a repository of bills in the Hansard dataset."""

    async def save_bills(self, bills: list[Bill]) -> None:
        """Save a list of bills to the repository."""
        ...

    async def list_bills(self) -> list[Bill]:
        """List all bills in the repository."""
        ...

    async def get_bill_by_id(self, id: str) -> Bill | None:
        """Get a bill by its ID from the repository."""
        ...


class MongoBillRepository(AbstractBillRepository):
    def __init__(self, client: AsyncMongoClient, db_name: str = "hansard"):
        self.client = client
        self.db = client[db_name]
        self.collection = self.db["bills"]

    async def save_bills(self, bills: list[Bill]) -> None:
        """Save a list of bills to the repository."""
        operations: list[UpdateOne] = []
        for bill in bills:
            operations.append(
                UpdateOne(
                    {"id": bill.id},
                    {"$set": bill.model_dump()},
                    upsert=True,
                ),
            )
        if operations:
            await self.collection.bulk_write(operations)

    async def list_bills(self) -> list[Bill]:
        """List all bills in the repository."""
        bills: list[Bill] = []
        async for doc in self.collection.find():
            bill = Bill.model_validate(doc)
            bills.append(bill)
        return bills

    async def get_bill_by_id(self, id: str) -> Bill | None:
        """Get a bill by its ID from the repository."""
        doc = await self.collection.find_one({"id": id})
        if doc:
            return Bill.model_validate(doc)
        return None
