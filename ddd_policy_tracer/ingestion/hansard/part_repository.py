"""Repository for speech parts in the Hansard dataset."""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Protocol

from pymongo import AsyncMongoClient


class SpeechPartType(str, Enum):  # noqa: UP042
    """The type of speech part."""

    INTERJECTION = "interjection"
    CONTINUATION = "continuation"
    SPEECH = "speech"


class ChamberType(str, Enum):  # noqa: UP042
    """Chamber of the AU Parliament where the speech part was delivered."""

    HOR_MAIN = "main"
    HOR_FEDERATION = "federation"
    UNKNOWN = "unknown"


class HouseType(str, Enum):  # noqa: UP042
    """House of the AU Parliament where the speech part was delivered."""

    SENATE = "senate"
    HOR = "hor"


class PartType(str, Enum):  # noqa: UP042
    """The type of part in the Hansard dataset."""

    SPEECH = "speech"
    FIRST_READING = "first_reading"
    DIVISION = "division"


@dataclass(frozen=True)
class SpeechPart:
    """Represent one speech part from the Hansard dataset."""

    id: str
    date: datetime
    bill_ids: list[str] | None
    house: HouseType
    chamber: ChamberType
    type: PartType
    debate_category: str
    debate_seq: int
    subdebate_1_title: str
    subdebate_1_info: str | None
    subdebate_1_seq: int | None
    subdebate_2_title: str | None
    subdebate_2_info: str | None
    subdebate_2_seq: int | None
    speech_id: str
    speech_seq: int
    part_seq: int
    talker_id: str
    speech_content: str
    speech_part_type: SpeechPartType


class AbstractPartRepository(Protocol):
    """Protocol for a repository of speech parts from the Hansard dataset."""

    async def list_all_speech_parts(self) -> list[SpeechPart]:
        """List all speech parts in the repository."""


class MongoPartRepository(AbstractPartRepository):
    """Repository implementation for speech parts using MongoDB as the backend."""

    def __init__(
        self,
        mongo_client: AsyncMongoClient,
        db_name: str = "hansard",
    ) -> None:
        """Initialize the repository with a MongoDB client and database name."""
        self.db = mongo_client[db_name]
        self.collection = self.db["parts"]

    async def list_all_speech_parts(self) -> list[SpeechPart]:
        """List all speech parts in the repository."""
        parts: list[SpeechPart] = []
        curs = self.collection.find({"type": "speech"})
        async for doc in curs:
            # drop _id field
            doc.pop("_id", None)
            parts.append(SpeechPart(**doc))
        return parts
