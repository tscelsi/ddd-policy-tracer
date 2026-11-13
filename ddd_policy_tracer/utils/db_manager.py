"""Module to manage the SQLite database connection and vector extension loading."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import aiosqlite
import sqlite_vec

from ddd_policy_tracer.paths import DATA_DIR

DB_PATH = DATA_DIR / "new2.db"

ASYNC_CONN: aiosqlite.Connection | None = None


async def get_db_async() -> aiosqlite.Connection:
    """Helper function to get an async DB instance."""
    global ASYNC_CONN
    if ASYNC_CONN is not None:
        return ASYNC_CONN
    ASYNC_CONN = await aiosqlite.connect(DB_PATH)
    await ASYNC_CONN.enable_load_extension(True)
    await ASYNC_CONN.load_extension(sqlite_vec.loadable_path())
    await ASYNC_CONN.enable_load_extension(False)
    return ASYNC_CONN


@asynccontextmanager
async def with_transaction() -> AsyncGenerator[aiosqlite.Connection, None]:
    """Async context manager to provide the DB connection and commit properly."""
    conn = await get_db_async()
    try:
        yield conn
    finally:
        await conn.commit()


class DbManager:
    """Manage the SQLite database connection and vector extension loading."""

    _instances: dict[str, "DbManager"] = {}

    @staticmethod
    def get_instance(db_path: str | None = None) -> "DbManager":
        """Get a singleton instance of the DbManager."""
        key = db_path or str(DB_PATH)
        if key not in DbManager._instances:
            DbManager._instances[key] = DbManager(db_path=db_path)
        return DbManager._instances[key]

    def __init__(self, db_path: str | None = None) -> None:
        """Initialize the DbManager with an optional custom database path."""
        self.db_path = db_path or DB_PATH
        self._conn: aiosqlite.Connection | None = None

    async def get_db(self) -> aiosqlite.Connection:
        """Get the async DB connection."""
        if self._conn is not None:
            return self._conn
        self._conn = await aiosqlite.connect(self.db_path)
        await self._conn.enable_load_extension(True)
        await self._conn.load_extension(sqlite_vec.loadable_path())
        await self._conn.enable_load_extension(False)
        return self._conn

    @asynccontextmanager
    async def with_transaction(self) -> AsyncGenerator[aiosqlite.Connection, None]:
        """Async context manager to provide the DB connection and commit properly."""
        conn = await self.get_db()
        try:
            yield conn
        finally:
            await conn.commit()
