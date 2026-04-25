"""Database connection and session management"""

from typing import AsyncGenerator
from sqlmodel import SQLModel, create_engine
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
import aiosqlite

# Global engine reference
_engine = None
_async_session_maker = None


async def init_db(database_url: str):
    """Initialize database connection and create tables"""
    global _engine, _async_session_maker

    # Convert sqlite:// to sqlite+aiosqlite://
    if database_url.startswith("sqlite:///"):
        async_url = database_url.replace("sqlite:///", "sqlite+aiosqlite:///")
    else:
        async_url = database_url

    _engine = create_async_engine(async_url, echo=False)
    _async_session_maker = sessionmaker(
        _engine, class_=AsyncSession, expire_on_commit=False
    )

    # Create tables
    async with _engine.begin() as conn:
        from .models import OHLCVCandle, NewsArticle, SentimentSnapshot, AnalysisCache
        await conn.run_sync(SQLModel.metadata.create_all)


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Get async database session"""
    if _async_session_maker is None:
        raise RuntimeError("Database not initialized. Call init_db first.")

    async with _async_session_maker() as session:
        yield session


async def close_db():
    """Close database connection"""
    global _engine
    if _engine:
        await _engine.dispose()
