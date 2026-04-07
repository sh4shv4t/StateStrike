from __future__ import annotations

"""Database setup for the StateStrike honeypot.

Theory:
    A local SQLite backend keeps the demo deterministic and lightweight while
    preserving enough statefulness for multi-step fuzzing trajectories.
"""

import logging
import os
from collections.abc import Generator

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker

load_dotenv()

LOGGER = logging.getLogger(__name__)

DATABASE_FILE = os.getenv("DATABASE_FILE", "statestrike.db")
DATABASE_URL = f"sqlite:///{DATABASE_FILE}"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    """Yield a SQLAlchemy session for request-scoped DB access.

    Yields:
        An open SQLAlchemy Session object.

    Raises:
        RuntimeError: If session creation fails.
    """

    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create database schema if tables do not yet exist.

    Raises:
        Exception: Propagates SQLAlchemy creation errors.
    """

    from honeypot import models  # Local import avoids circular import at module load.

    Base.metadata.create_all(bind=engine)
    LOGGER.info("Initialized SQLite schema at %s", DATABASE_URL)
