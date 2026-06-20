"""
Database Connection & Session Factory
=====================================
Initializes the SQLAlchemy engine, session maker, and declarative base class.
Supports environment configurations via DATABASE_URL.
"""

import os

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

# Default to SQLite local database file for development
# In production, this will be overridden by the DATABASE_URL environment variable
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./revumind.db")

# SQLite adjustments for concurrent thread access
connect_args = {}
if DATABASE_URL.startswith("sqlite"):
    connect_args = {"check_same_thread": False}

# Initialize engine
engine = create_engine(
    DATABASE_URL,
    connect_args=connect_args,
    pool_pre_ping=True,  # Detect and recover from stale connections
)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Declarative base class for models
Base = declarative_base()


def get_db():
    """
    Context manager dependency for FastAPI endpoints and Celery tasks
    Yields a database session and closes it upon task completion.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
