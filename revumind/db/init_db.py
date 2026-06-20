"""
Database Initialization Script
==============================
Creates all tables mapped by the ORM models in the target database.
"""

import logging

from revumind.core.database import Base, engine

# Import models to ensure they are registered with Base metadata
from revumind.db import models

# Setup logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def init_database():
    """
    Creates all mapped tables in the configured database engine.
    """
    logger.info("Initializing RevuMind database tables...")
    try:
        # Create all tables
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables initialized successfully!")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise e


if __name__ == "__main__":
    init_database()
