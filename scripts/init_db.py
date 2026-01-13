"""
Database initialization script for BOT GPT.

This script:
1. Creates all database tables
2. Optionally seeds sample data for testing
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.database import db_manager, User, Conversation, Message, Document
from src.core.logger import app_logger as logger


async def init_database(drop_existing: bool = False):
    """
    Initialize the database.
    
    Args:
        drop_existing: If True, drop all existing tables first
    """
    try:
        if drop_existing:
            logger.warning("Dropping all existing tables...")
            await db_manager.drop_tables()
        
        logger.info("Creating database tables...")
        await db_manager.create_tables()
        
        logger.info("Database initialization complete!")
    
    except Exception as e:
        logger.error(f"Error initializing database: {e}", exc_info=True)
        raise


async def seed_sample_data():
    """Seed the database with sample data for testing"""
    try:
        logger.info("Seeding sample data...")
        
        async with db_manager.async_session_maker() as session:
            # Create sample user
            user = User(
                username="Sahil Khan",
                email="mr.sahil786khan@gmail.com",
            )
            session.add(user)
            await session.flush()
            
            logger.info(f"Created sample user: {user.id}")
            
            await session.commit()
        
        logger.info("Sample data seeded successfully!")
    
    except Exception as e:
        logger.error(f"Error seeding data: {e}", exc_info=True)
        raise


async def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Initialize BOT GPT database")
    parser.add_argument(
        "--drop",
        action="store_true",
        help="Drop existing tables before creating new ones"
    )
    parser.add_argument(
        "--seed",
        action="store_true",
        help="Seed sample data after initialization"
    )
    
    args = parser.parse_args()
    
    # Initialize database
    await init_database(drop_existing=args.drop)
    
    # Seed data if requested
    if args.seed:
        await seed_sample_data()
    
    # Close database connection
    await db_manager.close()


if __name__ == "__main__":
    asyncio.run(main())
