#!/usr/bin/env python3
"""
Database migration script to add the enable_qs column to modifiers_config table.
This script safely adds the new column if it doesn't exist yet.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from db_utils import get_db_connection
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def add_enable_qs_column():
    """
    Add enable_qs column to modifiers_config table.
    Uses IF NOT EXISTS to safely add the column without errors if it already exists.
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Add the column with default value TRUE
                logger.info("Adding enable_qs column to modifiers_config table...")
                cur.execute("""
                    ALTER TABLE modifiers_config 
                    ADD COLUMN IF NOT EXISTS enable_qs BOOLEAN NOT NULL DEFAULT TRUE;
                """)
                
                conn.commit()
                logger.info("✅ Successfully added enable_qs column!")
                
                # Verify the column was added
                cur.execute("""
                    SELECT column_name, data_type, column_default
                    FROM information_schema.columns
                    WHERE table_name = 'modifiers_config' AND column_name = 'enable_qs';
                """)
                result = cur.fetchone()
                
                if result:
                    logger.info(f"Column details: {result}")
                    logger.info("✅ Column verified successfully!")
                    return True
                else:
                    logger.error("❌ Column was not found after creation")
                    return False
                    
    except Exception as e:
        logger.error(f"❌ Failed to add enable_qs column: {e}")
        return False


if __name__ == "__main__":
    print("=" * 80)
    print("Database Migration: Add enable_qs column")
    print("=" * 80)
    print()
    
    success = add_enable_qs_column()
    
    if success:
        print()
        print("=" * 80)
        print("Migration completed successfully!")
        print("=" * 80)
        print()
        print("The enable_qs column has been added to the modifiers_config table.")
        print("Default value: TRUE (QS modifier is enabled by default for all insurances)")
        print()
        print("To disable QS for a specific insurance, update the enable_qs field to FALSE")
        print("using the API or database directly.")
    else:
        print()
        print("=" * 80)
        print("Migration failed!")
        print("=" * 80)
        sys.exit(1)

