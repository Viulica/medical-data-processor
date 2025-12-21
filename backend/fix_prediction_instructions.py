#!/usr/bin/env python3
"""
Script to diagnose and fix prediction_instructions table issues
"""

import sys
import os

# Add the backend directory to path so we can import modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from db_utils import get_db_connection, init_database
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_and_fix():
    """Check if prediction_instructions table exists and create it if needed"""
    try:
        logger.info("Connecting to database...")
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Check if table exists
                cur.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'prediction_instructions'
                    );
                """)
                exists = cur.fetchone()[0]
                
                if exists:
                    logger.info("✅ prediction_instructions table exists")
                    
                    # Check table structure
                    cur.execute("""
                        SELECT column_name, data_type 
                        FROM information_schema.columns 
                        WHERE table_name = 'prediction_instructions'
                        ORDER BY ordinal_position;
                    """)
                    columns = cur.fetchall()
                    logger.info("Table columns:")
                    for col in columns:
                        logger.info(f"  - {col[0]}: {col[1]}")
                    
                    # Count existing records
                    cur.execute("SELECT COUNT(*) FROM prediction_instructions")
                    count = cur.fetchone()[0]
                    logger.info(f"Total records: {count}")
                    
                    # Show records by type
                    cur.execute("""
                        SELECT instruction_type, COUNT(*) 
                        FROM prediction_instructions 
                        GROUP BY instruction_type
                    """)
                    type_counts = cur.fetchall()
                    if type_counts:
                        logger.info("Records by type:")
                        for type_name, type_count in type_counts:
                            logger.info(f"  - {type_name}: {type_count}")
                    
                else:
                    logger.warning("❌ prediction_instructions table does not exist")
                    logger.info("Running init_database() to create tables...")
                    init_database()
                    logger.info("✅ Database initialized")
                
                # Test the query that's failing
                logger.info("\nTesting the problematic query...")
                cur.execute("""
                    SELECT id, name, description, instruction_type, created_at, updated_at
                    FROM prediction_instructions
                    WHERE instruction_type = %s
                    ORDER BY instruction_type, name
                    LIMIT 100 OFFSET 0
                """, ('cpt',))
                results = cur.fetchall()
                logger.info(f"✅ CPT query successful, returned {len(results)} records")
                
                cur.execute("""
                    SELECT id, name, description, instruction_type, created_at, updated_at
                    FROM prediction_instructions
                    WHERE instruction_type = %s
                    ORDER BY instruction_type, name
                    LIMIT 100 OFFSET 0
                """, ('icd',))
                results = cur.fetchall()
                logger.info(f"✅ ICD query successful, returned {len(results)} records")
                
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    logger.info("\n✅ All checks passed!")
    return True

if __name__ == "__main__":
    success = check_and_fix()
    sys.exit(0 if success else 1)

