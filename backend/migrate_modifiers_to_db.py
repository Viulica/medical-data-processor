#!/usr/bin/env python3
"""
Migration script to transfer modifiers from CSV to PostgreSQL database.
Run this once to populate the database with existing CSV data.
"""

import pandas as pd
from pathlib import Path
from db_utils import init_database, upsert_modifier, get_all_modifiers
import logging
import time
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def migrate_csv_to_db():
    """
    Read modifiers_definition.csv and populate the database.
    """
    # Initialize database schema
    logger.info("Initializing database schema...")
    if not init_database():
        logger.error("Failed to initialize database!")
        return False
    
    # Find CSV file
    script_dir = Path(__file__).parent
    csv_path = script_dir / "modifiers" / "modifiers_definition.csv"
    
    if not csv_path.exists():
        logger.error(f"CSV file not found: {csv_path}")
        return False
    
    # Read CSV
    logger.info(f"Reading CSV file: {csv_path}")
    try:
        df = pd.read_csv(csv_path, dtype=str)
        logger.info(f"Found {len(df)} rows in CSV")
    except Exception as e:
        logger.error(f"Failed to read CSV: {e}")
        return False
    
    # Validate columns
    required_columns = ['MedNet Code', 'Medicare Modifiers', 'Bill Medical Direction']
    for col in required_columns:
        if col not in df.columns:
            logger.error(f"Missing required column: {col}")
            return False
    
    # Check how many records already exist
    existing_modifiers = get_all_modifiers()
    logger.info(f"Database currently has {len(existing_modifiers)} records")
    logger.info(f"CSV has {len(df)} records to insert/update")
    
    # Insert data into database with batch commits and retry logic
    success_count = 0
    fail_count = 0
    batch_size = 500
    max_retries = 3
    
    logger.info("Starting bulk insert with retry logic...")
    
    # Use a single connection for all inserts
    from db_utils import get_db_connection
    import psycopg2
    
    def insert_with_retry(conn, cur, mednet_code, medicare_modifiers, bill_medical_direction, retries=0):
        """Insert with retry logic for connection issues"""
        try:
            cur.execute("""
                INSERT INTO modifiers_config (mednet_code, medicare_modifiers, bill_medical_direction, updated_at)
                VALUES (%s, %s, %s, CURRENT_TIMESTAMP)
                ON CONFLICT (mednet_code) 
                DO UPDATE SET 
                    medicare_modifiers = EXCLUDED.medicare_modifiers,
                    bill_medical_direction = EXCLUDED.bill_medical_direction,
                    updated_at = CURRENT_TIMESTAMP
            """, (mednet_code, medicare_modifiers, bill_medical_direction))
            return True
        except (psycopg2.OperationalError, psycopg2.InterfaceError) as e:
            if retries < max_retries:
                logger.warning(f"Connection error on {mednet_code}, retry {retries+1}/{max_retries}")
                time.sleep(2 ** retries)  # Exponential backoff
                # Reconnect
                try:
                    conn.close()
                except:
                    pass
                return None  # Signal to reconnect
            else:
                logger.error(f"Failed after {max_retries} retries: {mednet_code}")
                return False
        except Exception as e:
            logger.error(f"Unexpected error on {mednet_code}: {e}")
            return False
    
    conn = None
    cur = None
    retry_count = 0
    
    for idx, row in df.iterrows():
        mednet_code = str(row['MedNet Code']).strip()
        medicare_modifiers = str(row['Medicare Modifiers']).strip().upper() == 'YES'
        bill_medical_direction = str(row['Bill Medical Direction']).strip().upper() == 'YES'
        
        # Create connection if needed
        if conn is None:
            try:
                conn = psycopg2.connect(os.environ.get(
                    'DATABASE_URL',
                    'postgresql://postgres:YISwNRCXxndHsFmucFxwFhEXJrHxQaEC@centerbeam.proxy.rlwy.net:33249/railway'
                ))
                cur = conn.cursor()
                logger.info(f"Reconnected to database (processed: {success_count})")
            except Exception as e:
                logger.error(f"Failed to connect: {e}")
                time.sleep(5)
                continue
        
        result = insert_with_retry(conn, cur, mednet_code, medicare_modifiers, bill_medical_direction, retry_count)
        
        if result is True:
            success_count += 1
            retry_count = 0
            
            # Commit in batches
            if success_count % batch_size == 0:
                try:
                    conn.commit()
                    logger.info(f"✅ Committed batch: {success_count}/{len(df)} records ({success_count/len(df)*100:.1f}%)")
                except Exception as e:
                    logger.error(f"Commit failed: {e}")
                    conn = None
                    cur = None
        elif result is None:
            # Need to reconnect
            conn = None
            cur = None
            retry_count += 1
        else:
            fail_count += 1
            retry_count = 0
    
    # Final commit
    if conn:
        try:
            conn.commit()
            conn.close()
            logger.info(f"✅ Final commit: {success_count} records")
        except Exception as e:
            logger.error(f"Final commit failed: {e}")
    
    logger.info(f"✅ Migration complete!")
    logger.info(f"   Successfully migrated: {success_count}")
    logger.info(f"   Failed: {fail_count}")
    
    # Verify
    all_modifiers = get_all_modifiers()
    logger.info(f"   Total in database: {len(all_modifiers)}")
    
    return True


if __name__ == "__main__":
    print("=" * 80)
    print("Migrating Modifiers from CSV to PostgreSQL")
    print("=" * 80)
    
    success = migrate_csv_to_db()
    
    if success:
        print("\n✅ Migration completed successfully!")
    else:
        print("\n❌ Migration failed!")

