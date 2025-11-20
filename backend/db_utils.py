#!/usr/bin/env python3
"""
Database utility functions for connecting to PostgreSQL.
Handles modifiers configuration storage.
"""

import os
import psycopg2
from psycopg2.extras import RealDictCursor
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)

# Database connection string from environment
DATABASE_URL = os.environ.get(
    'DATABASE_URL',
    'postgresql://postgres:YISwNRCXxndHsFmucFxwFhEXJrHxQaEC@centerbeam.proxy.rlwy.net:33249/railway'
)


@contextmanager
def get_db_connection():
    """
    Context manager for database connections.
    Automatically handles connection closing.
    """
    conn = None
    try:
        conn = psycopg2.connect(DATABASE_URL)
        yield conn
        conn.commit()
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"Database error: {e}")
        raise
    finally:
        if conn:
            conn.close()


def init_database():
    """
    Initialize the database schema.
    Creates the modifiers_config and instruction_templates tables if they don't exist.
    """
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS modifiers_config (
        mednet_code VARCHAR(50) PRIMARY KEY,
        medicare_modifiers BOOLEAN NOT NULL DEFAULT FALSE,
        bill_medical_direction BOOLEAN NOT NULL DEFAULT FALSE,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    
    CREATE INDEX IF NOT EXISTS idx_mednet_code ON modifiers_config(mednet_code);
    
    CREATE TABLE IF NOT EXISTS instruction_templates (
        id SERIAL PRIMARY KEY,
        name VARCHAR(255) NOT NULL UNIQUE,
        description TEXT,
        template_data JSONB NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    
    CREATE INDEX IF NOT EXISTS idx_template_name ON instruction_templates(name);
    """
    
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(create_table_sql)
                logger.info("✅ Database schema initialized successfully")
                return True
    except Exception as e:
        logger.error(f"❌ Failed to initialize database: {e}")
        return False


def get_all_modifiers(page=1, page_size=50, search=None):
    """
    Get modifier configurations from the database with pagination.
    
    Args:
        page: Page number (1-indexed)
        page_size: Number of records per page
        search: Optional search term for mednet_code
    
    Returns:
        Dictionary with 'modifiers', 'total', 'page', 'page_size', 'total_pages'
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Build WHERE clause for search
                where_clause = ""
                params = []
                if search:
                    where_clause = "WHERE mednet_code ILIKE %s"
                    params.append(f"%{search}%")
                
                # Get total count
                count_query = f"SELECT COUNT(*) FROM modifiers_config {where_clause}"
                cur.execute(count_query, params)
                total = cur.fetchone()['count']
                
                # Calculate pagination
                offset = (page - 1) * page_size
                total_pages = (total + page_size - 1) // page_size  # Ceiling division
                
                # Get paginated results
                data_query = f"""
                    SELECT mednet_code, medicare_modifiers, bill_medical_direction, updated_at
                    FROM modifiers_config
                    {where_clause}
                    ORDER BY mednet_code
                    LIMIT %s OFFSET %s
                """
                cur.execute(data_query, params + [page_size, offset])
                results = cur.fetchall()
                
                return {
                    'modifiers': [dict(row) for row in results],
                    'total': total,
                    'page': page,
                    'page_size': page_size,
                    'total_pages': total_pages
                }
    except Exception as e:
        logger.error(f"Failed to get modifiers: {e}")
        return {
            'modifiers': [],
            'total': 0,
            'page': page,
            'page_size': page_size,
            'total_pages': 0
        }


def get_modifier(mednet_code):
    """
    Get a single modifier configuration by mednet code.
    Returns a dictionary or None if not found.
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT mednet_code, medicare_modifiers, bill_medical_direction, updated_at
                    FROM modifiers_config
                    WHERE mednet_code = %s
                """, (mednet_code,))
                result = cur.fetchone()
                return dict(result) if result else None
    except Exception as e:
        logger.error(f"Failed to get modifier {mednet_code}: {e}")
        return None


def upsert_modifier(mednet_code, medicare_modifiers, bill_medical_direction):
    """
    Insert or update a modifier configuration.
    Returns True on success, False on failure.
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
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
    except Exception as e:
        logger.error(f"Failed to upsert modifier {mednet_code}: {e}")
        return False


def delete_modifier(mednet_code):
    """
    Delete a modifier configuration.
    Returns True on success, False on failure.
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM modifiers_config WHERE mednet_code = %s", (mednet_code,))
                return True
    except Exception as e:
        logger.error(f"Failed to delete modifier {mednet_code}: {e}")
        return False


def get_modifiers_dict():
    """
    Get modifiers as a dictionary for use in generate_modifiers.py
    Returns: dict mapping mednet_code -> (medicare_modifiers, bill_medical_direction)
    """
    try:
        modifiers = get_all_modifiers()
        result = {}
        for mod in modifiers:
            result[mod['mednet_code']] = (
                mod['medicare_modifiers'],
                mod['bill_medical_direction']
            )
        return result
    except Exception as e:
        logger.error(f"Failed to get modifiers dict: {e}")
        return {}


# ============================================================
# Instruction Templates Management
# ============================================================

def get_all_templates(page=1, page_size=50, search=None):
    """
    Get instruction templates from the database with pagination.
    
    Args:
        page: Page number (1-indexed)
        page_size: Number of records per page
        search: Optional search term for template name
    
    Returns:
        Dictionary with 'templates', 'total', 'page', 'page_size', 'total_pages'
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Build WHERE clause for search
                where_clause = ""
                params = []
                if search:
                    where_clause = "WHERE name ILIKE %s OR description ILIKE %s"
                    params.extend([f"%{search}%", f"%{search}%"])
                
                # Get total count
                count_query = f"SELECT COUNT(*) FROM instruction_templates {where_clause}"
                cur.execute(count_query, params)
                total = cur.fetchone()['count']
                
                # Calculate pagination
                offset = (page - 1) * page_size
                total_pages = (total + page_size - 1) // page_size  # Ceiling division
                
                # Get paginated results
                data_query = f"""
                    SELECT id, name, description, created_at, updated_at
                    FROM instruction_templates
                    {where_clause}
                    ORDER BY name
                    LIMIT %s OFFSET %s
                """
                cur.execute(data_query, params + [page_size, offset])
                results = cur.fetchall()
                
                return {
                    'templates': [dict(row) for row in results],
                    'total': total,
                    'page': page,
                    'page_size': page_size,
                    'total_pages': total_pages
                }
    except Exception as e:
        logger.error(f"Failed to get templates: {e}")
        return {
            'templates': [],
            'total': 0,
            'page': page,
            'page_size': page_size,
            'total_pages': 0
        }


def get_template(template_id=None, template_name=None):
    """
    Get a single instruction template by ID or name.
    Returns a dictionary or None if not found.
    
    Args:
        template_id: Template ID (int)
        template_name: Template name (str)
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                if template_id:
                    cur.execute("""
                        SELECT id, name, description, template_data, created_at, updated_at
                        FROM instruction_templates
                        WHERE id = %s
                    """, (template_id,))
                elif template_name:
                    cur.execute("""
                        SELECT id, name, description, template_data, created_at, updated_at
                        FROM instruction_templates
                        WHERE name = %s
                    """, (template_name,))
                else:
                    return None
                
                result = cur.fetchone()
                return dict(result) if result else None
    except Exception as e:
        logger.error(f"Failed to get template: {e}")
        return None


def create_template(name, description, template_data):
    """
    Create a new instruction template.
    Returns the created template ID on success, None on failure.
    
    Args:
        name: Template name (unique)
        description: Template description
        template_data: Dictionary/JSON containing the template field definitions
    """
    import json
    
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    INSERT INTO instruction_templates (name, description, template_data, created_at, updated_at)
                    VALUES (%s, %s, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                    RETURNING id
                """, (name, description, json.dumps(template_data)))
                result = cur.fetchone()
                return result['id'] if result else None
    except Exception as e:
        logger.error(f"Failed to create template '{name}': {e}")
        return None


def update_template(template_id, name=None, description=None, template_data=None):
    """
    Update an existing instruction template.
    Returns True on success, False on failure.
    
    Args:
        template_id: Template ID to update
        name: New name (optional)
        description: New description (optional)
        template_data: New template data (optional)
    """
    import json
    
    # Build dynamic UPDATE statement
    updates = []
    params = []
    
    if name is not None:
        updates.append("name = %s")
        params.append(name)
    
    if description is not None:
        updates.append("description = %s")
        params.append(description)
    
    if template_data is not None:
        updates.append("template_data = %s")
        params.append(json.dumps(template_data))
    
    if not updates:
        return True  # Nothing to update
    
    updates.append("updated_at = CURRENT_TIMESTAMP")
    params.append(template_id)
    
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                query = f"""
                    UPDATE instruction_templates
                    SET {', '.join(updates)}
                    WHERE id = %s
                """
                cur.execute(query, params)
                return True
    except Exception as e:
        logger.error(f"Failed to update template {template_id}: {e}")
        return False


def delete_template(template_id):
    """
    Delete an instruction template.
    Returns True on success, False on failure.
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM instruction_templates WHERE id = %s", (template_id,))
                return True
    except Exception as e:
        logger.error(f"Failed to delete template {template_id}: {e}")
        return False


if __name__ == "__main__":
    # Test database connection
    print("Testing database connection...")
    if init_database():
        print("✅ Database connection successful!")
        modifiers = get_all_modifiers()
        print(f"Found {len(modifiers)} modifiers in database")
    else:
        print("❌ Database connection failed!")

