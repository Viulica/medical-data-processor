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
    
    CREATE TABLE IF NOT EXISTS prediction_instructions (
        id SERIAL PRIMARY KEY,
        name VARCHAR(255) NOT NULL,
        description TEXT,
        instruction_type VARCHAR(50) NOT NULL,
        instructions_text TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(name, instruction_type)
    );
    
    CREATE INDEX IF NOT EXISTS idx_prediction_instructions_name ON prediction_instructions(name);
    CREATE INDEX IF NOT EXISTS idx_prediction_instructions_type ON prediction_instructions(instruction_type);
    
    CREATE TABLE IF NOT EXISTS insurance_mappings (
        id SERIAL PRIMARY KEY,
        input_code VARCHAR(255) NOT NULL UNIQUE,
        output_code VARCHAR(255) NOT NULL,
        description TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    
    CREATE INDEX IF NOT EXISTS idx_insurance_input_code ON insurance_mappings(input_code);
    CREATE INDEX IF NOT EXISTS idx_insurance_output_code ON insurance_mappings(output_code);
    
    CREATE TABLE IF NOT EXISTS special_cases_templates (
        id SERIAL PRIMARY KEY,
        name VARCHAR(255) NOT NULL UNIQUE,
        description TEXT,
        mappings JSONB NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    
    CREATE INDEX IF NOT EXISTS idx_special_cases_templates_name ON special_cases_templates(name);
    
    CREATE TABLE IF NOT EXISTS instruction_additions_templates (
        id SERIAL PRIMARY KEY,
        name VARCHAR(255) NOT NULL UNIQUE,
        description TEXT,
        field_instructions JSONB NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    
    CREATE INDEX IF NOT EXISTS idx_instruction_additions_templates_name ON instruction_additions_templates(name);
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
                # Build WHERE clause for search (exact match)
                where_clause = ""
                params = []
                if search:
                    where_clause = "WHERE mednet_code = %s"
                    params.append(search)
                
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


def get_all_templates_for_export():
    """
    Get ALL templates with full data for export (no pagination).
    Returns a list of all templates with their complete data.
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT id, name, description, template_data, created_at, updated_at
                    FROM instruction_templates
                    ORDER BY name
                """)
                results = cur.fetchall()
                return [dict(row) for row in results]
    except Exception as e:
        logger.error(f"Failed to get all templates for export: {e}")
        return []


def bulk_update_templates(templates_data):
    """
    Bulk update or create templates from imported JSON data.
    
    Args:
        templates_data: List of template dictionaries with keys: name, description, template_data
                       Optionally can include 'id' to update existing templates by ID
    
    Returns:
        Dictionary with results: {
            'created': number of new templates created,
            'updated': number of templates updated,
            'errors': list of error messages
        }
    """
    import json
    
    created = 0
    updated = 0
    errors = []
    
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                for template in templates_data:
                    try:
                        name = template.get('name')
                        description = template.get('description', '')
                        template_data = template.get('template_data')
                        template_id = template.get('id')
                        
                        if not name:
                            errors.append(f"Template missing 'name' field: {template}")
                            continue
                        
                        if not template_data:
                            errors.append(f"Template '{name}' missing 'template_data' field")
                            continue
                        
                        # Check if template exists (by ID or name)
                        if template_id:
                            cur.execute("SELECT id FROM instruction_templates WHERE id = %s", (template_id,))
                        else:
                            cur.execute("SELECT id FROM instruction_templates WHERE name = %s", (name,))
                        
                        existing = cur.fetchone()
                        
                        if existing:
                            # Update existing template
                            update_id = existing['id']
                            cur.execute("""
                                UPDATE instruction_templates
                                SET name = %s, description = %s, template_data = %s, updated_at = CURRENT_TIMESTAMP
                                WHERE id = %s
                            """, (name, description, json.dumps(template_data), update_id))
                            updated += 1
                        else:
                            # Create new template
                            cur.execute("""
                                INSERT INTO instruction_templates (name, description, template_data, created_at, updated_at)
                                VALUES (%s, %s, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                            """, (name, description, json.dumps(template_data)))
                            created += 1
                    
                    except Exception as e:
                        error_msg = f"Error processing template '{template.get('name', 'unknown')}': {str(e)}"
                        logger.error(error_msg)
                        errors.append(error_msg)
                
                return {
                    'created': created,
                    'updated': updated,
                    'errors': errors
                }
    
    except Exception as e:
        logger.error(f"Failed to bulk update templates: {e}")
        return {
            'created': 0,
            'updated': 0,
            'errors': [f"Database error: {str(e)}"]
        }


# ============================================================
# Prediction Instructions Management (CPT/ICD)
# ============================================================

def get_all_prediction_instructions(instruction_type=None, page=1, page_size=50, search=None):
    """
    Get prediction instructions from the database with pagination.
    
    Args:
        instruction_type: Filter by type ('cpt' or 'icd'), None for all
        page: Page number (1-indexed)
        page_size: Number of records per page
        search: Optional search term for name
    
    Returns:
        Dictionary with 'instructions', 'total', 'page', 'page_size', 'total_pages'
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Build WHERE clause
                where_conditions = []
                params = []
                
                if instruction_type:
                    where_conditions.append("instruction_type = %s")
                    params.append(instruction_type)
                
                if search:
                    where_conditions.append("(name ILIKE %s OR description ILIKE %s)")
                    params.extend([f"%{search}%", f"%{search}%"])
                
                where_clause = ""
                if where_conditions:
                    where_clause = "WHERE " + " AND ".join(where_conditions)
                
                # Get total count
                count_query = f"SELECT COUNT(*) FROM prediction_instructions {where_clause}"
                cur.execute(count_query, params)
                total = cur.fetchone()['count']
                
                # Calculate pagination
                offset = (page - 1) * page_size
                total_pages = (total + page_size - 1) // page_size
                
                # Get paginated results
                data_query = f"""
                    SELECT id, name, description, instruction_type, created_at, updated_at
                    FROM prediction_instructions
                    {where_clause}
                    ORDER BY instruction_type, name
                    LIMIT %s OFFSET %s
                """
                cur.execute(data_query, params + [page_size, offset])
                results = cur.fetchall()
                
                return {
                    'instructions': [dict(row) for row in results],
                    'total': total,
                    'page': page,
                    'page_size': page_size,
                    'total_pages': total_pages
                }
    except Exception as e:
        logger.error(f"Failed to get prediction instructions: {e}")
        return {
            'instructions': [],
            'total': 0,
            'page': page,
            'page_size': page_size,
            'total_pages': 0
        }


def get_prediction_instruction(instruction_id=None, name=None, instruction_type=None):
    """
    Get a single prediction instruction by ID or by name+type.
    Returns a dictionary or None if not found.
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                if instruction_id:
                    cur.execute("""
                        SELECT id, name, description, instruction_type, instructions_text, created_at, updated_at
                        FROM prediction_instructions
                        WHERE id = %s
                    """, (instruction_id,))
                elif name and instruction_type:
                    cur.execute("""
                        SELECT id, name, description, instruction_type, instructions_text, created_at, updated_at
                        FROM prediction_instructions
                        WHERE name = %s AND instruction_type = %s
                    """, (name, instruction_type))
                else:
                    return None
                
                result = cur.fetchone()
                return dict(result) if result else None
    except Exception as e:
        logger.error(f"Failed to get prediction instruction: {e}")
        return None


def create_prediction_instruction(name, instruction_type, instructions_text, description=""):
    """
    Create a new prediction instruction template.
    Returns the created instruction ID on success, None on failure.
    
    Args:
        name: Template name
        instruction_type: Type - 'cpt' or 'icd'
        instructions_text: The instruction text
        description: Optional description
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    INSERT INTO prediction_instructions (name, description, instruction_type, instructions_text, created_at, updated_at)
                    VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                    RETURNING id
                """, (name, description, instruction_type, instructions_text))
                result = cur.fetchone()
                return result['id'] if result else None
    except Exception as e:
        logger.error(f"Failed to create prediction instruction '{name}': {e}")
        return None


def update_prediction_instruction(instruction_id, name=None, description=None, instructions_text=None):
    """
    Update an existing prediction instruction.
    Returns True on success, False on failure.
    """
    # Build dynamic UPDATE statement
    updates = []
    params = []
    
    if name is not None:
        updates.append("name = %s")
        params.append(name)
    
    if description is not None:
        updates.append("description = %s")
        params.append(description)
    
    if instructions_text is not None:
        updates.append("instructions_text = %s")
        params.append(instructions_text)
    
    if not updates:
        return True  # Nothing to update
    
    updates.append("updated_at = CURRENT_TIMESTAMP")
    params.append(instruction_id)
    
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                query = f"""
                    UPDATE prediction_instructions
                    SET {', '.join(updates)}
                    WHERE id = %s
                """
                cur.execute(query, params)
                return True
    except Exception as e:
        logger.error(f"Failed to update prediction instruction {instruction_id}: {e}")
        return False


def delete_prediction_instruction(instruction_id):
    """
    Delete a prediction instruction.
    Returns True on success, False on failure.
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM prediction_instructions WHERE id = %s", (instruction_id,))
                return True
    except Exception as e:
        logger.error(f"Failed to delete prediction instruction {instruction_id}: {e}")
        return False


# ============================================================
# Insurance Mappings Management
# ============================================================

def get_all_insurance_mappings(page=1, page_size=50, search=None):
    """
    Get insurance mappings from the database with pagination.
    
    Args:
        page: Page number (1-indexed)
        page_size: Number of records per page
        search: Optional search term for input_code or output_code (exact match)
    
    Returns:
        Dictionary with 'mappings', 'total', 'page', 'page_size', 'total_pages'
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Build WHERE clause for search (exact match on either field)
                where_clause = ""
                params = []
                if search:
                    where_clause = "WHERE input_code = %s OR output_code = %s"
                    params.extend([search, search])
                
                # Get total count
                count_query = f"SELECT COUNT(*) FROM insurance_mappings {where_clause}"
                cur.execute(count_query, params)
                total = cur.fetchone()['count']
                
                # Calculate pagination
                offset = (page - 1) * page_size
                total_pages = (total + page_size - 1) // page_size
                
                # Get paginated results
                data_query = f"""
                    SELECT id, input_code, output_code, description, updated_at
                    FROM insurance_mappings
                    {where_clause}
                    ORDER BY input_code
                    LIMIT %s OFFSET %s
                """
                cur.execute(data_query, params + [page_size, offset])
                results = cur.fetchall()
                
                return {
                    'mappings': [dict(row) for row in results],
                    'total': total,
                    'page': page,
                    'page_size': page_size,
                    'total_pages': total_pages
                }
    except Exception as e:
        logger.error(f"Failed to get insurance mappings: {e}")
        return {
            'mappings': [],
            'total': 0,
            'page': page,
            'page_size': page_size,
            'total_pages': 0
        }


def get_insurance_mapping(mapping_id=None, input_code=None):
    """
    Get a single insurance mapping by ID or input_code.
    Returns a dictionary or None if not found.
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                if mapping_id:
                    cur.execute("""
                        SELECT id, input_code, output_code, description, updated_at
                        FROM insurance_mappings
                        WHERE id = %s
                    """, (mapping_id,))
                elif input_code:
                    cur.execute("""
                        SELECT id, input_code, output_code, description, updated_at
                        FROM insurance_mappings
                        WHERE input_code = %s
                    """, (input_code,))
                else:
                    return None
                
                result = cur.fetchone()
                return dict(result) if result else None
    except Exception as e:
        logger.error(f"Failed to get insurance mapping: {e}")
        return None


def upsert_insurance_mapping(input_code, output_code, description=""):
    """
    Insert or update an insurance mapping.
    Returns True on success, False on failure.
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO insurance_mappings (input_code, output_code, description, updated_at)
                    VALUES (%s, %s, %s, CURRENT_TIMESTAMP)
                    ON CONFLICT (input_code) 
                    DO UPDATE SET 
                        output_code = EXCLUDED.output_code,
                        description = EXCLUDED.description,
                        updated_at = CURRENT_TIMESTAMP
                """, (input_code, output_code, description))
                return True
    except Exception as e:
        logger.error(f"Failed to upsert insurance mapping {input_code}: {e}")
        return False


def delete_insurance_mapping(mapping_id):
    """
    Delete an insurance mapping.
    Returns True on success, False on failure.
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM insurance_mappings WHERE id = %s", (mapping_id,))
                return True
    except Exception as e:
        logger.error(f"Failed to delete insurance mapping {mapping_id}: {e}")
        return False


def get_insurance_mappings_dict():
    """
    Get insurance mappings as a dictionary for use in conversion scripts.
    Returns: dict mapping input_code -> output_code
    """
    try:
        result = get_all_insurance_mappings(page=1, page_size=10000)  # Get all
        mappings_dict = {}
        for mapping in result['mappings']:
            mappings_dict[mapping['input_code']] = mapping['output_code']
        return mappings_dict
    except Exception as e:
        logger.error(f"Failed to get insurance mappings dict: {e}")
        return {}


def bulk_import_insurance_mappings(mappings_data, clear_existing=False):
    """
    Bulk import insurance mappings from a list of dictionaries.
    Each dictionary should have 'input_code' and 'output_code' keys.
    
    Args:
        mappings_data: List of dicts with 'input_code', 'output_code', 'description' (optional)
        clear_existing: If True, delete all existing mappings before importing
    
    Returns:
        Dictionary with 'success', 'imported', 'updated', 'skipped', 'errors'
    """
    imported = 0
    updated = 0
    skipped = 0
    errors = []
    
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Clear existing mappings if requested
                if clear_existing:
                    cur.execute("DELETE FROM insurance_mappings")
                    logger.info("Cleared all existing insurance mappings")
                
                # Import each mapping
                for idx, mapping in enumerate(mappings_data):
                    try:
                        input_code = mapping.get('input_code', '').strip()
                        output_code = mapping.get('output_code', '').strip()
                        description = mapping.get('description', '').strip()
                        
                        if not input_code or not output_code:
                            skipped += 1
                            continue
                        
                        # Use upsert logic (INSERT ... ON CONFLICT UPDATE)
                        cur.execute("""
                            INSERT INTO insurance_mappings (input_code, output_code, description, created_at, updated_at)
                            VALUES (%s, %s, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                            ON CONFLICT (input_code) 
                            DO UPDATE SET 
                                output_code = EXCLUDED.output_code,
                                description = EXCLUDED.description,
                                updated_at = CURRENT_TIMESTAMP
                            RETURNING (xmax = 0) AS inserted
                        """, (input_code, output_code, description))
                        
                        result = cur.fetchone()
                        if result and result[0]:  # xmax = 0 means it was an INSERT
                            imported += 1
                        else:
                            updated += 1
                            
                    except Exception as e:
                        errors.append(f"Row {idx + 1}: {str(e)}")
                        logger.error(f"Error importing mapping row {idx + 1}: {e}")
                
                conn.commit()
                
        return {
            'success': True,
            'imported': imported,
            'updated': updated,
            'skipped': skipped,
            'errors': errors,
            'total': len(mappings_data)
        }
        
    except Exception as e:
        logger.error(f"Failed to bulk import insurance mappings: {e}")
        return {
            'success': False,
            'imported': imported,
            'updated': updated,
            'skipped': skipped,
            'errors': [str(e)],
            'total': len(mappings_data)
        }


# ============================================================================
# Special Cases Templates CRUD Operations
# ============================================================================

def get_all_special_cases_templates(page=1, page_size=50, search=None):
    """
    Get all special cases templates with optional search and pagination.
    Returns dict with templates list and pagination info.
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Build WHERE clause
                where_clause = ""
                params = []
                
                if search:
                    where_clause = "WHERE name ILIKE %s OR description ILIKE %s"
                    search_param = f"%{search}%"
                    params.extend([search_param, search_param])
                
                # Get total count
                count_query = f"SELECT COUNT(*) FROM special_cases_templates {where_clause}"
                cur.execute(count_query, params)
                total = cur.fetchone()['count']
                
                # Get paginated results
                offset = (page - 1) * page_size
                query = f"""
                    SELECT id, name, description, mappings, created_at, updated_at
                    FROM special_cases_templates
                    {where_clause}
                    ORDER BY updated_at DESC
                    LIMIT %s OFFSET %s
                """
                cur.execute(query, params + [page_size, offset])
                templates = [dict(row) for row in cur.fetchall()]
                
                return {
                    'templates': templates,
                    'total': total,
                    'page': page,
                    'page_size': page_size,
                    'total_pages': (total + page_size - 1) // page_size
                }
    except Exception as e:
        logger.error(f"Failed to get special cases templates: {e}")
        return {
            'templates': [],
            'total': 0,
            'page': page,
            'page_size': page_size,
            'total_pages': 0
        }


def get_special_cases_template(template_id=None, name=None):
    """
    Get a single special cases template by ID or name.
    Returns a dictionary or None if not found.
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                if template_id:
                    cur.execute("""
                        SELECT id, name, description, mappings, created_at, updated_at
                        FROM special_cases_templates
                        WHERE id = %s
                    """, (template_id,))
                elif name:
                    cur.execute("""
                        SELECT id, name, description, mappings, created_at, updated_at
                        FROM special_cases_templates
                        WHERE name = %s
                    """, (name,))
                else:
                    return None
                
                result = cur.fetchone()
                return dict(result) if result else None
    except Exception as e:
        logger.error(f"Failed to get special cases template: {e}")
        return None


def create_special_cases_template(name, mappings, description=""):
    """
    Create a new special cases template.
    
    Args:
        name: Template name
        mappings: List of dicts with 'company_name' and 'mednet_code'
        description: Optional description
    
    Returns:
        Created template ID on success, None on failure
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    INSERT INTO special_cases_templates (name, description, mappings, created_at, updated_at)
                    VALUES (%s, %s, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                    RETURNING id
                """, (name, description, json.dumps(mappings)))
                result = cur.fetchone()
                return result['id'] if result else None
    except Exception as e:
        logger.error(f"Failed to create special cases template '{name}': {e}")
        return None


def update_special_cases_template(template_id, name=None, description=None, mappings=None):
    """
    Update an existing special cases template.
    Returns True on success, False on failure.
    """
    updates = []
    params = []
    
    if name is not None:
        updates.append("name = %s")
        params.append(name)
    
    if description is not None:
        updates.append("description = %s")
        params.append(description)
    
    if mappings is not None:
        updates.append("mappings = %s")
        params.append(json.dumps(mappings))
    
    if not updates:
        return True
    
    updates.append("updated_at = CURRENT_TIMESTAMP")
    params.append(template_id)
    
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                query = f"""
                    UPDATE special_cases_templates
                    SET {', '.join(updates)}
                    WHERE id = %s
                """
                cur.execute(query, params)
                return True
    except Exception as e:
        logger.error(f"Failed to update special cases template {template_id}: {e}")
        return False


def delete_special_cases_template(template_id):
    """Delete a special cases template. Returns True on success, False on failure."""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM special_cases_templates WHERE id = %s", (template_id,))
                return True
    except Exception as e:
        logger.error(f"Failed to delete special cases template {template_id}: {e}")
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

