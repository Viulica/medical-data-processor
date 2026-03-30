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
from typing import Optional

logger = logging.getLogger(__name__)

# Database connection string from environment
DATABASE_URL = os.environ.get(
    'DATABASE_URL',
    'postgresql://postgres:YISwNRCXxndHsFmucFxwFhEXJrHxQaEC@centerbeam.proxy.rlwy.net:33249/railway'
)

# Supabase Storage configuration
SUPABASE_URL = os.environ.get('SUPABASE_URL', 'https://tfndwkxikqttrchxqakl.supabase.co')
SUPABASE_KEY = os.environ.get('SUPABASE_KEY', 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InRmbmR3a3hpa3F0dHJjaHhxYWtsIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc3MzMzNDY3NiwiZXhwIjoyMDg4OTEwNjc2fQ.bhGr3rFy9sLXVqNkpfNeut1d8LGLph4u01QtrDWy_iw')
SUPABASE_BUCKET = 'results'


def upload_to_supabase(path: str, data: bytes, content_type: str) -> str:
    """Upload bytes to Supabase Storage. Returns the storage path."""
    import httpx
    url = f"{SUPABASE_URL}/storage/v1/object/{SUPABASE_BUCKET}/{path}"
    headers = {
        'Authorization': f'Bearer {SUPABASE_KEY}',
        'Content-Type': content_type,
        'Content-Length': str(len(data)),
        'x-upsert': 'true',
    }
    r = httpx.put(url, content=data, headers=headers, timeout=300)
    if not r.is_success:
        logger.error(f"Supabase upload failed {r.status_code}: {r.text}")
    r.raise_for_status()
    return path


def get_supabase_signed_url(path: str, expires_in: int = 3600) -> str:
    """Get a signed download URL for a Supabase Storage object."""
    import httpx
    url = f"{SUPABASE_URL}/storage/v1/object/sign/{SUPABASE_BUCKET}/{path}"
    headers = {
        'Authorization': f'Bearer {SUPABASE_KEY}',
        'Content-Type': 'application/json',
    }
    r = httpx.post(url, json={'expiresIn': expires_in}, headers=headers, timeout=10)
    r.raise_for_status()
    signed_path = r.json()['signedURL']
    # Supabase returns path like /object/sign/... but needs /storage/v1 prefix
    if signed_path.startswith('/object/'):
        signed_path = f"/storage/v1{signed_path}"
    return f"{SUPABASE_URL}{signed_path}"


def delete_from_supabase(paths: list) -> None:
    """Delete objects from Supabase Storage."""
    import httpx
    url = f"{SUPABASE_URL}/storage/v1/object/{SUPABASE_BUCKET}"
    headers = {
        'Authorization': f'Bearer {SUPABASE_KEY}',
        'Content-Type': 'application/json',
    }
    r = httpx.delete(url, json={'prefixes': paths}, headers=headers, timeout=10)
    r.raise_for_status()


def ensure_supabase_bucket() -> None:
    """Create the Supabase Storage bucket if it doesn't exist."""
    import httpx
    url = f"{SUPABASE_URL}/storage/v1/bucket"
    headers = {
        'Authorization': f'Bearer {SUPABASE_KEY}',
        'Content-Type': 'application/json',
    }
    try:
        r = httpx.post(
            url,
            json={'id': SUPABASE_BUCKET, 'name': SUPABASE_BUCKET, 'public': False},
            headers=headers,
            timeout=10
        )
        if r.status_code not in (200, 201, 409):  # 409 = already exists
            r.raise_for_status()
        logger.info(f"✅ Supabase bucket '{SUPABASE_BUCKET}' ready")
    except Exception as e:
        logger.warning(f"Failed to ensure Supabase bucket: {e}")


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


def migrate_provider_mapping_columns():
    """
    Add provider_mapping and extract_providers_from_annotations columns to instruction_templates if they don't exist.
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Check if columns exist
                cur.execute("""
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_name='instruction_templates' 
                    AND column_name IN ('provider_mapping', 'extract_providers_from_annotations')
                """)
                existing_columns = {row[0] for row in cur.fetchall()}
                
                # Add provider_mapping column if it doesn't exist
                if 'provider_mapping' not in existing_columns:
                    cur.execute("""
                        ALTER TABLE instruction_templates 
                        ADD COLUMN provider_mapping TEXT
                    """)
                    logger.info("✅ Added provider_mapping column to instruction_templates")
                
                # Add extract_providers_from_annotations column if it doesn't exist
                if 'extract_providers_from_annotations' not in existing_columns:
                    cur.execute("""
                        ALTER TABLE instruction_templates 
                        ADD COLUMN extract_providers_from_annotations BOOLEAN NOT NULL DEFAULT FALSE
                    """)
                    logger.info("✅ Added extract_providers_from_annotations column to instruction_templates")
                
                return True
    except Exception as e:
        logger.error(f"Failed to migrate provider mapping columns: {e}")
        return False


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
        enable_qs BOOLEAN NOT NULL DEFAULT TRUE,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    
    CREATE INDEX IF NOT EXISTS idx_mednet_code ON modifiers_config(mednet_code);
    
    CREATE TABLE IF NOT EXISTS instruction_templates (
        id SERIAL PRIMARY KEY,
        name VARCHAR(255) NOT NULL UNIQUE,
        description TEXT,
        template_data JSONB NOT NULL,
        provider_mapping TEXT,
        extract_providers_from_annotations BOOLEAN NOT NULL DEFAULT FALSE,
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
        input_code VARCHAR(255) NOT NULL,
        output_code VARCHAR(255) NOT NULL,
        description TEXT,
        client VARCHAR(50) NOT NULL DEFAULT 'uni',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(input_code, client)
    );

    CREATE INDEX IF NOT EXISTS idx_insurance_input_code ON insurance_mappings(input_code);
    CREATE INDEX IF NOT EXISTS idx_insurance_output_code ON insurance_mappings(output_code);
    CREATE INDEX IF NOT EXISTS idx_insurance_client ON insurance_mappings(client);
    
    CREATE TABLE IF NOT EXISTS base_prompts (
        id SERIAL PRIMARY KEY,
        name VARCHAR(100) NOT NULL UNIQUE,
        prompt_type VARCHAR(50) NOT NULL,
        content TEXT NOT NULL,
        description VARCHAR(500),
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS unified_results (
        id SERIAL PRIMARY KEY,
        job_id VARCHAR(255) NOT NULL UNIQUE,
        filename VARCHAR(255),
        file_path_csv VARCHAR(500) NOT NULL,
        file_path_xlsx VARCHAR(500),
        file_path_renamed_zip VARCHAR(500),
        file_size_bytes BIGINT,
        row_count INTEGER,
        enabled_extraction BOOLEAN DEFAULT FALSE,
        enabled_cpt BOOLEAN DEFAULT FALSE,
        enabled_icd BOOLEAN DEFAULT FALSE,
        extraction_model VARCHAR(100),
        cpt_vision_model VARCHAR(100),
        icd_vision_model VARCHAR(100),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        expires_at TIMESTAMP NOT NULL,
        status VARCHAR(50) DEFAULT 'completed'
    );

    -- Migration: Add file_path_renamed_zip column if it doesn't exist
    DO $$
    BEGIN
        IF NOT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_name = 'unified_results' AND column_name = 'file_path_renamed_zip'
        ) THEN
            ALTER TABLE unified_results ADD COLUMN file_path_renamed_zip VARCHAR(500);
        END IF;
    END $$;

    -- Migration: Add supabase_path column if it doesn't exist
    DO $$
    BEGIN
        IF NOT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_name = 'unified_results' AND column_name = 'supabase_path'
        ) THEN
            ALTER TABLE unified_results ADD COLUMN supabase_path VARCHAR(500);
        END IF;
    END $$;

    -- Migration: Make file_path_csv nullable
    DO $$
    BEGIN
        ALTER TABLE unified_results ALTER COLUMN file_path_csv DROP NOT NULL;
    EXCEPTION WHEN OTHERS THEN
        NULL;
    END $$;

    -- Migration: Add input_zip_supabase_path column if it doesn't exist
    DO $$
    BEGIN
        IF NOT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_name = 'unified_results' AND column_name = 'input_zip_supabase_path'
        ) THEN
            ALTER TABLE unified_results ADD COLUMN input_zip_supabase_path VARCHAR(500);
        END IF;
    END $$;

    -- Migration: Add worktracker columns if they don't exist
    DO $$
    BEGIN
        IF NOT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_name = 'unified_results' AND column_name = 'worktracker_group'
        ) THEN
            ALTER TABLE unified_results ADD COLUMN worktracker_group VARCHAR(255);
            ALTER TABLE unified_results ADD COLUMN worktracker_batch VARCHAR(255);
        END IF;
    END $$;

    CREATE INDEX IF NOT EXISTS idx_unified_results_job_id ON unified_results(job_id);
    CREATE INDEX IF NOT EXISTS idx_unified_results_created_at ON unified_results(created_at);
    CREATE INDEX IF NOT EXISTS idx_unified_results_expires_at ON unified_results(expires_at);
    
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
    
    CREATE TABLE IF NOT EXISTS refinement_jobs (
        id SERIAL PRIMARY KEY,
        job_id VARCHAR(255) UNIQUE NOT NULL,
        user_email VARCHAR(255) NOT NULL,
        original_cpt_template_id INT,
        original_icd_template_id INT,
        current_cpt_template_id INT,
        current_icd_template_id INT,
        best_cpt_template_id INT,
        best_icd_template_id INT,
        phase VARCHAR(20) NOT NULL,
        iteration INT NOT NULL DEFAULT 0,
        cpt_accuracy FLOAT,
        icd1_accuracy FLOAT,
        best_cpt_accuracy FLOAT,
        best_icd1_accuracy FLOAT,
        status VARCHAR(50) NOT NULL,
        error_message TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    
    CREATE INDEX IF NOT EXISTS idx_refinement_jobs_job_id ON refinement_jobs(job_id);
    CREATE INDEX IF NOT EXISTS idx_refinement_jobs_status ON refinement_jobs(status);
    """
    
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(create_table_sql)
                logger.info("✅ Database schema initialized successfully")
        
        # Run migrations
        migrate_provider_mapping_columns()
        migrate_insurance_mappings_client_column()

        # Ensure Supabase bucket exists
        ensure_supabase_bucket()

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
                # Check if enable_qs column exists
                cur.execute("""
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_name = 'modifiers_config' AND column_name = 'enable_qs'
                """)
                has_enable_qs = cur.fetchone() is not None
                
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
                
                # Get paginated results - conditionally include enable_qs if column exists
                if has_enable_qs:
                    data_query = f"""
                        SELECT mednet_code, medicare_modifiers, bill_medical_direction, enable_qs, updated_at
                        FROM modifiers_config
                        {where_clause}
                        ORDER BY mednet_code
                        LIMIT %s OFFSET %s
                    """
                else:
                    # Column doesn't exist yet - select without it and default to True
                    data_query = f"""
                        SELECT mednet_code, medicare_modifiers, bill_medical_direction, TRUE as enable_qs, updated_at
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
        import traceback
        logger.error(traceback.format_exc())
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
                # Check if enable_qs column exists
                cur.execute("""
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_name = 'modifiers_config' AND column_name = 'enable_qs'
                """)
                has_enable_qs = cur.fetchone() is not None
                
                # Conditionally include enable_qs if column exists
                if has_enable_qs:
                    cur.execute("""
                        SELECT mednet_code, medicare_modifiers, bill_medical_direction, enable_qs, updated_at
                        FROM modifiers_config
                        WHERE mednet_code = %s
                    """, (mednet_code,))
                else:
                    cur.execute("""
                        SELECT mednet_code, medicare_modifiers, bill_medical_direction, TRUE as enable_qs, updated_at
                        FROM modifiers_config
                        WHERE mednet_code = %s
                    """, (mednet_code,))
                
                result = cur.fetchone()
                return dict(result) if result else None
    except Exception as e:
        logger.error(f"Failed to get modifier {mednet_code}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def upsert_modifier(mednet_code, medicare_modifiers, bill_medical_direction, enable_qs=True):
    """
    Insert or update a modifier configuration.
    Returns True on success, False on failure.
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Check if enable_qs column exists, if not add it
                cur.execute("""
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_name = 'modifiers_config' AND column_name = 'enable_qs'
                """)
                has_enable_qs = cur.fetchone() is not None
                
                if not has_enable_qs:
                    # Add the column if it doesn't exist
                    logger.info("Adding enable_qs column to modifiers_config table...")
                    cur.execute("""
                        ALTER TABLE modifiers_config 
                        ADD COLUMN enable_qs BOOLEAN NOT NULL DEFAULT TRUE
                    """)
                    conn.commit()
                
                # Now perform the upsert
                cur.execute("""
                    INSERT INTO modifiers_config (mednet_code, medicare_modifiers, bill_medical_direction, enable_qs, updated_at)
                    VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP)
                    ON CONFLICT (mednet_code) 
                    DO UPDATE SET 
                        medicare_modifiers = EXCLUDED.medicare_modifiers,
                        bill_medical_direction = EXCLUDED.bill_medical_direction,
                        enable_qs = EXCLUDED.enable_qs,
                        updated_at = CURRENT_TIMESTAMP
                """, (mednet_code, medicare_modifiers, bill_medical_direction, enable_qs))
                return True
    except Exception as e:
        logger.error(f"Failed to upsert modifier {mednet_code}: {e}")
        import traceback
        logger.error(traceback.format_exc())
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
    Returns: dict mapping mednet_code -> (medicare_modifiers, bill_medical_direction, enable_qs)
    """
    try:
        # Get all modifiers with a large page size to retrieve all records
        modifiers_data = get_all_modifiers(page=1, page_size=100000)
        result = {}
        for mod in modifiers_data['modifiers']:
            # enable_qs will be included by get_all_modifiers (either from column or defaulted to True)
            result[mod['mednet_code']] = (
                mod['medicare_modifiers'],
                mod['bill_medical_direction'],
                mod.get('enable_qs', True)  # Default to True if not present (shouldn't happen now)
            )
        return result
    except Exception as e:
        logger.error(f"Failed to get modifiers dict: {e}")
        import traceback
        logger.error(traceback.format_exc())
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
                        SELECT id, name, description, template_data, provider_mapping, extract_providers_from_annotations, created_at, updated_at
                        FROM instruction_templates
                        WHERE id = %s
                    """, (template_id,))
                elif template_name:
                    cur.execute("""
                        SELECT id, name, description, template_data, provider_mapping, extract_providers_from_annotations, created_at, updated_at
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


def create_template(name, description, template_data, provider_mapping=None, extract_providers_from_annotations=False):
    """
    Create a new instruction template.
    Returns the created template ID on success, None on failure.
    
    Args:
        name: Template name (unique)
        description: Template description
        template_data: Dictionary/JSON containing the template field definitions
        provider_mapping: Optional provider mapping text
        extract_providers_from_annotations: Whether to extract providers from PDF annotations
    """
    import json
    
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    INSERT INTO instruction_templates (name, description, template_data, provider_mapping, extract_providers_from_annotations, created_at, updated_at)
                    VALUES (%s, %s, %s, %s, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                    RETURNING id
                """, (name, description, json.dumps(template_data), provider_mapping, extract_providers_from_annotations))
                result = cur.fetchone()
                return result['id'] if result else None
    except Exception as e:
        logger.error(f"Failed to create template '{name}': {e}")
        return None


def update_template(template_id, name=None, description=None, template_data=None, provider_mapping=None, extract_providers_from_annotations=None):
    """
    Update an existing instruction template.
    Returns True on success, False on failure.
    
    Args:
        template_id: Template ID to update
        name: New name (optional)
        description: New description (optional)
        template_data: New template data (optional)
        provider_mapping: New provider mapping text (optional, pass empty string to clear)
        extract_providers_from_annotations: Whether to extract providers from annotations (optional)
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
    
    if provider_mapping is not None:
        updates.append("provider_mapping = %s")
        params.append(provider_mapping if provider_mapping else None)
    
    if extract_providers_from_annotations is not None:
        updates.append("extract_providers_from_annotations = %s")
        params.append(extract_providers_from_annotations)
    
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

def migrate_insurance_mappings_client_column():
    """
    Add 'client' column to insurance_mappings table if it doesn't exist,
    and update the unique constraint from (input_code) to (input_code, client).
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Check if client column exists
                cur.execute("""
                    SELECT column_name FROM information_schema.columns
                    WHERE table_name='insurance_mappings' AND column_name='client'
                """)
                if not cur.fetchone():
                    logger.info("Adding 'client' column to insurance_mappings table...")
                    cur.execute("ALTER TABLE insurance_mappings ADD COLUMN client VARCHAR(50) NOT NULL DEFAULT 'uni'")
                    # Drop old unique constraint on input_code alone
                    cur.execute("""
                        DO $$ BEGIN
                            ALTER TABLE insurance_mappings DROP CONSTRAINT IF EXISTS insurance_mappings_input_code_key;
                        EXCEPTION WHEN undefined_object THEN NULL;
                        END $$;
                    """)
                    # Add new composite unique constraint
                    cur.execute("""
                        ALTER TABLE insurance_mappings
                        ADD CONSTRAINT insurance_mappings_input_code_client_key UNIQUE (input_code, client)
                    """)
                    cur.execute("CREATE INDEX IF NOT EXISTS idx_insurance_client ON insurance_mappings(client)")
                    conn.commit()
                    logger.info("Successfully migrated insurance_mappings with client column")
    except Exception as e:
        logger.error(f"Failed to migrate insurance_mappings client column: {e}")


def get_all_insurance_mappings(page=1, page_size=50, search=None, client=None):
    """
    Get insurance mappings from the database with pagination.

    Args:
        page: Page number (1-indexed)
        page_size: Number of records per page
        search: Optional search term for input_code or output_code (exact match)
        client: Optional client filter (e.g. 'uni', 'pac'). If None, returns all.

    Returns:
        Dictionary with 'mappings', 'total', 'page', 'page_size', 'total_pages'
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Build WHERE clause
                conditions = []
                params = []
                if search:
                    conditions.append("(input_code = %s OR output_code = %s)")
                    params.extend([search, search])
                if client:
                    conditions.append("client = %s")
                    params.append(client)

                where_clause = ("WHERE " + " AND ".join(conditions)) if conditions else ""

                # Get total count
                count_query = f"SELECT COUNT(*) FROM insurance_mappings {where_clause}"
                cur.execute(count_query, params)
                total = cur.fetchone()['count']

                # Calculate pagination
                offset = (page - 1) * page_size
                total_pages = (total + page_size - 1) // page_size

                # Get paginated results
                data_query = f"""
                    SELECT id, input_code, output_code, description, client, updated_at
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


def get_insurance_mapping(mapping_id=None, input_code=None, client=None):
    """
    Get a single insurance mapping by ID or input_code.
    Returns a dictionary or None if not found.
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                if mapping_id:
                    cur.execute("""
                        SELECT id, input_code, output_code, description, client, updated_at
                        FROM insurance_mappings
                        WHERE id = %s
                    """, (mapping_id,))
                elif input_code:
                    query = """
                        SELECT id, input_code, output_code, description, client, updated_at
                        FROM insurance_mappings
                        WHERE input_code = %s
                    """
                    params = [input_code]
                    if client:
                        query += " AND client = %s"
                        params.append(client)
                    cur.execute(query, params)
                else:
                    return None

                result = cur.fetchone()
                return dict(result) if result else None
    except Exception as e:
        logger.error(f"Failed to get insurance mapping: {e}")
        return None


def upsert_insurance_mapping(input_code, output_code, description="", client="uni"):
    """
    Insert or update an insurance mapping.
    Returns True on success, False on failure.
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO insurance_mappings (input_code, output_code, description, client, updated_at)
                    VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP)
                    ON CONFLICT (input_code, client)
                    DO UPDATE SET
                        output_code = EXCLUDED.output_code,
                        description = EXCLUDED.description,
                        updated_at = CURRENT_TIMESTAMP
                """, (input_code, output_code, description, client))
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


def get_insurance_mappings_dict(client="uni"):
    """
    Get insurance mappings as a dictionary for use in conversion scripts.
    Args:
        client: Client identifier (e.g. 'uni', 'pac')
    Returns: dict mapping input_code -> output_code
    """
    try:
        result = get_all_insurance_mappings(page=1, page_size=10000, client=client)
        mappings_dict = {}
        for mapping in result['mappings']:
            mappings_dict[mapping['input_code']] = mapping['output_code']
        return mappings_dict
    except Exception as e:
        logger.error(f"Failed to get insurance mappings dict: {e}")
        return {}


def bulk_import_insurance_mappings(mappings_data, clear_existing=False, client="uni"):
    """
    Bulk import insurance mappings from a list of dictionaries.
    Each dictionary should have 'input_code' and 'output_code' keys.

    Args:
        mappings_data: List of dicts with 'input_code', 'output_code', 'description' (optional)
        clear_existing: If True, delete all existing mappings for this client before importing
        client: Client identifier (e.g. 'uni', 'pac')

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
                # Clear existing mappings for this client if requested
                if clear_existing:
                    cur.execute("DELETE FROM insurance_mappings WHERE client = %s", (client,))
                    logger.info(f"Cleared all existing insurance mappings for client '{client}'")

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
                            INSERT INTO insurance_mappings (input_code, output_code, description, client, created_at, updated_at)
                            VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                            ON CONFLICT (input_code, client)
                            DO UPDATE SET
                                output_code = EXCLUDED.output_code,
                                description = EXCLUDED.description,
                                updated_at = CURRENT_TIMESTAMP
                            RETURNING (xmax = 0) AS inserted
                        """, (input_code, output_code, description, client))

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


# ============================================================
# Refinement Jobs Management
# ============================================================

def create_refinement_job(
    job_id: str,
    user_email: str,
    original_cpt_template_id: Optional[int] = None,
    original_icd_template_id: Optional[int] = None,
    current_cpt_template_id: Optional[int] = None,
    current_icd_template_id: Optional[int] = None,
    phase: str = "cpt",
    status: str = "running"
) -> bool:
    """
    Create a new refinement job record.
    
    Args:
        job_id: Unique job identifier
        user_email: User email for notifications
        original_cpt_template_id: Original CPT template ID
        original_icd_template_id: Original ICD template ID
        current_cpt_template_id: Current CPT template ID
        current_icd_template_id: Current ICD template ID
        phase: Current phase ('cpt' or 'icd')
        status: Job status ('running', 'completed', 'failed')
    
    Returns:
        True on success, False on failure
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO refinement_jobs (
                        job_id, user_email, original_cpt_template_id, original_icd_template_id,
                        current_cpt_template_id, current_icd_template_id,
                        best_cpt_template_id, best_icd_template_id,
                        phase, iteration, status, created_at, updated_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, 0, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                """, (
                    job_id, user_email, original_cpt_template_id, original_icd_template_id,
                    current_cpt_template_id, current_icd_template_id,
                    original_cpt_template_id, original_icd_template_id,  # Best starts as original
                    phase, status
                ))
                return True
    except Exception as e:
        logger.error(f"Failed to create refinement job {job_id}: {e}")
        return False


def update_refinement_job(
    job_id: str,
    phase: Optional[str] = None,
    iteration: Optional[int] = None,
    cpt_accuracy: Optional[float] = None,
    icd1_accuracy: Optional[float] = None,
    best_cpt_accuracy: Optional[float] = None,
    best_icd1_accuracy: Optional[float] = None,
    current_cpt_template_id: Optional[int] = None,
    current_icd_template_id: Optional[int] = None,
    best_cpt_template_id: Optional[int] = None,
    best_icd_template_id: Optional[int] = None,
    status: Optional[str] = None,
    error_message: Optional[str] = None
) -> bool:
    """
    Update a refinement job record.
    
    Args:
        job_id: Job identifier
        phase: Current phase
        iteration: Current iteration number
        cpt_accuracy: Current CPT accuracy
        icd1_accuracy: Current ICD1 accuracy
        best_cpt_accuracy: Best CPT accuracy achieved
        best_icd1_accuracy: Best ICD1 accuracy achieved
        current_cpt_template_id: Current CPT template ID
        current_icd_template_id: Current ICD template ID
        best_cpt_template_id: Best CPT template ID
        best_icd_template_id: Best ICD template ID
        status: Job status
        error_message: Error message if failed
    
    Returns:
        True on success, False on failure
    """
    updates = []
    params = []
    
    if phase is not None:
        updates.append("phase = %s")
        params.append(phase)
    
    if iteration is not None:
        updates.append("iteration = %s")
        params.append(iteration)
    
    if cpt_accuracy is not None:
        updates.append("cpt_accuracy = %s")
        params.append(cpt_accuracy)
    
    if icd1_accuracy is not None:
        updates.append("icd1_accuracy = %s")
        params.append(icd1_accuracy)
    
    if best_cpt_accuracy is not None:
        updates.append("best_cpt_accuracy = %s")
        params.append(best_cpt_accuracy)
    
    if best_icd1_accuracy is not None:
        updates.append("best_icd1_accuracy = %s")
        params.append(best_icd1_accuracy)
    
    if current_cpt_template_id is not None:
        updates.append("current_cpt_template_id = %s")
        params.append(current_cpt_template_id)
    
    if current_icd_template_id is not None:
        updates.append("current_icd_template_id = %s")
        params.append(current_icd_template_id)
    
    if best_cpt_template_id is not None:
        updates.append("best_cpt_template_id = %s")
        params.append(best_cpt_template_id)
    
    if best_icd_template_id is not None:
        updates.append("best_icd_template_id = %s")
        params.append(best_icd_template_id)
    
    if status is not None:
        updates.append("status = %s")
        params.append(status)
    
    if error_message is not None:
        updates.append("error_message = %s")
        params.append(error_message)
    
    if not updates:
        return True  # Nothing to update
    
    updates.append("updated_at = CURRENT_TIMESTAMP")
    params.append(job_id)
    
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                query = f"""
                    UPDATE refinement_jobs
                    SET {', '.join(updates)}
                    WHERE job_id = %s
                """
                cur.execute(query, params)
                return True
    except Exception as e:
        logger.error(f"Failed to update refinement job {job_id}: {e}")
        return False


def get_refinement_job(job_id: str):
    """
    Get a refinement job by job_id.
    
    Args:
        job_id: Job identifier
    
    Returns:
        Dictionary with job data or None if not found
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT * FROM refinement_jobs
                    WHERE job_id = %s
                """, (job_id,))
                result = cur.fetchone()
                return dict(result) if result else None
    except Exception as e:
        logger.error(f"Failed to get refinement job {job_id}: {e}")
        return None




if __name__ == "__main__":
    # Test database connection
    print("Testing database connection...")
    if init_database():
        print("✅ Database connection successful!")
        modifiers = get_all_modifiers()
        print(f"Found {len(modifiers)} modifiers in database")
    else:
        print("❌ Database connection failed!")


def save_unified_result(
    job_id: str,
    filename: str,
    supabase_path: str,
    input_zip_supabase_path: Optional[str] = None,
    file_size_bytes: Optional[int] = None,
    row_count: Optional[int] = None,
    enabled_extraction: bool = False,
    enabled_cpt: bool = False,
    enabled_icd: bool = False,
    extraction_model: Optional[str] = None,
    cpt_vision_model: Optional[str] = None,
    icd_vision_model: Optional[str] = None,
    worktracker_group: Optional[str] = None,
    worktracker_batch: Optional[str] = None,
    status: str = "completed"
):
    """
    Save unified processing result metadata to database.

    Returns:
        Result ID on success, None on failure
    """
    try:
        from datetime import datetime, timedelta

        expires_at = datetime.now() + timedelta(days=7)

        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    INSERT INTO unified_results (
                        job_id, filename, supabase_path, input_zip_supabase_path,
                        file_size_bytes, row_count, enabled_extraction, enabled_cpt, enabled_icd,
                        extraction_model, cpt_vision_model, icd_vision_model,
                        worktracker_group, worktracker_batch,
                        status, expires_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (job_id)
                    DO UPDATE SET
                        filename = EXCLUDED.filename,
                        supabase_path = EXCLUDED.supabase_path,
                        input_zip_supabase_path = EXCLUDED.input_zip_supabase_path,
                        file_size_bytes = EXCLUDED.file_size_bytes,
                        row_count = EXCLUDED.row_count,
                        enabled_extraction = EXCLUDED.enabled_extraction,
                        enabled_cpt = EXCLUDED.enabled_cpt,
                        enabled_icd = EXCLUDED.enabled_icd,
                        extraction_model = EXCLUDED.extraction_model,
                        cpt_vision_model = EXCLUDED.cpt_vision_model,
                        icd_vision_model = EXCLUDED.icd_vision_model,
                        worktracker_group = EXCLUDED.worktracker_group,
                        worktracker_batch = EXCLUDED.worktracker_batch,
                        status = EXCLUDED.status,
                        expires_at = EXCLUDED.expires_at
                    RETURNING id
                """, (
                    job_id, filename, supabase_path, input_zip_supabase_path,
                    file_size_bytes, row_count, enabled_extraction, enabled_cpt, enabled_icd,
                    extraction_model, cpt_vision_model, icd_vision_model,
                    worktracker_group or None, worktracker_batch or None,
                    status, expires_at
                ))
                result = cur.fetchone()
                return result['id'] if result else None
    except Exception as e:
        logger.error(f"Failed to save unified result for job {job_id}: {e}")
        return None


# ========== Base Prompts CRUD ==========

def get_all_base_prompts():
    """Get all base prompts."""
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM base_prompts ORDER BY prompt_type, name")
            return cur.fetchall()

def get_base_prompt(name: str):
    """Get a specific base prompt by name."""
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM base_prompts WHERE name = %s", (name,))
            return cur.fetchone()

def upsert_base_prompt(name: str, prompt_type: str, content: str, description: str = ''):
    """Create or update a base prompt."""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO base_prompts (name, prompt_type, content, description, updated_at)
                VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP)
                ON CONFLICT (name) DO UPDATE SET
                    content = EXCLUDED.content,
                    description = EXCLUDED.description,
                    prompt_type = EXCLUDED.prompt_type,
                    updated_at = CURRENT_TIMESTAMP
            """, (name, prompt_type, content, description))
            conn.commit()
            return True

def delete_base_prompt(name: str):
    """Delete a base prompt by name."""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM base_prompts WHERE name = %s", (name,))
            conn.commit()
            return cur.rowcount > 0


def get_all_unified_results(page: int = 1, page_size: int = 50, search_group: Optional[str] = None, search_batch: Optional[str] = None):
    """
    Get all unified processing results with pagination and optional search by group/batch.
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                conditions = []
                params = []
                if search_group:
                    conditions.append("worktracker_group ILIKE %s")
                    params.append(f"%{search_group}%")
                if search_batch:
                    conditions.append("worktracker_batch ILIKE %s")
                    params.append(f"%{search_batch}%")

                where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

                cur.execute(f"SELECT COUNT(*) as total FROM unified_results {where_clause}", params)
                total = cur.fetchone()['total']

                offset = (page - 1) * page_size
                cur.execute(f"""
                    SELECT id, job_id, filename, supabase_path, input_zip_supabase_path,
                           file_size_bytes, row_count, enabled_extraction, enabled_cpt,
                           enabled_icd, extraction_model, cpt_vision_model, icd_vision_model,
                           worktracker_group, worktracker_batch,
                           created_at, expires_at, status
                    FROM unified_results
                    {where_clause}
                    ORDER BY created_at DESC
                    LIMIT %s OFFSET %s
                """, params + [page_size, offset])

                results = [dict(row) for row in cur.fetchall()]

                return {
                    'results': results,
                    'total': total,
                    'page': page,
                    'page_size': page_size
                }
    except Exception as e:
        logger.error(f"Failed to get unified results: {e}")
        return {'results': [], 'total': 0, 'page': page, 'page_size': page_size}


def get_unified_result(job_id: str):
    """
    Get a specific unified result by job_id.
    
    Args:
        job_id: Job identifier
    
    Returns:
        Dictionary with result data or None if not found
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT id, job_id, filename, supabase_path, input_zip_supabase_path,
                           file_size_bytes, row_count, enabled_extraction, enabled_cpt,
                           enabled_icd, extraction_model, cpt_vision_model, icd_vision_model,
                           worktracker_group, worktracker_batch,
                           created_at, expires_at, status
                    FROM unified_results
                    WHERE job_id = %s
                """, (job_id,))
                result = cur.fetchone()
                return dict(result) if result else None
    except Exception as e:
        logger.error(f"Failed to get unified result for job {job_id}: {e}")
        return None


def delete_expired_unified_results():
    """
    Delete unified results that have expired (older than 3 days).
    Also deletes the associated files from Supabase Storage.

    Returns:
        Number of deleted results
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT id, job_id, supabase_path, input_zip_supabase_path
                    FROM unified_results
                    WHERE expires_at < CURRENT_TIMESTAMP
                """)

                expired_results = cur.fetchall()
                deleted_count = 0

                for result in expired_results:
                    # Delete from Supabase Storage
                    paths_to_delete = [
                        p for p in [result.get('supabase_path'), result.get('input_zip_supabase_path')]
                        if p
                    ]
                    if paths_to_delete:
                        try:
                            delete_from_supabase(paths_to_delete)
                            logger.info(f"Deleted Supabase files: {paths_to_delete}")
                        except Exception as e:
                            logger.warning(f"Failed to delete Supabase files {paths_to_delete}: {e}")

                    cur.execute("DELETE FROM unified_results WHERE id = %s", (result['id'],))
                    deleted_count += 1
                    logger.info(f"Deleted expired unified result: {result['job_id']}")

                return deleted_count
    except Exception as e:
        logger.error(f"Failed to delete expired unified results: {e}")
        return 0

