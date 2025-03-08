import json
import requests
import time
import os
import psycopg2
from psycopg2 import sql
import logging
import random
import re

logger = logging.getLogger(__name__)

def get_db_connection(db_config):
    """Establish a connection to the PostgreSQL database."""
    try:
        conn = psycopg2.connect(**db_config)
        return conn
    except Exception as e:
        logger.error(f"Database connection error: {e}")
        return None

def get_database_schema(db_config):
    """Extract the database schema including tables, columns, and relationships."""
    schema = {}
    try:
        conn = get_db_connection(db_config)
        if not conn:
            return None
            
        cur = conn.cursor()
        
        # Get tables
        cur.execute("""
            SELECT table_name FROM information_schema.tables 
            WHERE table_schema = 'public'
        """)
        tables = [table[0] for table in cur.fetchall()]
        schema['tables'] = tables
        
        # Get columns
        schema['columns'] = {}
        for table in tables:
            cur.execute("""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = %s
            """, (table,))
            columns = {col[0]: col[1] for col in cur.fetchall()}
            schema['columns'][table] = columns
        
        # Get primary keys
        schema['primary_keys'] = {}
        for table in tables:
            cur.execute("""
                SELECT kcu.column_name 
                FROM information_schema.table_constraints tc
                JOIN information_schema.key_column_usage kcu
                ON tc.constraint_name = kcu.constraint_name
                WHERE tc.constraint_type = 'PRIMARY KEY' 
                AND tc.table_name = %s
            """, (table,))
            pks = [pk[0] for pk in cur.fetchall()]
            schema['primary_keys'][table] = pks
        
        # Get foreign keys
        schema['foreign_keys'] = []
        cur.execute("""
            SELECT
                tc.table_name, 
                kcu.column_name,
                ccu.table_name AS foreign_table_name,
                ccu.column_name AS foreign_column_name
            FROM information_schema.table_constraints AS tc
            JOIN information_schema.key_column_usage AS kcu
            ON tc.constraint_name = kcu.constraint_name
            JOIN information_schema.constraint_column_usage AS ccu
            ON ccu.constraint_name = tc.constraint_name
            WHERE tc.constraint_type = 'FOREIGN KEY'
        """)
        for fk in cur.fetchall():
            schema['foreign_keys'].append({
                'table': fk[0],
                'column': fk[1],
                'foreign_table': fk[2],
                'foreign_column': fk[3]
            })
        
        cur.close()
        conn.close()
        
        return schema
    except Exception as e:
        logger.error(f"Error fetching schema: {e}")
        return None

def format_schema_text(db_schema):
    """Format database schema into a readable string for prompts."""
    schema_text = "Database Schema:\nTables:\n"
    for table in db_schema['tables']:
        schema_text += f"- {table} ("
        column_texts = []
        for col, dtype in db_schema['columns'][table].items():
            is_pk = col in db_schema['primary_keys'].get(table, [])
            pk_str = " PRIMARY KEY" if is_pk else ""
            column_texts.append(f"{col} {dtype}{pk_str}")
        schema_text += ", ".join(column_texts)
        schema_text += ")\n"
    
    schema_text += "\nRelationships:\n"
    for fk in db_schema['foreign_keys']:
        schema_text += f"- {fk['table']}.{fk['column']} references {fk['foreign_table']}.{fk['foreign_column']}\n"
    
    return schema_text

def clean_sql_response(text):
    """Clean the SQL query from model response."""
    if '```' in text:
        code_block_pattern = r'```(?:sql)?(.*?)```'
        matches = re.findall(code_block_pattern, text, re.DOTALL)
        if matches:
            text = matches[0]
    
    text = text.strip()
    
    sql_keywords = ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'ALTER', 'DROP', 'WITH']
    lines = text.split('\n')
    sql_lines = []
    started = False
    
    for line in lines:
        line_upper = line.strip().upper()
        if not started and any(line_upper.startswith(keyword) for keyword in sql_keywords):
            started = True
            sql_lines.append(line)
        elif started:
            sql_lines.append(line)
    
    return '\n'.join(sql_lines) if sql_lines else text

def validate_sql(query, db_config):
    """Validate if the SQL query is syntactically correct."""
    try:
        query = clean_sql_response(query)
        if query.endswith(';'):
            query = query[:-1]
        
        conn = get_db_connection(db_config)
        if not conn:
            return False, "Database connection error"
            
        cur = conn.cursor()
        conn.autocommit = False
        try:
            cur.execute("BEGIN")
            cur.execute(f"EXPLAIN {query}")
            conn.rollback()
            return True, None
        except Exception as e:
            conn.rollback()
            return False, str(e)
        finally:
            cur.close()
            conn.close()
    except Exception as e:
        return False, str(e)

def call_groq_api(api_key, model, messages, temperature=0.0, max_tokens=1000, n=1, max_retries=5):
    """Call the Groq API with retry logic."""
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    data = {
        "model": model,
        "messages": messages,
        'temperature': temperature,
        'max_tokens': max_tokens,
        'n': n
    }
    
    retry_count = 0
    base_wait_time = 2
    
    while retry_count <= max_retries:
        try:
            response = requests.post(url, headers=headers, json=data)
            response_json = response.json()
            
            if response.status_code == 429 or (
                'error' in response_json and 
                response_json.get('error', {}).get('code') == 'rate_limit_exceeded'
            ):
                retry_count += 1
                if retry_count > max_retries:
                    logger.error(f"Maximum retries ({max_retries}) exceeded for API call")
                    return {"choices": [{"message": {"content": ""}}]}, 0
                
                wait_time = base_wait_time * (2 ** (retry_count - 1)) * (0.4 + 0.3 * random.random())
                error_message = response_json.get('error', {}).get('message', '')
                wait_time_match = re.search(r'try again in (\d+\.\d+)s', error_message)
                if wait_time_match:
                    wait_time = float(wait_time_match.group(1)) + 0.3
                
                logger.warning(f"Rate limit exceeded. Retrying in {wait_time:.2f} seconds (attempt {retry_count}/{max_retries})")
                time.sleep(wait_time)
                continue
            
            if 'error' in response_json:
                error_message = response_json.get('error', {}).get('message', 'Unknown error')
                logger.error(f"API error: {error_message}")
                return {"choices": [{"message": {"content": ""}}]}, 0
            
            tokens_used = response_json.get('usage', {}).get('completion_tokens', 0)
            return response_json, tokens_used
            
        except Exception as e:
            retry_count += 1
            if retry_count > max_retries:
                logger.error(f"Maximum retries ({max_retries}) exceeded for API call")
                return {"choices": [{"message": {"content": ""}}]}, 0
            
            wait_time = base_wait_time * (2 ** (retry_count - 1)) * (0.8 + 0.4 * random.random())
            time.sleep(wait_time)
    
    return {"choices": [{"message": {"content": ""}}]}, 0

def load_input_file(file_path):
    """Load input file which is a list of dictionaries."""
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data
    except Exception as e:
        logger.error(f"Error loading input file: {e}")
        return []
