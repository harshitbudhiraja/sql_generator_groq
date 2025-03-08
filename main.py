# Import necessary libraries
import json
import requests
import time
import os
import psycopg2
from psycopg2 import sql
import logging
from dotenv import load_dotenv
import random
import re
from vector_store import VectorStore
from utils import (
    get_database_schema, format_schema_text, clean_sql_response, 
    validate_sql, call_groq_api, load_input_file
)
from prompt import get_sql_generation_prompt, get_sql_correction_prompt

load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

total_tokens = 0

DB_CONFIG = {
    'dbname': os.getenv('DB_NAME'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'), 
    'host': os.getenv('DB_HOST'),
    'port': os.getenv('DB_PORT')
}

def get_db_connection():
    """
    Establish a connection to the PostgreSQL database.
    
    :return: Database connection object
    """
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        return conn
    except Exception as e:
        logger.error(f"Database connection error: {e}")
        return None

def get_database_schema():
    """
    Extract the database schema including tables, columns, and relationships.
    
    :return: Dictionary containing database schema information
    """
    schema = {}
    try:
        conn = get_db_connection()
        if conn:
            cur = conn.cursor()
            
            cur.execute("""
                SELECT table_name FROM information_schema.tables 
                WHERE table_schema = 'public'
            """)
            tables = [table[0] for table in cur.fetchall()]
            schema['tables'] = tables
            
            # Get columns for each table
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
        else:
            logger.error("Couldn't establish database connection")
            return None
    except Exception as e:
        logger.error(f"Error fetching schema: {e}")
        return None

# Function to clean SQL responses
def clean_sql_response(text):
    """
    Clean the SQL query from model response by removing markdown formatting,
    extra explanations, and other non-SQL content.
    
    :param text: Raw text from model response
    :return: Cleaned SQL query
    """
    # Remove markdown code block markers
    if '```' in text:
        # Extract content between markdown code blocks if present
        code_block_pattern = r'```(?:sql)?(.*?)```'
        matches = re.findall(code_block_pattern, text, re.DOTALL)
        if matches:
            # Use the first code block found
            text = matches[0]
    
    # Remove any leading/trailing whitespace
    text = text.strip()
    
    # Remove explanatory text before or after the SQL query
    # This is a heuristic - assuming SQL queries start with SELECT, INSERT, UPDATE, etc.
    sql_keywords = ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'ALTER', 'DROP', 'WITH']
    lines = text.split('\n')
    sql_lines = []
    started = False
    
    for line in lines:
        line_upper = line.strip().upper()
        if not started:
            # Check if line starts with SQL keyword
            if any(line_upper.startswith(keyword) for keyword in sql_keywords):
                started = True
                sql_lines.append(line)
        else:
            sql_lines.append(line)
    
    if sql_lines:
        return '\n'.join(sql_lines)
    
    # If no SQL keywords found, return the original cleaned text
    return text

# Function to validate SQL query against the database
def validate_sql(query):
    """
    Validate if the SQL query is syntactically correct and can be executed.
    
    :param query: SQL query to validate
    :return: Tuple (is_valid, error_message)
    """
    try:
        # Clean the query from markdown formatting and other common issues
        # Remove markdown code block markers (```sql and ```)
        if query.strip().startswith('```'):
            # Extract content between markdown code blocks
            lines = query.split('\n')
            filtered_lines = []
            for line in lines:
                if line.strip().startswith('```') or line.strip() == '```':
                    continue
                filtered_lines.append(line)
            query = '\n'.join(filtered_lines)
        
        # Remove trailing semicolons and extra whitespace
        query = query.strip()
        if query.endswith(';'):
            query = query[:-1]
        
        # Log the cleaned query for debugging
        logger.debug(f"Cleaned query for validation: {query}")
        
        conn = get_db_connection()
        if conn:
            cur = conn.cursor()
            # Using a transaction to validate the query without committing changes
            conn.autocommit = False
            try:
                cur.execute("BEGIN")
                # Use EXPLAIN to validate the query without actually executing it
                explain_query = f"EXPLAIN {query}"
                cur.execute(explain_query)
                conn.rollback()
                return True, None
            except Exception as e:
                conn.rollback()
                error_msg = str(e)
                # Log the specific query and error for debugging
                logger.debug(f"Query validation error: {error_msg}")
                logger.debug(f"Failed query: {query}")
                return False, error_msg
            finally:
                cur.close()
                conn.close()
        else:
            return False, "Database connection error"
    except Exception as e:
        logger.error(f"Unexpected error in validate_sql: {str(e)}")
        return False, str(e)

# Function to load input file
def load_input_file(file_path):
    """
    Load input file which is a list of dictionaries.
    
    :param file_path: Path to the input file
    :return: List of dictionaries
    """
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data
    except Exception as e:
        logger.error(f"Error loading input file: {e}")
        return []

# Function to generate SQL statements from NL queries using vector retrieval
def generate_sqls(data, db_schema, db_config, vector_store=None, batch_size=5):
    """
    Generate SQL statements from the natural language queries using vector retrieval when possible.
    
    :param data: List of dictionaries with NL queries
    :param db_schema: Database schema information
    :param vector_store: Vector store for similar query retrieval
    :param batch_size: Size of batches to process to manage rate limits
    :return: List of dictionaries with NL queries and generated SQL
    """
    global total_tokens
    results = []
    
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        logger.error("GROQ API key not found.")
        return results
    
    model = "llama-3.1-8b-instant"
    schema_text = format_schema_text(db_schema)
    
    # Process in batches to manage rate limits
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        logger.info(f"Processing batch {i//batch_size + 1}/{(len(data) + batch_size - 1)//batch_size}")
        
        # Process each NL query in the batch
        for item in batch:
            nl_query = item['NL']
            
            # Try vector retrieval first if available
            retrieved_sql = None
            few_shot_examples = []
            if vector_store and vector_store.nl_to_sql_index is not None:
                similar_queries = vector_store.find_similar_nl_queries(nl_query)
                if similar_queries:
                    # Use the most similar query's SQL as a basis
                    best_match = similar_queries[0]
                    logger.info(f"Found similar query with similarity score: {best_match['similarity']:.4f}")
                    
                    # If similarity is very high, we can use it directly
                    if best_match['similarity'] > 0.9:
                        retrieved_sql = best_match['sql_query']
                        logger.info("Using retrieved SQL directly due to high similarity")
                    else:
                        # Use few-shot learning with examples for better results
                        for match in similar_queries[:3]:  # Use top 3 matches
                            few_shot_examples.append({
                                "role": "user", 
                                "content": f"Convert this natural language query to SQL: \"{match['nl_query']}\""
                            })
                            few_shot_examples.append({
                                "role": "assistant", 
                                "content": match['sql_query']
                            })
            
            # If no suitable SQL was retrieved or vector store not available, use the API
            if retrieved_sql is None:
                # Create system prompt
                system_prompt = get_sql_generation_prompt(schema_text)
                
                messages = [
                    {"role": "system", "content": system_prompt},
                ]
                
                if few_shot_examples:
                    messages.extend(few_shot_examples)
                
                messages.append({"role": "user", "content": f"Convert this natural language query to SQL: \"{nl_query}\""})
                
                response_json, tokens = call_groq_api(api_key, model, messages)
                total_tokens += tokens
                
                try:
                    sql_query = response_json['choices'][0]['message']['content'].strip()
                    
                    sql_query = clean_sql_response(sql_query)
                    logger.debug(f"Generated SQL query: {sql_query}")
                    
                    is_valid, error_message = validate_sql(sql_query)
                    
                    if not is_valid:
                        fix_messages = [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": f"Convert this natural language query to SQL: \"{nl_query}\""},
                            {"role": "assistant", "content": sql_query},
                            {"role": "user", "content": f"The SQL query has an error: {error_message}. Please fix it."}
                        ]
                        
                        fix_response, fix_tokens = call_groq_api(api_key, model, fix_messages)
                        total_tokens += fix_tokens
                        sql_query = clean_sql_response(fix_response['choices'][0]['message']['content'].strip())
                    
                    results.append({"NL": nl_query, "Query": sql_query})
                    
                    logger.info(f"Processed NL query: {nl_query[:50]}...")
                    
                except Exception as e:
                    logger.error(f"Error processing query: {e}")
                    results.append({"NL": nl_query, "Query": ""})
            else:
                results.append({"NL": nl_query, "Query": retrieved_sql})
                logger.info(f"Used vector retrieval for query: {nl_query[:50]}...")
        
        if i + batch_size < len(data):
            time.sleep(1)
    
    return results

# Function to correct SQL statements using vector retrieval
def correct_sqls(data, db_schema, db_config, vector_store=None, batch_size=5):
    """
    Correct SQL statements using vector retrieval when possible.
    
    :param data: List of dictionaries with incorrect SQL statements
    :param db_schema: Database schema information
    :param vector_store: Vector store for similar query retrieval
    :param batch_size: Size of batches to process to manage rate limits
    :return: List of dictionaries with corrected SQL statements
    """
    global total_tokens
    results = []
    
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        logger.error("GROQ API key not found. Set the GROQ_API_KEY environment variable.")
        return results
    
    model = "llama-3.1-8b-instant"
    schema_text = format_schema_text(db_schema)
    
    # Process in batches to manage rate limits
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        logger.info(f"Processing correction batch {i//batch_size + 1}/{(len(data) + batch_size - 1)//batch_size}")
        
        # Process each incorrect SQL query in the batch
        for item in batch:
            incorrect_query = item['IncorrectQuery']
            
            # Try vector retrieval first if available
            retrieved_correction = None
            few_shot_examples = []
            if vector_store and vector_store.incorrect_sql_index is not None:
                similar_queries = vector_store.find_similar_incorrect_queries(incorrect_query)
                if similar_queries:
                    # Use the most similar query's correction as a basis
                    best_match = similar_queries[0]
                    logger.info(f"Found similar incorrect query with similarity score: {best_match['similarity']:.4f}")
                    
                    # If similarity is very high, we can use it directly
                    if best_match['similarity'] > 0.9:
                        retrieved_correction = best_match['correct_query']
                        logger.info("Using retrieved correction directly due to high similarity")
                    else:
                        # Use few-shot learning with examples for better results
                        for match in similar_queries[:3]:  # Use top 3 matches
                            few_shot_examples.append({
                                "role": "user", 
                                "content": f"Fix this SQL query: \"{match['incorrect_query']}\""
                            })
                            few_shot_examples.append({
                                "role": "assistant", 
                                "content": match['correct_query']
                            })
            
            # If no suitable correction was retrieved or vector store not available, use the API
            if retrieved_correction is None:
                # First, determine what's wrong with the query
                validation_result, error_message = validate_sql(incorrect_query)
                
                # Create system prompt
                system_prompt = get_sql_correction_prompt(schema_text)
                
                # Construct messages with error information if available
                messages = [
                    {"role": "system", "content": system_prompt},
                ]
                
                # Add few-shot examples if available
                if few_shot_examples:
                    messages.extend(few_shot_examples)
                
                if validation_result:
                    # If the query is valid but might have logical errors
                    messages.append({"role": "user", "content": f"Fix any errors in this SQL query: \"{incorrect_query}\""})
                else:
                    # If there's a specific error
                    messages.append({"role": "user", "content": f"Fix this SQL query that has the following error: {error_message}\n\nQuery: \"{incorrect_query}\""})
                
                # Call the Groq API
                response_json, tokens = call_groq_api(api_key, model, messages)
                total_tokens += tokens
                
                try:
                    corrected_sql = response_json['choices'][0]['message']['content'].strip()
                    
                    # Clean the SQL query to remove markdown and other formatting
                    corrected_sql = clean_sql_response(corrected_sql)
                    
                    # Validate the corrected query
                    is_valid, new_error_message = validate_sql(corrected_sql)
                    
                    if not is_valid:
                        # Try once more with the new error message
                        fix_messages = [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": f"Fix this SQL query: \"{incorrect_query}\""},
                            {"role": "assistant", "content": corrected_sql},
                            {"role": "user", "content": f"The corrected SQL query still has an error: {new_error_message}. Please fix it again."}
                        ]
                        
                        fix_response, fix_tokens = call_groq_api(api_key, model, fix_messages)
                        total_tokens += fix_tokens
                        corrected_sql = clean_sql_response(fix_response['choices'][0]['message']['content'].strip())
                    
                    # Add result to output list
                    results.append({"IncorrectQuery": incorrect_query, "CorrectQuery": corrected_sql})
                    
                    # Log progress
                    logger.info(f"Processed incorrect SQL: {incorrect_query[:50]}...")
                    
                except Exception as e:
                    logger.error(f"Error processing SQL correction: {e}")
                    # Add empty result in case of error
                    results.append({"IncorrectQuery": incorrect_query, "CorrectQuery": ""})
            else:
                # Use the retrieved correction directly
                results.append({"IncorrectQuery": incorrect_query, "CorrectQuery": retrieved_correction})
                logger.info(f"Used vector retrieval for SQL correction: {incorrect_query[:50]}...")
        
        if i + batch_size < len(data):
            time.sleep(1)
    
    return results

def main():

    if not os.getenv("GROQ_API_KEY"):
        logger.error("GROQ_API_KEY environment variable not set")
        print("Please set the GROQ_API_KEY environment variable")
        return 0, 0
    
    input_file_path_1 = 'data/train_generate_task.json'
    input_file_path_2 = 'data/train_query_correction_task.json'
    
    data_1 = load_input_file(input_file_path_1)
    data_2 = load_input_file(input_file_path_2)
    
    if not data_1 and not data_2:
        logger.error("No input data found. Please check the input files.")
        return 0, 0
    
    # Get database schema
    logger.info("Extracting database schema...")
    db_schema = get_database_schema()
    
    if not db_schema:
        logger.error("Failed to extract database schema.")
        return 0, 0
    
    logger.info("Initializing vector store...")
    vector_store = VectorStore()
    
    indexes_loaded = vector_store.load_indexes()
    slice_size = 5
    if  not indexes_loaded:

        logger.info("Building vector indexes from training data...")
        if data_1:
            vector_store.build_nl_to_sql_index(data_1[slice_size:])
        if data_2:
            vector_store.build_sql_correction_index(data_2[slice_size:])
    
    test_data_1 = data_1
    test_data_2 = data_2
    

    start = time.time()
    logger.info("Starting SQL generation from natural language...")
    sql_statements = generate_sqls(test_data_1, db_schema, DB_CONFIG, vector_store=vector_store)
    generate_sqls_time = time.time() - start
    logger.info(f"SQL generation completed in {generate_sqls_time:.2f} seconds")
    
    start = time.time()
    logger.info("Starting SQL correction...")
    corrected_sqls = correct_sqls(test_data_2, db_schema, DB_CONFIG, vector_store=vector_store)
    correct_sqls_time = time.time() - start
    logger.info(f"SQL correction completed in {correct_sqls_time:.2f} seconds")
    
    if len(test_data_1) > 0:
        assert len(test_data_1) == len(sql_statements), "Number of generated SQL statements doesn't match input"
    
    if len(test_data_2) > 0:
        assert len(test_data_2) == len(corrected_sqls), "Number of corrected SQL statements doesn't match input"
    
    logger.info("Writing results to output files...")
    
    with open('output_sql_generation_task.json', 'w') as f:
        json.dump(sql_statements, f, indent=2)
    
    with open('output_sql_correction_task.json', 'w') as f:
        json.dump(corrected_sqls, f, indent=2)
    
    logger.info(f"Total tokens used: {total_tokens}")
    
    return generate_sqls_time, correct_sqls_time

if __name__ == "__main__":
    try:
        generate_sqls_time, correct_sqls_time = main()
        print(f"Time taken to generate SQLs: {generate_sqls_time:.2f} seconds")
        print(f"Time taken to correct SQLs: {correct_sqls_time:.2f} seconds")
        print(f"Total tokens: {total_tokens}")
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        print(f"An error occurred: {e}")