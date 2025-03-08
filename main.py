# Import necessary libraries
import json
import requests
import time
import os
import psycopg2
from psycopg2 import sql
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global variable to keep track of the total number of tokens
total_tokens = 0

# PostgreSQL Connection Parameters
DB_CONFIG = {
    'dbname': 'your_database',
    'user': 'your_username',
    'password': 'your_password',
    'host': 'localhost',
    'port': '5432'
}

# Function to establish database connection
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

# Function to extract database schema
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
            
            # Get all tables
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

# Function to validate SQL query against the database
def validate_sql(query):
    """
    Validate if the SQL query is syntactically correct and can be executed.
    
    :param query: SQL query to validate
    :return: Tuple (is_valid, error_message)
    """
    try:
        conn = get_db_connection()
        if conn:
            cur = conn.cursor()
            # Using a transaction to validate the query without committing changes
            conn.autocommit = False
            try:
                cur.execute("BEGIN")
                cur.execute(query)
                conn.rollback()
                return True, None
            except Exception as e:
                conn.rollback()
                return False, str(e)
            finally:
                cur.close()
                conn.close()
        else:
            return False, "Database connection error"
    except Exception as e:
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

# Function to generate SQL statements from NL queries
def generate_sqls(data, db_schema):
    """
    Generate SQL statements from the natural language queries.
    
    :param data: List of dictionaries with NL queries
    :param db_schema: Database schema information
    :return: List of dictionaries with NL queries and generated SQL
    """
    global total_tokens
    results = []
    
    # Get your API key from environment variable
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        logger.error("GROQ API key not found. Set the GROQ_API_KEY environment variable.")
        return results
    
    # Choose an appropriate model within the 7B parameter limit
    model = "llama-3.1-8b-instant"  # You can choose a different model if needed
    
    # Create a concise schema representation for the prompt
    schema_text = "Database Schema:\n"
    schema_text += "Tables:\n"
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
    
    # Process each NL query
    for item in data:
        nl_query = item['NL']
        
        # Create system prompt
        system_prompt = f"""You are an expert SQL query generator. Your task is to convert natural language questions into correct and efficient PostgreSQL queries.

{schema_text}

Guidelines:
1. Generate only the SQL query, no explanations.
2. Ensure proper JOIN syntax and table relationships.
3. Use aliasing for tables if needed for clarity.
4. Include proper ORDER BY, GROUP BY, or filtering clauses as needed.
5. The query must be executable in PostgreSQL.
6. Return only the SQL query, nothing else."""
        
        # Construct messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Convert this natural language query to SQL: \"{nl_query}\""}
        ]
        
        # Call the Groq API
        response_json, tokens = call_groq_api(api_key, model, messages)
        total_tokens += tokens
        
        try:
            sql_query = response_json['choices'][0]['message']['content'].strip()
            
            # Validate the query
            is_valid, error_message = validate_sql(sql_query)
            
            if not is_valid:
                # Try to fix the query if it's invalid
                fix_messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Convert this natural language query to SQL: \"{nl_query}\""},
                    {"role": "assistant", "content": sql_query},
                    {"role": "user", "content": f"The SQL query has an error: {error_message}. Please fix it."}
                ]
                
                fix_response, fix_tokens = call_groq_api(api_key, model, fix_messages)
                total_tokens += fix_tokens
                sql_query = fix_response['choices'][0]['message']['content'].strip()
            
            # Add result to output list
            results.append({"NL": nl_query, "Query": sql_query})
            
            # Log progress
            logger.info(f"Processed NL query: {nl_query[:50]}...")
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            # Add empty result in case of error
            results.append({"NL": nl_query, "Query": ""})
    
    return results

# Function to correct SQL statements
def correct_sqls(data, db_schema):
    """
    Correct SQL statements if necessary.
    
    :param data: List of dictionaries with incorrect SQL statements
    :param db_schema: Database schema information
    :return: List of dictionaries with corrected SQL statements
    """
    global total_tokens
    results = []
    
    # Get your API key from environment variable
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        logger.error("GROQ API key not found. Set the GROQ_API_KEY environment variable.")
        return results
    
    # Choose an appropriate model within the 7B parameter limit
    model = "llama-3.1-8b-instant"  # You can choose a different model if needed
    
    # Create a concise schema representation for the prompt
    schema_text = "Database Schema:\n"
    schema_text += "Tables:\n"
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
    
    # Process each incorrect SQL query
    for item in data:
        incorrect_query = item['IncorrectQuery']
        
        # First, determine what's wrong with the query
        validation_result, error_message = validate_sql(incorrect_query)
        
        # Create system prompt
        system_prompt = f"""You are an expert SQL query corrector. Your task is to fix incorrect PostgreSQL queries.

{schema_text}

Guidelines:
1. Identify and fix any syntax errors.
2. Check for incorrect table or column names.
3. Ensure proper JOIN syntax and table relationships.
4. Fix any logical errors in the query.
5. The corrected query must be executable in PostgreSQL.
6. Return only the corrected SQL query, nothing else."""
        
        # Construct messages with error information if available
        if validation_result:
            # If the query is valid but might have logical errors
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Fix any errors in this SQL query: \"{incorrect_query}\""}
            ]
        else:
            # If there's a specific error
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Fix this SQL query that has the following error: {error_message}\n\nQuery: \"{incorrect_query}\""}
            ]
        
        # Call the Groq API
        response_json, tokens = call_groq_api(api_key, model, messages)
        total_tokens += tokens
        
        try:
            corrected_sql = response_json['choices'][0]['message']['content'].strip()
            
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
                corrected_sql = fix_response['choices'][0]['message']['content'].strip()
            
            # Add result to output list
            results.append({"IncorrectQuery": incorrect_query, "CorrectQuery": corrected_sql})
            
            # Log progress
            logger.info(f"Processed incorrect SQL: {incorrect_query[:50]}...")
            
        except Exception as e:
            logger.error(f"Error processing SQL correction: {e}")
            # Add empty result in case of error
            results.append({"IncorrectQuery": incorrect_query, "CorrectQuery": ""})
    
    return results

# Function to call the Groq API
def call_groq_api(api_key, model, messages, temperature=0.0, max_tokens=1000, n=1):
    """
    Call the Groq API to get a response from the language model.
    
    :param api_key: API key for authentication
    :param model: Model name to use
    :param messages: List of message dictionaries
    :param temperature: Temperature for the model
    :param max_tokens: Maximum number of tokens to generate
    :param n: Number of responses to generate
    :return: Response from the API and number of tokens used
    """
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
    
    try:
        response = requests.post(url, headers=headers, json=data)
        response_json = response.json()
        
        # Get token count
        tokens_used = response_json.get('usage', {}).get('completion_tokens', 0)
        
        return response_json, tokens_used
    except Exception as e:
        logger.error(f"API call error: {e}")
        return {"choices": [{"message": {"content": ""}}]}, 0

# Main function
def main():
    # Check for environment variable
    if not os.getenv("GROQ_API_KEY"):
        logger.error("GROQ_API_KEY environment variable not set")
        print("Please set the GROQ_API_KEY environment variable")
        print("Example: export GROQ_API_KEY=your_api_key")
        return 0, 0
    
    # Specify the paths to input files
    input_file_path_1 = 'input_nl_to_sql.json'
    input_file_path_2 = 'input_sql_correction.json'
    
    # Load data from input files
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
    
    # Generate SQL statements
    start = time.time()
    logger.info("Starting SQL generation from natural language...")
    sql_statements = generate_sqls(data_1, db_schema)
    generate_sqls_time = time.time() - start
    logger.info(f"SQL generation completed in {generate_sqls_time:.2f} seconds")
    
    # Correct SQL statements
    start = time.time()
    logger.info("Starting SQL correction...")
    corrected_sqls = correct_sqls(data_2, db_schema)
    correct_sqls_time = time.time() - start
    logger.info(f"SQL correction completed in {correct_sqls_time:.2f} seconds")
    
    # Ensure outputs match input lengths
    if len(data_1) > 0:
        assert len(data_1) == len(sql_statements), "Number of generated SQL statements doesn't match input"
    
    if len(data_2) > 0:
        assert len(data_2) == len(corrected_sqls), "Number of corrected SQL statements doesn't match input"
    
    # Write outputs to files
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