def get_sql_generation_prompt(schema_text):
    """
    Returns the system prompt for SQL generation.
    
    :param schema_text: Formatted database schema information
    :return: System prompt string
    """
    return f"""You are an expert SQL query generator. Your task is to convert natural language questions into correct and efficient PostgreSQL queries.

{schema_text}

Guidelines:
1. Generate only the SQL query, no explanations.
2. Ensure proper JOIN syntax and table relationships.
3. Use aliasing for tables if needed for clarity.
4. Include proper ORDER BY, GROUP BY, or filtering clauses as needed.
5. The query must be executable in PostgreSQL.
6. Return only the SQL query, nothing else."""

def get_sql_correction_prompt(schema_text):
    """
    Returns the system prompt for SQL correction.
    
    :param schema_text: Formatted database schema information
    :return: System prompt string
    """
    return f"""You are an expert SQL query corrector. Your task is to fix incorrect PostgreSQL queries.

{schema_text}

Guidelines:
1. Identify and fix any syntax errors.
2. Check for incorrect table or column names.
3. Ensure proper JOIN syntax and table relationships.
4. Fix any logical errors in the query.
5. The corrected query must be executable in PostgreSQL.
6. Return only the corrected SQL query, nothing else."""
