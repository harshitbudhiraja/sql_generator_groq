import psycopg2
from psycopg2 import sql

'''
These are a few database functions that you can use to create and manipulate the database.
'''

def create_tables_from_sql_file(sql_file_path, db_name, user, password, host='localhost', port='5432'):
    # Connect to PostgreSQL server
    conn = psycopg2.connect(dbname=db_name, user=user, password=password, host=host, port=port)
    conn.autocommit = True
    cursor = conn.cursor()

    # Read SQL file
    with open(sql_file_path, 'r') as file:
        sql_commands = file.read()

    # Execute SQL commands
    # Split commands and execute each separately
    sql_statements = sql_commands.split(';')
    for statement in sql_statements:
        # Skip empty statements
        if statement.strip():
            try:
                cursor.execute(sql.SQL(statement))
                print(f"Successfully executed statement.", statement)
            except Exception as e:
                print(f"Error executing statement: {e}")
                print(f"Failed statement: {statement[:100]}...")
                continue
    print("Finished executing all SQL commands.")
    
    cursor.close()
    conn.close()

# List all tables
def list_all_tables(db_name, user, password, host='localhost', port='5432'):
    conn = psycopg2.connect(dbname=db_name, user=user, password=password, host=host, port=port)
    cursor = conn.cursor()
    cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='public'")
    tables = cursor.fetchall()
    return [table[0] for table in tables]

# Get table schema
def get_table_schema(db_name, user, password, host='localhost', port='5432', table_name=''):
    conn = psycopg2.connect(dbname=db_name, user=user, password=password, host=host, port=port)
    cursor = conn.cursor()
    cursor.execute(f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{table_name}'")
    columns = cursor.fetchall()
    return {col[0]: col[1] for col in columns}

# Function to delete all tables
def delete_all_tables(db_name, user, password, host='localhost', port='5432'):
    conn = psycopg2.connect(dbname=db_name, user=user, password=password, host=host, port=port)
    cursor = conn.cursor()
    cursor.execute("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                ORDER BY table_name;
            """)
    tables = cursor.fetchall()
    
    for table in tables:    
        cursor.execute(f"DROP TABLE IF EXISTS {table[0]} CASCADE")
        print(f"Dropped table: {table[0]}")
        print(" -------------------------------- ")
        
    # List of enums you wish to delete
    enums_to_delete = [
        'wishlist_status_enum',
        'priority_level_enum',
        'added_from_source_enum'
    ]
    
    for enum in enums_to_delete:
        cursor.execute(f"DROP TYPE IF EXISTS {enum}")
        print(f"Dropped enum: {enum}")
        print(" -------------------------------- ")
        
    conn.commit()
    cursor.close()
    conn.close()

if __name__ == "__main__":
    # Define your database connection parameters
    db_name = 'adobe_emerge'
    user = 'postgres'
    password = '1234'
    sql_file_path = '/Users/anikamurarka/Desktop/sql_generator_groq/data/data.sql'

    # Run this to get started !
    create_tables_from_sql_file(sql_file_path, db_name, user, password)
    
    #delete_all_tables(db_name, user, password)
    