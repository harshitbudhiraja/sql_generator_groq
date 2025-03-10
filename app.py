import streamlit as st
import json
import sys
import logging
from main import get_database_schema, generate_sqls, correct_sqls, VectorStore

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def initialize_session_state():
    """Initialize session state variables if they don't exist"""
    if 'db_schema' not in st.session_state:
        st.session_state.db_schema = None
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None
    if 'token_usage' not in st.session_state:
        st.session_state.token_usage = {
            'total': 0,
            'generate': 0,
            'correct': 0,
            'history': []
        }

def load_schema():
    """Load database schema and vector store"""
    with st.spinner('Loading database schema...'):
        db_schema = get_database_schema()
        if db_schema:
            st.session_state.db_schema = db_schema
            st.success('Database schema loaded successfully!')
            
            # Initialize vector store
            st.session_state.vector_store = VectorStore()
            if st.session_state.vector_store.load_indexes():
                st.success('Vector indexes loaded successfully!')
            else:
                st.warning('Vector indexes not found. They will be built when processing queries.')
        else:
            st.error('Failed to load database schema. Please check your database connection.')

def update_token_usage(operation_type, tokens_used, query_text=None):
    """Update token usage statistics"""
    st.session_state.token_usage['total'] += tokens_used
    st.session_state.token_usage[operation_type] += tokens_used
    
    # Add to history with timestamp
    from datetime import datetime
    st.session_state.token_usage['history'].append({
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'operation': operation_type,
        'tokens': tokens_used,
        'query': query_text[:50] + '...' if query_text and len(query_text) > 50 else query_text
    })

def main():
    st.set_page_config(
        page_title="SQL Query Assistant",
        page_icon="🔍",
        layout="wide"
    )

    st.title("SQL Query Assistant 🔍")
    
    # Initialize session state
    initialize_session_state()
    
    # Sidebar
    with st.sidebar:
        st.header("Controls")
        if st.button("Load/Reload Schema"):
            load_schema()
        
        st.divider()
        
        # Token usage display in sidebar
        
        # Token usage history
        if st.session_state.token_usage['history']:
            with st.expander("Usage History"):
                for entry in reversed(st.session_state.token_usage['history']):
                    st.text(f"{entry['timestamp']} - {entry['operation']}")
                    st.text(f"Tokens: {entry['tokens']}")
                    if entry['query']:
                        st.text(f"Query: {entry['query']}")
                    st.divider()
        
        st.divider()
        st.markdown("""
        ### Instructions
        1. Load the database schema first
        2. Choose your task (Generate or Correct SQL)
        3. Enter your input or upload a file
        4. Process and view results
        """)

    # Main content
    if st.session_state.db_schema is None:
        st.info("Please load the database schema using the button in the sidebar to begin.")
        return

    # Create tabs for different functionalities
    tab1, tab2 = st.tabs(["Generate SQL", "Correct SQL"])

    with tab1:
        st.header("Generate SQL from Natural Language")
        
        # Input method selection
        input_method = st.radio(
            "Choose input method:",
            ["Direct Input", "File Upload"],
            key="generate_input_method"
        )

        if input_method == "Direct Input":
            nl_query = st.text_area("Enter your natural language query:", height=100)
            if st.button("Generate SQL", key="generate_single"):
                if nl_query:
                    with st.spinner('Generating SQL...'):
                        result, tokens = generate_sqls(
                            [{"NL": nl_query}],
                            st.session_state.db_schema,
                            st.session_state.vector_store
                        )
                        
                        # Assume the result includes token usage information
                        # If your generate_sqls function doesn't return token info,
                        # modify it to include this data or estimate it here
                        tokens_used =tokens
                        if tokens_used == 0:
                            # If not available, you can estimate based on input/output length
                            # This is a very rough estimation and should be replaced with actual token counts
                            tokens_used = len(nl_query) // 4
                        
                        update_token_usage('generate', tokens_used, nl_query)
                        
                        if result and result[0]["Query"]:
                            st.code(result[0]["Query"], language="sql")
                        else:
                            st.error("Failed to generate SQL query.")
                else:
                    st.warning("Please enter a query.")

        else:  # File Upload
            uploaded_file = st.file_uploader("Upload JSON file with NL queries", type="json", key="generate_file")
            if uploaded_file:
                data = json.load(uploaded_file)
                if st.button("Process File", key="generate_file_button"):
                    with st.spinner('Processing queries...'):
                        results, tokens = generate_sqls(
                            data,
                            st.session_state.db_schema,
                            st.session_state.vector_store
                        )
                        
                        # Estimate token usage for batch processing
                        total_tokens = 0
                        for item in data:
                            # Simple estimation - replace with actual token counting
                            total_tokens += tokens
                        
                        update_token_usage('generate', total_tokens, f"Batch processing {len(data)} queries")
                        
                        # Display results in an expandable section
                        for i, result in enumerate(results, 1):
                            with st.expander(f"Query {i}: {result['NL'][:50]}..."):
                                st.text("Natural Language Query:")
                                st.write(result['NL'])
                                st.text("Generated SQL:")
                                st.code(result['Query'], language="sql")
                        
                        # Offer download of results
                        st.download_button(
                            "Download Results",
                            json.dumps(results, indent=2),
                            "generated_sql_results.json",
                            "application/json"
                        )

    with tab2:
        st.header("Correct SQL Queries")
        
        input_method = st.radio(
            "Choose input method:",
            ["Direct Input", "File Upload"],
            key="correct_input_method"
        )

        if input_method == "Direct Input":
            incorrect_sql = st.text_area("Enter the SQL query to correct:", height=100)
            if st.button("Correct SQL", key="correct_single"):
                if incorrect_sql:
                    with st.spinner('Correcting SQL...'):
                        result, tokens = correct_sqls(
                            [{"IncorrectQuery": incorrect_sql}],
                            st.session_state.db_schema,
                            st.session_state.vector_store
                        )
                        
                        # Estimate tokens used
                        tokens_used = tokens
                        if tokens_used == 0:
                            # Rough estimation if not available
                            tokens_used = len(incorrect_sql) // 4
                        
                        update_token_usage('correct', tokens_used, incorrect_sql)
                        
                        if result and result[0]["CorrectQuery"]:
                            st.code(result[0]["CorrectQuery"], language="sql")
                        else:
                            st.error("Failed to correct SQL query.")
                else:
                    st.warning("Please enter a query.")

        else:  # File Upload
            uploaded_file = st.file_uploader("Upload JSON file with incorrect SQL queries", type="json", key="correct_file")
            if uploaded_file:
                data = json.load(uploaded_file)
                if st.button("Process File", key="correct_file_button"):
                    with st.spinner('Processing corrections...'):
                        results, tokens = correct_sqls(
                            data,
                            st.session_state.db_schema,
                            st.session_state.vector_store
                        )
                        
                        # Estimate token usage for batch processing
                        total_tokens = 0
                        for item in data:
                            # Simple estimation - replace with actual token counting
                            total_tokens += tokens
                        
                        update_token_usage('correct', total_tokens, f"Batch correcting {len(data)} queries")
                        
                        # Display results in an expandable section
                        for i, result in enumerate(results, 1):
                            with st.expander(f"Query {i}"):
                                st.text("Original SQL:")
                                st.code(result['IncorrectQuery'], language="sql")
                                st.text("Corrected SQL:")
                                st.code(result['CorrectQuery'], language="sql")
                        
                        # Offer download of results
                        st.download_button(
                            "Download Results",
                            json.dumps(results, indent=2),
                            "corrected_sql_results.json",
                            "application/json"
                        )

    # Add a dedicated Token Usage tab to the main content area
    with st.expander("Detailed Token Usage Statistics"):
        st.subheader("Token Usage Overview")
        
        # Create metrics in columns
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Tokens", st.session_state.token_usage['total'])
        with col2:
            st.metric("Generate SQL Tokens", st.session_state.token_usage['generate'])
        with col3:
            st.metric("Correct SQL Tokens", st.session_state.token_usage['correct'])
        
        # Display usage history as a table
        if st.session_state.token_usage['history']:
            st.subheader("Usage History")
            history_data = [{
                'Time': entry['timestamp'],
                'Operation': entry['operation'],
                'Tokens': entry['tokens'],
                'Query': entry['query']
            } for entry in st.session_state.token_usage['history']]
            
            st.dataframe(history_data, use_container_width=True)
            
            # Option to download token usage data
            st.download_button(
                "Download Token Usage Data",
                json.dumps(st.session_state.token_usage, indent=2),
                "token_usage_report.json",
                "application/json"
            )

if __name__ == "__main__":
    main()