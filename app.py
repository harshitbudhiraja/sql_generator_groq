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
                        result = generate_sqls(
                            [{"NL": nl_query}],
                            st.session_state.db_schema,
                            st.session_state.vector_store
                        )
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
                        results = generate_sqls(
                            data,
                            st.session_state.db_schema,
                            st.session_state.vector_store
                        )
                        
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
                        result = correct_sqls(
                            [{"IncorrectQuery": incorrect_sql}],
                            st.session_state.db_schema,
                            st.session_state.vector_store
                        )
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
                        results = correct_sqls(
                            data,
                            st.session_state.db_schema,
                            st.session_state.vector_store
                        )
                        
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

if __name__ == "__main__":
    main()