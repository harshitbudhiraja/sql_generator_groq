# Import necessary libraries
import json
import requests
import time

# Global variable to keep track of the total number of tokens
total_tokens = 0

# Function to load input file
def load_input_file(file_path):
    """
    Load input file which is a list of dictionaries.
    
    :param file_path: Path to the input file
    :return: List of dictionaries
    """
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

# Function to generate SQL statements
def generate_sqls(data):
    """
    Generate SQL statements from the NL queries.
    
    :param data: List of NL queries
    :return: List of SQL statements
    """
    sql_statements = []
    # TODO: Implement logic to generate SQL statements
    return sql_statements

# Function to correct SQL statements
def correct_sqls(sql_statements):
    """
    Correct SQL statements if necessary.
    
    :param sql_statements: List of Dict with incorrect SQL statements and NL query
    :return: List of corrected SQL statements
    """
    corrected_sqls = []
    # TODO: Implement logic to correct SQL statements
    return corrected_sqls

# Function to call the Groq API
def call_groq_api(api_key, model, messages, temperature=0.0, max_tokens=1000, n=1):
    """
    NOTE: DO NOT CHANGE/REMOVE THE TOKEN COUNT CALCULATION 
    Call the Groq API to get a response from the language model.
    :param api_key: API key for authentication
    :param model: Model name to use
    :param messages: List of message dictionaries
    :param temperature: Temperature for the model
    :param max_tokens: Maximum number of tokens to generate (these are max new tokens)
    :param n: Number of responses to generate
    :return: Response from the API
    """
    global total_tokens
    url = "https://api.groq.com/openai/v1/chat/completions"
    
    api_key = "<your_api_key>"
    model = "llama-3.3-70b-versatile"
    messages = [
        {
            "role": "user",
            "content": "Explain the importance of fast language models"
        }
    ]
    
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

    response = requests.post(url, headers=headers, json=data)
    response_json = response.json()


    # Update the global token count
    total_tokens += response_json.get('usage', {}).get('completion_tokens', 0)

    # You can get the completion from response_json['choices'][0]['message']['content']
    return response_json, total_tokens

# Main function
def main():
    # TODO: Specify the path to your input file
    input_file_path_1 = 'path/to/your/input_file_for_sql_generation.json'
    input_file_path_2 = 'path/to/your/input_file_for_sql_correction.json'
    
    # Load data from input file
    data_1 = load_input_file(input_file_path_1)
    data_2 = load_input_file(input_file_path_2)
    
    start = time.time()
    # Generate SQL statements
    sql_statements = generate_sqls(data_1)
    generate_sqls_time = time.time() - start
    
    start = time.time()
    # Correct SQL statements
    corrected_sqls = correct_sqls(data_2)
    correct_sqls_time = time.time() - start
    
    assert len(data_2) == len(corrected_sqls) # If no answer, leave blank
    assert len(data_1) == len(sql_statements) # If no answer, leave blank
    
    # TODO: Process the outputs
    
    # Get the outputs as a list of dicts with keys 'IncorrectQuery' and 'CorrectQuery'
    with open('output_sql_correction_task.json', 'w') as f:
        json.dump(corrected_sqls, f)    
    
    # Get the outputs as a list of dicts with keys 'NL' and 'Query'
    with open('output_sql_generation_task.json', 'w') as f:
        json.dump(sql_statements, f)
    
    return generate_sqls_time, correct_sqls_time



if __name__ == "__main__":
    generate_sqls_time, correct_sqls_time = main()
    print(f"Time taken to generate SQLs: {generate_sqls_time} seconds")
    print(f"Time taken to correct SQLs: {correct_sqls_time} seconds")
    print(f"Total tokens: {total_tokens}")

    
