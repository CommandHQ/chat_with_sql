import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_core.tools import tool
from typing import Annotated, Literal, List, Dict, Any, Optional, TypedDict
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import AnyMessage, add_messages
import urllib.parse
import re
from sqlalchemy.exc import OperationalError

def create_llm(model_name="gemini-1.5-flash", temperature=0, api_key=None):
    """Create LLM with proper error handling."""
    if not api_key:
        raise ValueError("API key is required for ChatGoogleGenerativeAI")
    
    return ChatGoogleGenerativeAI(
        model=model_name,
        temperature=temperature,
        api_key=api_key,
        convert_system_message_to_human=True,
    )

def create_db_connection(user, password, host, database, port=None):
    """Create MySQL database connection with improved error handling and validation."""
    try:
        encoded_password = urllib.parse.quote_plus(password)
        
        # If a port is explicitly provided, use it
        if port:
            connection_string = f"mysql+pymysql://{user}:{encoded_password}@{host}:{port}/{database}"
        # Otherwise check if host contains port information
        elif ":" in host:
            host_parts = host.split(":")
            host = host_parts[0]
            port = host_parts[1]
            connection_string = f"mysql+pymysql://{user}:{encoded_password}@{host}:{port}/{database}"
        else:
            # Default MySQL connection without port
            connection_string = f"mysql+pymysql://{user}:{encoded_password}@{host}/{database}"
        
        connection_string += "?charset=utf8mb4&connect_timeout=10"
        db = SQLDatabase.from_uri(connection_string)
        
        # Basic connection test
        db.get_usable_table_names()
        return db
    except OperationalError as e:
        error_msg = str(e)
        if "Access denied" in error_msg:
            raise ConnectionError("MySQL authentication failed.")
        elif "Unknown database" in error_msg:
            raise ConnectionError(f"Database '{database}' does not exist.")
        elif "Can't connect" in error_msg:
            raise ConnectionError(f"Could not connect to MySQL server at {host}")
        else:
            raise ConnectionError(f"Failed to connect to MySQL database: {error_msg}")
    except Exception as e:
        raise ConnectionError(f"Unexpected error connecting to database: {str(e)}")

class AgentState(TypedDict):
    """State for the SQL agent."""
    messages: Annotated[List[AnyMessage], add_messages]
    db_schema: Optional[str]
    current_step: str
    error: Optional[str]

@tool
def execute_sql_query(query: str) -> str:
    """
    Execute the SQL query against the MySQL database and return the result.
    If the query is invalid or returns no result, an error message will be returned.
    """
    try:
        if not query.strip().upper().startswith("SELECT"):
            return "Error: Only SELECT queries are allowed for data exploration."
            
        result = db.run_no_throw(query)
        if not result or result.strip() == "":
            return "The query executed successfully but returned no results."
        return result
    except Exception as e:
        return f"Error executing query: {str(e)}."

SYSTEM_PROMPT = """You are a MySQL database expert assistant. You help users explore and understand their data by writing SQL queries and explaining the results in plain language.

Available database tables: {tables}

Schema information:
{schema}

When responding:
1. Think carefully about what the user is asking
2. Generate an appropriate MySQL query
3. Execute the query and interpret the results
4. Respond in a natural, conversational way
5. Include relevant insights about the data when possible

Guidelines:
- Write queries that are optimized for MySQL syntax
- Limit results to 5 rows unless specified otherwise
- Only select columns relevant to the question
- Never write data modification queries (INSERT, UPDATE, DELETE)
- Use backticks (`) for table/column names when they contain special characters
- For dates, use MySQL functions like DATE_FORMAT() when needed
"""

def initialize_state() -> AgentState:
    """Initialize the agent state."""
    return {
        "messages": [],
        "db_schema": None,
        "current_step": "initialize",
        "error": None
    }

def fetch_schema(state: AgentState) -> AgentState:
    """Fetch database schema information with direct SQL queries."""
    new_state = state.copy()
    
    try:
        tables = db.get_usable_table_names()
        if not tables:
            raise ValueError("No tables found in the database.")
        
        schema_info = ""
        
        for table in tables:
            try:
                describe_query = f"DESCRIBE `{table}`;"
                table_structure = db.run_no_throw(describe_query)
                
                sample_query = f"SELECT * FROM `{table}` LIMIT 3;"
                sample_data = db.run_no_throw(sample_query)
                
                schema_info += f"\n--- Table: {table} ---\n"
                schema_info += f"Structure:\n{table_structure}\n"
                schema_info += f"Sample data:\n{sample_data}\n"
                
            except Exception:
                schema_info += f"\n--- Table: {table} ---\nError retrieving schema\n"
        
        new_state["db_schema"] = f"Tables: {', '.join(tables)}\n\nDetailed Schema:\n{schema_info}"

        new_state["current_step"] = "schema_fetched"
        return new_state
    except Exception as e:
        new_state["error"] = f"Failed to fetch schema: {str(e)}"
        return new_state

def process_user_input(state: AgentState) -> AgentState:
    """Process user input with improved error handling for missing schema."""
    new_state = state.copy()
    
    user_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
    if not user_messages:
        new_state["error"] = "No user input found."
        return new_state
    
    user_input = user_messages[-1].content
    
    if not new_state["db_schema"] or len(new_state["db_schema"]) < 20:
        new_state["error"] = "Database schema information is missing or incomplete."
        return new_state
    
    try:
        system_prompt = SYSTEM_PROMPT.format(
            tables=db.get_usable_table_names(),
            schema=new_state["db_schema"]
        )
        
        message_history = []
        all_messages = state["messages"][:-1]
        context_messages = []
        
        for msg in all_messages:
            if isinstance(msg, HumanMessage) or isinstance(msg, AIMessage):
                context_messages.append(msg)
        
        context_messages = context_messages[-6:]
        message_history.extend(context_messages)
        
        messages = [
            SystemMessage(content=system_prompt),
            *message_history,
            HumanMessage(content=user_input)
        ]
        
        response = llm.invoke(messages)
        sql_query = extract_sql_query(response.content)
        
        if sql_query:
            query_result = execute_sql_query.invoke(sql_query)
            
            final_response_messages = [
                SystemMessage(content=system_prompt),
                SystemMessage(content=f"You previously suggested this SQL query: {sql_query}"),
                SystemMessage(content=f"The query returned these results: {query_result}"),
                HumanMessage(content=f"Original question: {user_input}\n\nProvide a helpful, natural language response that answers the question based on the query results. Include relevant insights and context. Do not mention that you ran a SQL query unless specifically asked about the query.")
            ]
            
            final_response = llm.invoke(final_response_messages)
            new_state["messages"].append(AIMessage(content=final_response.content))
        else:
            new_state["messages"].append(AIMessage(content=response.content))
        
        new_state["current_step"] = "completed"
        return new_state
    except Exception as e:
        new_state["error"] = f"Error processing input: {str(e)}"
        return new_state

def handle_error(state: AgentState) -> AgentState:
    """Handle errors in the workflow."""
    new_state = state.copy()
    error_message = f"I apologize, but I encountered an error: {new_state['error']}. Please try rephrasing your question or ask something else."
    new_state["messages"].append(AIMessage(content=error_message))
    new_state["error"] = None
    new_state["current_step"] = "error_handled"
    return new_state

def extract_sql_query(text: str) -> Optional[str]:
    """Extract SQL query from text."""
    code_block_pattern = r"```sql\s+(.*?)\s+```"
    matches = re.findall(code_block_pattern, text, re.DOTALL | re.IGNORECASE)
    
    if matches:
        return matches[0].strip()
    
    select_pattern = r"SELECT\s+.*?(?:;|$)"
    matches = re.findall(select_pattern, text, re.DOTALL | re.IGNORECASE)
    
    if matches:
        query = matches[0].strip()
        if not query.endswith(";"):
            query += ";"
        return query
    
    return None

def should_route(state: AgentState) -> Literal["process_input", "handle_error", END]:
    """Determine the next route based on state."""
    if state["error"]:
        return "handle_error"
    
    if state["current_step"] == "schema_fetched":
        return "process_input"
    
    if state["current_step"] in ["completed", "error_handled"]:
        return END
    
    return "process_input"

def create_sql_agent(api_key, db_user, db_password, db_host, db_name, db_port=None):
    """Create the SQL agent with the simplified workflow."""
    global llm, db
    
    llm = create_llm(api_key=api_key)
    db = create_db_connection(db_user, db_password, db_host, db_name, db_port)
    
    workflow = StateGraph(AgentState)
    
    workflow.add_node("initialize", lambda _: initialize_state())
    workflow.add_node("fetch_schema", fetch_schema)
    workflow.add_node("process_input", process_user_input)
    workflow.add_node("handle_error", handle_error)
    
    workflow.add_edge(START, "initialize")
    workflow.add_edge("initialize", "fetch_schema")
    
    workflow.add_conditional_edges(
        "fetch_schema",
        should_route,
        {
            "process_input": "process_input",
            "handle_error": "handle_error",
            END: END
        }
    )
    
    workflow.add_conditional_edges(
        "process_input",
        should_route,
        {
            "process_input": "process_input",
            "handle_error": "handle_error",
            END: END
        }
    )
    
    workflow.add_conditional_edges(
        "handle_error",
        should_route,
        {
            "process_input": "process_input",
            "handle_error": "handle_error",
            END: END
        }
    )
    
    app = workflow.compile()
    
    def run_sql_agent(user_input):
        """Run the SQL agent on user input."""
        input_state = {"messages": [HumanMessage(content=user_input)]}
        result = app.invoke(input_state)
        return result["messages"][-1].content
    
    # Update flag to indicate agent is initialized
    return run_sql_agent

if __name__ == "__main__":
    try:
        load_dotenv()
        
        API_KEY = os.getenv("GOOGLE_API_KEY")
        DB_USER = os.getenv("DB_USER")
        DB_PASSWORD = os.getenv("DB_PASSWORD")
        DB_HOST = os.getenv("DB_HOST")
        DB_NAME = os.getenv("DB_NAME")
        DB_PORT = os.getenv("DB_PORT")
        
        if not all([API_KEY, DB_USER, DB_PASSWORD, DB_HOST, DB_NAME]):
            raise ValueError("One or more required environment variables are missing")
        
        sql_agent = create_sql_agent(
            api_key=API_KEY,
            db_user=DB_USER,
            db_password=DB_PASSWORD,
            db_host=DB_HOST,
            db_name=DB_NAME,
            db_port=DB_PORT if DB_PORT else None
        )
        
        while True:
            query = input("\nAsk a question (or type 'exit' to quit): ").strip()
            if query.lower() in ['exit', 'quit']:
                break
            try:
                response = sql_agent(query)
                print("---------------------------------------------------------")
                print("\nResponse:", response)
                print("----------------------------------------------------------")
            except Exception:
                print("\nI encountered an error processing your request. Please try again.")
        
    except Exception as e:
        print(f"Failed to initialize the agent: {str(e)}")