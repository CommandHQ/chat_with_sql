import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import SQLDatabase
from langchain_core.tools import tool
from typing import Annotated, Literal, List, Dict, Any, Optional, TypedDict
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import AnyMessage, add_messages
import urllib.parse
import re
from sqlalchemy.exc import OperationalError
import json

def create_llm(model_name="gemini-1.5-flash", temperature=0, api_key=None):
    """Create LLM with proper error handling."""
    if not api_key:
        raise ValueError("API key is required for ChatGoogleGenerativeAI")
    
    return ChatGoogleGenerativeAI(
        model=model_name,
        temperature=temperature,
        api_key=api_key,
        # Don't convert system messages automatically - we'll handle them explicitly
        convert_system_message_to_human=False,
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
    """State for the SQL agent with in-memory conversation retention."""
    messages: Annotated[List[AnyMessage], add_messages]
    db_schema: Optional[str]
    current_step: str
    error: Optional[str]
    conversation_summary: Optional[str]  # Periodically updated summary
    summarization_index: int  # Index to track where summarization was last done
    important_insights: List[str]  # Track key database insights for reference
    tables_discussed: List[str]  # Keep track of which tables have been discussed
    relevant_context: List[Dict]  # Store key query-result pairs for reference

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

SYSTEM_PROMPT = """You are a MySQL database expert assistant with conversation memory. You help users explore and understand their data by writing SQL queries and explaining the results in plain language.

Available database tables: {tables}

Schema information:
{schema}

Memory Context:
{memory_context}

When responding:
1. Think carefully about what the user is asking
2. Generate an appropriate MySQL query
3. Execute the query and interpret the results
4. Respond in a natural, conversational way
5. Include relevant insights about the data when possible
6. Reference previous questions and findings when relevant

Guidelines:
- Write queries that are optimized for MySQL syntax
- Limit results to 5 rows unless specified otherwise
- Only select columns relevant to the question
- Never write data modification queries (INSERT, UPDATE, DELETE)
- Use backticks (`) for table/column names when they contain special characters
- For dates, use MySQL functions like DATE_FORMAT() when needed
"""

def initialize_state() -> AgentState:
    """Initialize the agent state with enhanced memory components."""
    return {
        "messages": [],
        "db_schema": None,
        "current_step": "initialize",
        "error": None,
        "conversation_summary": None,
        "summarization_index": 0,
        "important_insights": [],
        "tables_discussed": [],
        "relevant_context": []  # New field to store query-result pairs
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

def extract_tables_from_query(query: str) -> List[str]:
    """Extract table names from a SQL query."""
    if not query:
        return []
    
    # Simple regex to find table names in FROM and JOIN clauses
    from_pattern = r"FROM\s+`?(\w+)`?"
    join_pattern = r"JOIN\s+`?(\w+)`?"
    
    tables = []
    
    from_matches = re.findall(from_pattern, query, re.IGNORECASE)
    if from_matches:
        tables.extend(from_matches)
    
    join_matches = re.findall(join_pattern, query, re.IGNORECASE)
    if join_matches:
        tables.extend(join_matches)
    
    return list(set(tables))  # Remove duplicates

def extract_insight(query_result: str, user_question: str) -> Optional[str]:
    """Try to extract a key insight from the query result based on the user's question."""
    if not query_result or "error" in query_result.lower():
        return None
    
    # Use a direct approach instead of an LLM call to avoid API issues
    try:
        # Check if we have results in a tabular format
        if '|' in query_result and '\n' in query_result:
            lines = query_result.strip().split('\n')
            if len(lines) >= 3:  # Header, separator, and at least one row
                return f"Found {len(lines)-2} results relevant to '{user_question}'"
        return None
    except Exception:
        return None

def generate_memory_context(state: AgentState) -> str:
    """Generate memory context from conversation history and tracked insights."""
    memory_parts = []
    
    # Include conversation summary if available
    if state["conversation_summary"]:
        memory_parts.append(f"Conversation summary:\n{state['conversation_summary']}")
    
    # Include important database insights
    if state["important_insights"]:
        memory_parts.append("Key insights from previous queries:")
        for idx, insight in enumerate(state["important_insights"][-5:], 1):  # Last 5 insights
            memory_parts.append(f"{idx}. {insight}")
    
    # Include information about previously discussed tables
    if state["tables_discussed"]:
        tables_str = ", ".join(state["tables_discussed"])
        memory_parts.append(f"Tables previously discussed: {tables_str}")
    
    # Include recent query contexts
    if state["relevant_context"]:
        memory_parts.append("Recent query results:")
        for idx, ctx in enumerate(state["relevant_context"][-3:], 1):  # Last 3 query contexts
            memory_parts.append(f"{idx}. Query about '{ctx['question']}' returned {ctx['result_summary']}")
    
    if not memory_parts:
        return "No previous conversation context."
    
    return "\n\n".join(memory_parts)

def check_if_summarization_needed(state: AgentState) -> bool:
    """Check if conversation needs summarization based on message count."""
    message_count = len([m for m in state["messages"] if isinstance(m, (HumanMessage, AIMessage))])
    # Need at least 8 messages (4 exchanges) and at least 4 new messages since last summarization
    return message_count >= 8 and (message_count - state["summarization_index"]) >= 4

def summarize_conversation(state: AgentState) -> AgentState:
    """Create a summary of the conversation without using LLM to avoid API issues."""
    new_state = state.copy()
    
    if not check_if_summarization_needed(state):
        return new_state
    
    try:
        # Get relevant context messages
        context_messages = [m for m in state["messages"] if isinstance(m, (HumanMessage, AIMessage))][-10:]
        
        # Create a simple summary without using LLM
        topics = []
        for msg in context_messages:
            if isinstance(msg, HumanMessage):
                content = msg.content.lower()
                if "show" in content or "list" in content or "what" in content:
                    topics.append("Data retrieval")
                elif "how many" in content or "count" in content:
                    topics.append("Count analysis")
                elif "compare" in content or "difference" in content:
                    topics.append("Comparison")
                elif "trend" in content or "over time" in content:
                    topics.append("Trend analysis")
                elif "group" in content or "categorize" in content:
                    topics.append("Grouping data")
        
        # Remove duplicates and create summary
        unique_topics = list(set(topics))
        if unique_topics:
            summary = f"This conversation covers {', '.join(unique_topics)}."
            # Add info about tables if available
            if new_state["tables_discussed"]:
                summary += f" Tables discussed: {', '.join(new_state['tables_discussed'])}."
            
            new_state["conversation_summary"] = summary
            new_state["summarization_index"] = len([m for m in new_state["messages"] if isinstance(m, (HumanMessage, AIMessage))])
    except Exception as e:
        print(f"Error in simple summarization: {str(e)}")
        # If summarization fails, we can still continue without it
    
    return new_state

def process_user_input(state: AgentState) -> AgentState:
    """Process user input with memory-enhanced context."""
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
        # Generate memory context
        memory_context = generate_memory_context(new_state)
        
        system_prompt = SYSTEM_PROMPT.format(
            tables=db.get_usable_table_names(),
            schema=new_state["db_schema"],
            memory_context=memory_context
        )
        
        message_history = []
        all_messages = state["messages"][:-1]
        context_messages = []
        
        for msg in all_messages:
            if isinstance(msg, HumanMessage) or isinstance(msg, AIMessage):
                context_messages.append(msg)
        
        # Include more context messages with memory
        context_messages = context_messages[-12:]  # Increased from 6 to 12
        message_history.extend(context_messages)
        
        # Convert SystemMessage to HumanMessage for Gemini compatibility
        messages = [
            HumanMessage(content=f"[SYSTEM INFORMATION]\n{system_prompt}"),
            *message_history,
            HumanMessage(content=user_input)
        ]
        
        response = llm.invoke(messages)
        sql_query = extract_sql_query(response.content)
        
        if sql_query:
            query_result = execute_sql_query.invoke(sql_query)
            
            # Update memory with tables discussed
            tables_used = extract_tables_from_query(sql_query)
            for table in tables_used:
                if table not in new_state["tables_discussed"]:
                    new_state["tables_discussed"].append(table)
            
            # Create a result summary
            result_lines = query_result.split('\n')
            result_summary = f"{len(result_lines)-2 if len(result_lines) > 2 else 0} rows of data"
            
            # Store query context for memory
            new_state["relevant_context"].append({
                "question": user_input,
                "query": sql_query,
                "result_summary": result_summary
            })
            
            # Keep only the last 5 contexts
            if len(new_state["relevant_context"]) > 5:
                new_state["relevant_context"] = new_state["relevant_context"][-5:]
            
            # Try to extract an insight (now using our non-LLM version)
            insight = extract_insight(query_result, user_input)
            if insight and insight not in new_state["important_insights"]:
                new_state["important_insights"].append(insight)
                # Keep only the last 10 insights
                if len(new_state["important_insights"]) > 10:
                    new_state["important_insights"] = new_state["important_insights"][-10:]
            
            # Convert SystemMessage to HumanMessage for final response
            final_response_messages = [
                HumanMessage(content=f"[SYSTEM INFORMATION]\nYou previously suggested this SQL query: {sql_query}\n\nThe query returned these results: {query_result}\n\nOriginal question: {user_input}\n\nProvide a helpful, natural language response that answers the question based on the query results. Include relevant insights and context. Do not mention that you ran a SQL query unless specifically asked about the query. Refer to previous conversations and insights when relevant.")
            ]
            
            final_response = llm.invoke(final_response_messages)
            new_state["messages"].append(AIMessage(content=final_response.content))
        else:
            new_state["messages"].append(AIMessage(content=response.content))
        
        # Update conversation summary using our simplified approach
        new_state = summarize_conversation(new_state)
        
        new_state["current_step"] = "completed"
        return new_state
    except Exception as e:
        new_state["error"] = f"Error processing input: {str(e)}"
        return new_state

def handle_error(state: AgentState) -> AgentState:
    """Handle errors in the workflow while preserving memory."""
    new_state = state.copy()
    error_message = f"I apologize, but I encountered an error: {new_state['error']}. Please try rephrasing your question or ask something else."
    new_state["messages"].append(AIMessage(content=error_message))
    new_state["error"] = None
    new_state["current_step"] = "error_handled"
    return new_state

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
    """Create the SQL agent with enhanced in-memory conversation retention."""
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
    
    # Create compiled app
    app = workflow.compile()
    
    # Keep track of the graph state between calls
    graph_state = None
    
    def run_sql_agent(user_input):
        """Run the SQL agent on user input with persistent memory."""
        nonlocal graph_state
        
        if graph_state is None:
            # First call - initialize from scratch
            input_state = {"messages": [HumanMessage(content=user_input)]}
            result = app.invoke(input_state)
        else:
            # Subsequent calls - continue from previous state
            current_messages = graph_state["messages"].copy()
            current_messages.append(HumanMessage(content=user_input))
            
            input_state = graph_state.copy()
            input_state["messages"] = current_messages
            result = app.invoke(input_state)
        
        # Update stored state for next call
        graph_state = result
        
        # Return just the last message
        return result["messages"][-1].content
    
    # Return the function to interact with the agent
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
        
        print("\nSQL Agent initialized with in-memory conversation retention.")
        print("Type your questions about the database or type 'exit' to quit.")
        
        while True:
            query = input("\nAsk a question (or type 'exit' to quit): ").strip()
            if query.lower() in ['exit', 'quit']:
                break
            try:
                response = sql_agent(query)
                print("---------------------------------------------------------")
                print("\nResponse:", response)
                print("----------------------------------------------------------")
            except Exception as e:
                print(f"\nI encountered an error processing your request: {str(e)}")
        
    except Exception as e:
        print(f"Failed to initialize the agent: {str(e)}")