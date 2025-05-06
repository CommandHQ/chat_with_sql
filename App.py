import streamlit as st
import os
import time
from langchain_core.messages import HumanMessage, AIMessage
from create_agent import create_sql_agent
from dotenv import load_dotenv

# Load environment variables once at startup
load_dotenv()


def initialize_session_state():
    """Initialize all session state variables"""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "agent_initialized" not in st.session_state:
        st.session_state.agent_initialized = False
    if "agent" not in st.session_state:
        st.session_state.agent = None
    if "connection_status" not in st.session_state:
        st.session_state.connection_status = {"status": None, "message": None}
    if "db_config" not in st.session_state:
        st.session_state.db_config = {
            "user": os.getenv("DB_USER", ""),
            "password": os.getenv("DB_PASSWORD", ""),
            "host": os.getenv("DB_HOST", ""),
            "name": os.getenv("DB_NAME", ""),
            "port": os.getenv("DB_PORT", "28469"),
            "api_key": os.getenv("GOOGLE_API_KEY", "")
        }
    if "awaiting_response" not in st.session_state:
        st.session_state.awaiting_response = False


def connect_to_database():
    """Attempt to connect to the database with the current configuration"""
    try:
        # Validate that required fields are provided
        required_fields = ["user", "password", "host", "name", "api_key"]
        missing_fields = [field for field in required_fields 
                         if not st.session_state.db_config.get(field)]
        
        if missing_fields:
            st.session_state.connection_status = {
                "status": "error",
                "message": f"Missing required fields: {', '.join(missing_fields)}"
            }
            return
            
        with st.spinner("Connecting to database..."):
            # Create agent with stored credentials
            st.session_state.agent = create_sql_agent(
                api_key=st.session_state.db_config["api_key"],
                db_user=st.session_state.db_config["user"],
                db_password=st.session_state.db_config["password"],
                db_host=st.session_state.db_config["host"],
                db_name=st.session_state.db_config["name"],
                db_port=int(st.session_state.db_config["port"]) if st.session_state.db_config["port"] else None
            )
            
            st.session_state.agent_initialized = True
            st.session_state.connection_status = {
                "status": "success",
                "message": "Connected successfully!"
            }
            time.sleep(1)  # Give visual feedback
            
    except Exception as e:
        st.session_state.agent_initialized = False
        error_message = str(e)
        # Handle common errors with more user-friendly messages
        if "could not connect" in error_message.lower():
            error_message = "Could not connect to database. Please check your credentials and network connection."
        elif "authentication failed" in error_message.lower():
            error_message = "Authentication failed. Please check your username and password."
        
        st.session_state.connection_status = {
            "status": "error",
            "message": f"Connection failed: {error_message}"
        }


def render_sidebar():
    """Render the sidebar for database configuration"""
    with st.sidebar:
        st.title("🛢️ Database Configuration")
        
        with st.expander("Connection Settings", expanded=not st.session_state.agent_initialized):
            # Use columns for a more compact layout
            col1, col2 = st.columns(2)
            
            with col1:
                st.session_state.db_config["host"] = st.text_input(
                    "Host", 
                    value=st.session_state.db_config["host"],
                    placeholder="Enter host address"
                )
                
                st.session_state.db_config["port"] = st.text_input(
                    "Port", 
                    value=st.session_state.db_config["port"],
                    placeholder="Enter port number"
                )
                
                st.session_state.db_config["name"] = st.text_input(
                    "Database", 
                    value=st.session_state.db_config["name"],
                    placeholder="Enter database name"
                )
                
            with col2:
                st.session_state.db_config["user"] = st.text_input(
                    "Username", 
                    value=st.session_state.db_config["user"],
                    placeholder="Enter username"
                )
                
                # Only show actual value if empty, otherwise show placeholder
                password_display = "" if st.session_state.db_config["password"] else st.session_state.db_config["password"]
                entered_pw = st.text_input(
                    "Password", 
                    value=password_display,
                    type="password",
                    placeholder="••••••••" if st.session_state.db_config["password"] else "Enter password"
                )
                # Only update if user changed the field
                if entered_pw != password_display:
                    st.session_state.db_config["password"] = entered_pw
                
                # Same approach for API key
                api_key_display = "" if st.session_state.db_config["api_key"] else st.session_state.db_config["api_key"]
                entered_api_key = st.text_input(
                    "API Key", 
                    value=api_key_display,
                    type="password",
                    placeholder="••••••••" if st.session_state.db_config["api_key"] else "Enter Google API key"
                )
                # Only update if user changed the field
                if entered_api_key != api_key_display:
                    st.session_state.db_config["api_key"] = entered_api_key
            
            connect_btn = st.button(
                "🔌 Connect to Database", 
                use_container_width=True,
                type="primary"
            )
            
            if connect_btn:
                connect_to_database()
        
        # Display connection status
        if st.session_state.connection_status["status"] == "success":
            st.success(st.session_state.connection_status["message"])
        elif st.session_state.connection_status["status"] == "error":
            st.error(st.session_state.connection_status["message"])
        
        # Show additional information when connected
        if st.session_state.agent_initialized:
            st.divider()
            
            st.markdown("### Database Info")
            st.markdown(f"**Host:** {st.session_state.db_config['host']}")
            st.markdown(f"**Database:** {st.session_state.db_config['name']}")
            st.markdown(f"**User:** {st.session_state.db_config['user']}")
            st.markdown("**Password:** •••••••• (secured)")
            st.markdown("**API Key:** •••••••• (secured)")
            
            if st.button("Disconnect", use_container_width=True):
                st.session_state.agent = None
                st.session_state.agent_initialized = False
                st.session_state.connection_status = {"status": None, "message": None}
                st.rerun()


def render_chat_interface():
    """Render the main chat interface"""
    # Container for the header with custom styling
    header = st.container()
    with header:
        col1, col2 = st.columns([3, 1])
        col1.title("💬 SQL Chat Assistant")
        
        if st.session_state.agent_initialized:
            col2.markdown(
                """<div style='background-color: #28a745; color: white; padding: 10px; 
                border-radius: 5px; text-align: center; margin-top: 10px;'>
                ✅ Connected</div>""", 
                unsafe_allow_html=True
            )
        else:
            col2.markdown(
                """<div style='background-color: #dc3545; color: white; padding: 10px; 
                border-radius: 5px; text-align: center; margin-top: 10px;'>
                ❌ Disconnected</div>""", 
                unsafe_allow_html=True
            )
    
    st.divider()
    
    if not st.session_state.agent_initialized:
        st.info("Please connect to a database from the sidebar to start chatting.")
        
        # Example queries section
        with st.expander("Example queries you can ask once connected"):
            st.markdown("""
            - "Show me the top 5 customers by revenue"
            - "What was the sales trend over the last 6 months?"
            - "How many orders were placed in the Northwest region?"
            - "Which products have the highest profit margin?"
            """)
        return

    # Create a container for the chat history
    chat_container = st.container()
    
    # Chat input - positioned before displaying history but will appear at the bottom
    user_input = st.chat_input("Ask a question about your database...", key="user_input")
    
    # Display chat history in the container
    with chat_container:
        for message in st.session_state.chat_history:
            if isinstance(message, HumanMessage):
                with st.chat_message("user"):
                    st.markdown(message.content)
            elif isinstance(message, AIMessage):
                with st.chat_message("assistant"):
                    st.markdown(message.content)
    
    # Process new input if any
    if user_input:
        # Add user message to chat history
        st.session_state.chat_history.append(HumanMessage(content=user_input))
        
        # Force a rerun to update the UI with the user message
        if not st.session_state.awaiting_response:
            st.session_state.awaiting_response = True
            st.session_state.current_question = user_input
            st.rerun()
            
    # Process the pending response if needed
    if st.session_state.awaiting_response:
        user_question = st.session_state.current_question
        
        try:
            with st.spinner("Analyzing your database..."):
                response = st.session_state.agent(user_question)
                st.session_state.chat_history.append(AIMessage(content=response))
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            st.session_state.chat_history.append(AIMessage(content=error_msg))
        
        # Reset the flags
        st.session_state.awaiting_response = False
        if "current_question" in st.session_state:
            del st.session_state.current_question
        
        # Force a rerun to update the UI with the complete chat history
        st.rerun()


def main():
    """Main app entry point"""
    # Set page configuration
    st.set_page_config(
        page_title="SQL Chat Assistant",
        page_icon="🛢️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Apply custom CSS
    st.markdown("""
    <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .stButton button {
            border-radius: 20px;
        }
        .stTextInput input {
            border-radius: 5px;
        }
        .stExpander {
            border-radius: 10px;
        }
        .stSidebar .block-container {
            padding-top: 1rem;
        }
        div[data-testid="stSidebarNav"] {
            background-image: linear-gradient(#4e54c8, #8f94fb);
            color: white;
            padding: 10px;
            border-radius: 10px;
        }
        /* Add visual indication for password fields */
        [data-testid="stPasswordInput"] input {
            background-color: #f8f9fa;
            border: 1px solid #dee2e6;
        }
        /* Hide error messages related to rerun data */
        .st-emotion-cache-16txtl3:has(div:contains("RerunData")) {
            display: none;
        }
    </style>
    """, unsafe_allow_html=True)

    # Initialize session state
    initialize_session_state()
    
    # Render sidebar and main interface
    render_sidebar()
    render_chat_interface()


if __name__ == "__main__":
    main()