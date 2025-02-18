import streamlit as st
import os
from datetime import datetime
import logging
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(
    filename='chatbot.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

# Load environment variables
load_dotenv()

# Configure LangChain
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

def initialize_session_state():
    """Initialize session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "conversation_memory" not in st.session_state:
        st.session_state.conversation_memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            k=5  # Keep last 5 exchanges
        )

def validate_input(input_text: str) -> bool:
    """Validate user input"""
    if not input_text or input_text.isspace():
        st.error("Please enter a question")
        return False
    if len(input_text) > 1000:
        st.error("Please keep your question under 1000 characters")
        return False
    return True

def log_conversation(question: str, response: str):
    """Log conversation to file"""
    try:
        logging.info(f"Q: {question}\nA: {response}\n")
    except Exception as e:
        logging.error(f"Logging failed: {str(e)}")

def setup_sidebar():
    """Setup sidebar with controls"""
    with st.sidebar:
        st.title("Chat Settings")
        temperature = st.slider(
            "Response Creativity",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            help="Higher values make responses more creative"
        )
        
        if st.button("Clear Conversation"):
            st.session_state.messages = []
            st.session_state.conversation_memory.clear()
            st.success("Conversation cleared!")
            
        return temperature

def get_llm_chain(temperature: float):
    """Setup LLM chain with prompt template"""
    # Create prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Please provide clear and concise responses to user queries."),
        ("user", "Question: {question}")
    ])
    
    # Initialize LLM
    llm = Ollama(model="llama2", temperature=temperature)
    
    # Setup output parser
    output_parser = StrOutputParser()
    
    # Create chain
    chain = prompt | llm | output_parser
    
    return chain

def process_response(chain, input_text: str) -> str:
    """Process user input and get response"""
    try:
        response = chain.invoke({"question": input_text})
        return response
    except Exception as e:
        error_msg = f"Error processing response: {str(e)}"
        logging.error(error_msg)
        st.error(error_msg)
        return None

def main():
    # Page configuration
    st.set_page_config(
        page_title="AI Chatbot",
        page_icon="🤖",
        layout="wide"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Main title
    st.title("🤖 AI Chatbot")
    st.markdown("Ask me anything! I'm here to help.")
    
    # Setup sidebar and get temperature
    temperature = setup_sidebar()
    
    # Setup LLM chain
    chain = get_llm_chain(temperature)
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Get user input
    if input_text := st.chat_input("Type your message here..."):
        if validate_input(input_text):
            # Add user message to chat
            st.session_state.messages.append({"role": "user", "content": input_text})
            with st.chat_message("user"):
                st.write(input_text)
            
            # Show thinking message
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    # Get response
                    response = process_response(chain, input_text)
                    
                    if response:
                        # Log conversation
                        log_conversation(input_text, response)
                        
                        # Add response to chat
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        st.write(response)

if __name__ == "__main__":
    main()