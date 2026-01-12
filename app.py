import logging
import streamlit as st
from config import (
    GENERATIVE_MODEL_NAME,
    EMBEDDING_MODEL_NAME,
    MATCH_THRESHOLD,
    MATCH_COUNT
)
from services import (
    OllamaEmbeddingModel,
    SupabaseVectorStore,
    GeminiGenerativeModel
)
from rag import RagPipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

st.set_page_config(
    page_title="🏙️ Tokyo Hotel & Airbnb Chatbot",
    page_icon="🏨"
)

# --- Client Initialization ---

@st.cache_resource
def init_rag_pipeline():
    """Initializes and returns the RAG pipeline."""
    logging.info("Initializing RAG pipeline...")
    embedding_model = OllamaEmbeddingModel(EMBEDDING_MODEL_NAME)
    vector_store = SupabaseVectorStore()
    generative_model = GeminiGenerativeModel(GENERATIVE_MODEL_NAME)
    
    pipeline = RagPipeline(
        embedding_model=embedding_model,
        vector_store=vector_store,
        generative_model=generative_model
    )
    logging.info("RAG pipeline initialized successfully.")
    return pipeline

# --- Streamlit UI ---

st.sidebar.markdown("[Read the Code on GitHub](https://github.com/jansenzjh/hotel-chatbot)")

st.title("🏙️ Tokyo Hotel & Airbnb Chatbot")
st.caption(f"Powered by Supabase, {EMBEDDING_MODEL_NAME} model, and {GENERATIVE_MODEL_NAME} model")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    logging.info("Initialized new chat session.")

# Initialize RAG pipeline
rag_pipeline = init_rag_pipeline()

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask me about hotels in Tokyo (eg. 'hotel in shinjuku provide dryer and washer')..."):
    logging.info(f"User query: {prompt}")
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        # Get the last 2 messages for history
        chat_history = st.session_state.messages[-2:]
        
        response_stream = st.write_stream(
            rag_pipeline.get_rag_response(
                user_query=prompt,
                chat_history=chat_history,
                match_threshold=MATCH_THRESHOLD,
                match_count=MATCH_COUNT
            )
        )
    logging.info(f"Assistant response: {response_stream}")
        
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response_stream})
