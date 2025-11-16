import streamlit as st
import os
import google.generativeai as genai
from supabase import create_client, Client
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# --- Configuration ---
load_dotenv()
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

# Local Embedding Model
# NEW: Updated to the Gemma model
EMBEDDING_MODEL_NAME = 'google/embeddinggemma-300m' 
# NEW: This must match the VECTOR() dimension in your new SQL file (768)
VECTOR_DIMENSION = 768 

# Gemini Generative Model
GENERATIVE_MODEL_NAME = 'gemini-2.5-flash-lite'

# RAG Parameters
MATCH_THRESHOLD = 0.5  # Similarity threshold
MATCH_COUNT = 5        # Number of documents to retrieve

# --- Client Initialization ---

@st.cache_resource
def init_supabase() -> Client:
    """Initializes and returns the Supabase client."""
    return create_client(SUPABASE_URL, SUPABASE_KEY)

@st.cache_resource
def load_embedding_model() -> SentenceTransformer:
    """
    Loads and caches the SentenceTransformer model.
    This is the "hosting" part - it only runs once!
    """
    print(f"Loading local embedding model: {EMBEDDING_MODEL_NAME}...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print("Embedding model loaded.")
    return model

@st.cache_resource
def init_generative_model() -> genai.GenerativeModel:
    """Initializes and returns the Gemini generative model."""
    genai.configure(api_key=GOOGLE_API_KEY)
    return genai.GenerativeModel(GENERATIVE_MODEL_NAME)

# --- Main RAG Function ---

def get_rag_response(user_query: str):
    """
    Performs the full RAG pipeline and streams the response.
    """
    embed_model = load_embedding_model()
    supabase = init_supabase()
    gen_model = init_generative_model()

    # 1. Embed the user's query
    yield "Embedding your query...\n"
    query_embedding = embed_model.encode(user_query).tolist()

    # 2. Search Supabase for similar documents
    yield "Searching for relevant hotels...\n"
    try:
        results = supabase.rpc('match_properties', {
            'query_embedding': query_embedding,
            'match_threshold': MATCH_THRESHOLD,
            'match_count': MATCH_COUNT
        }).execute()
        
        matches = results.data
    except Exception as e:
        yield f"Error searching database: {e}"
        return

    if not matches:
        yield "I couldn't find any hotels that match your request. Try rephrasing your search."
        return

    # 3. Augment: Create the context
    context = "You are a helpful Tokyo hotel assistant. Answer the user's question based *only* on the following context. Do not make up information. Be concise and friendly.\n\n--- CONTEXT ---\n"
    for i, match in enumerate(matches):
        context += f"Result {i+1}:\n{match['rag_document']}\n\n"
    
    context += "--- END CONTEXT ---\n\nUser Question: " + user_query

    # 4. Generate: Stream the response from Gemini
    yield "Found matches! Asking the AI...\n\n"
    try:
        response = gen_model.generate_content(context, stream=True)
        for chunk in response:
            yield chunk.text
    except Exception as e:
        yield f"Error generating response: {e}"

# --- Streamlit UI ---

st.title("🏙️ Tokyo Hotel & Airbnb Chatbot")
st.caption(f"Powered by Supabase, {EMBEDDING_MODEL_NAME}, and {GENERATIVE_MODEL_NAME}")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask me about hotels in Tokyo..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        # This will display the "Embedding...", "Searching..." messages
        # and then stream the final answer.
        response_stream = st.write_stream(get_rag_response(prompt))
        
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response_stream})
