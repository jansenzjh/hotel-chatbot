import streamlit as st
import os
import google.generativeai as genai
from supabase import create_client, Client
import ollama
from dotenv import load_dotenv

# --- Configuration ---
load_dotenv()
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

# Ollama Embedding Model
EMBEDDING_MODEL_NAME = 'mxbai-embed-large' 
# Vector dimension for mxbai-embed-large
VECTOR_DIMENSION = 1024 

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
def init_generative_model() -> genai.GenerativeModel:
    """Initializes and returns the Gemini generative model."""
    genai.configure(api_key=GOOGLE_API_KEY)
    return genai.GenerativeModel(GENERATIVE_MODEL_NAME)

# --- Main RAG Function ---

def get_rag_response(user_query: str):
    """
    Performs the full RAG pipeline and streams the response.
    """
    supabase = init_supabase()
    gen_model = init_generative_model()

    # 1. Embed the user's query
    yield "Embedding your query...\n\n"
    try:
        response = ollama.embeddings(model=EMBEDDING_MODEL_NAME, prompt=user_query)
        query_embedding = response["embedding"]
    except Exception as e:
        yield f"Error embedding query with Ollama: {e}"
        return

    # 2. Search Supabase for similar documents
    yield "Searching for relevant hotels...\n\n"
    try:
        results = supabase.rpc('match_properties_1024', {
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
# Define a clearer, multi-step prompt for the AI
    context = (
        "You are a helpful Tokyo hotel assistant.\n"
        "1. Answer the user's question based *only* on the following context.\n"
        "2. Do not make up information. Be concise and friendly.\n"
        "3. After providing the summary/answer, create a new section titled 'Hotels Found:'.\n"
        "4. In this new section, list the **names** of all hotels you used to answer the question with bullet points.\n\n"
        "--- CONTEXT ---\n"
    )
    
    listing_urls = []
    for i, match in enumerate(matches):
        context += f"Result {i+1}:\n{match['rag_document']}\n\n"
        listing_urls.append(match['listing_url'])
    
    st.session_state.listing_urls = listing_urls
    
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
if prompt := st.chat_input("Ask me about hotels in Tokyo (eg. 'hotel in shinjuku provide dryer and washer')..."):
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

    # Display listing URLs if available
    if "listing_urls" in st.session_state and st.session_state.listing_urls:
        st.markdown("Listing URLs (static data from 2023, links could be invalid):\n\n")
        for url in st.session_state.listing_urls:
            st.markdown(f"- {url}")
        st.session_state.listing_urls = [] # Clear listing_urls after displaying
