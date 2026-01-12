"""
Abstractions and implementations for external services.
"""
import logging
import time
from abc import ABC, abstractmethod
from supabase import create_client, Client, ClientOptions
import google.generativeai as genai
import ollama
from config import SUPABASE_URL, SUPABASE_KEY, GOOGLE_API_KEY, EMBEDDING_MODEL_NAME

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Abstract Base Classes ---

class EmbeddingModel(ABC):
    """Abstract base class for an embedding model."""
    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Embeds a string of text and returns the embedding."""
        pass

class VectorStore(ABC):
    """Abstract base class for a vector store."""
    @abstractmethod
    def search(self, embedding: list[float], match_threshold: float, match_count: int, filters: dict = None) -> list[dict]:
        """Searches for similar documents in the vector store."""
        pass

class GenerativeModel(ABC):
    """Abstract base class for a generative model."""
    @abstractmethod
    def generate(self, context: str, stream: bool = False):
        """Generates content based on the given context."""
        pass
    
    @abstractmethod
    def extract_filters(self, query: str) -> dict:
        """Extracts structured filters from a natural language query."""
        pass

# --- Concrete Implementations ---

class OllamaEmbeddingModel(EmbeddingModel):
    """Implementation of the embedding model using Ollama."""
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self._model_name = model_name
        logging.info(f"Initialized OllamaEmbeddingModel with model: {model_name}")

    def embed(self, text: str) -> list[float]:
        """Embeds text using the Ollama model."""
        logging.info(f"Embedding text with Ollama: '{text[:50]}...'")
        try:
            response = ollama.embeddings(model=self._model_name, prompt=text)
            logging.info("Successfully embedded text with Ollama.")
            return response["embedding"]
        except Exception as e:
            logging.error(f"Error embedding query with Ollama: {e}")
            raise RuntimeError(f"Error embedding query with Ollama: {e}") from e

class SupabaseVectorStore(VectorStore):
    """Implementation of the vector store using Supabase."""
    def __init__(self):
        logging.info("Initializing SupabaseVectorStore...")
        options = ClientOptions(postgrest_client_timeout=30)
        self._client: Client = create_client(
            SUPABASE_URL,
            SUPABASE_KEY,
            options=options
        )
        logging.info("SupabaseVectorStore initialized successfully.")

    def search(self, embedding: list[float], match_threshold: float, match_count: int, filters: dict = None) -> list[dict]:
        """Searches for similar properties in Supabase."""
        retries = 3
        delay = 2
        
        rpc_params = {
            'query_embedding': embedding,
            'match_threshold': match_threshold,
            'match_count': match_count
        }
        
        # Add filters if present
        if filters:
            if 'min_price' in filters and filters['min_price'] is not None:
                rpc_params['min_price'] = filters['min_price']
            if 'max_price' in filters and filters['max_price'] is not None:
                rpc_params['max_price'] = filters['max_price']
        
        logging.info(f"Searching Supabase with params: {rpc_params}")
        
        rpc_function = 'match_properties_filtered' 
        
        for i in range(retries):
            try:
                results = self._client.rpc(rpc_function, rpc_params).execute()
                logging.info(f"Supabase search successful, found {len(results.data)} matches.")
                return results.data
            except Exception as e:
                logging.warning(f"Supabase search attempt {i+1}/{retries} failed: {e}")
                # Fallback to original function if the new one doesn't exist yet (optional safety)
                if "function match_properties_filtered" in str(e) and "does not exist" in str(e):
                     logging.warning("Fallback to match_properties_1024 (ignoring filters)")
                     try:
                        fallback_params = {
                            'query_embedding': embedding,
                            'match_threshold': match_threshold,
                            'match_count': match_count
                        }
                        results = self._client.rpc('match_properties_1024', fallback_params).execute()
                        return results.data
                     except Exception as fallback_e:
                         logging.error(f"Fallback failed: {fallback_e}")

                if i < retries - 1:
                    time.sleep(delay)
                else:
                    logging.error("All Supabase search attempts failed.")
                    raise RuntimeError(f"Error searching database: {e}") from e

class GeminiGenerativeModel(GenerativeModel):
    """Implementation of the generative model using Google Gemini."""
    def __init__(self, model_name: str):
        genai.configure(api_key=GOOGLE_API_KEY)
        self._model = genai.GenerativeModel(model_name)
        logging.info(f"Initialized GeminiGenerativeModel with model: {model_name}")

    def generate(self, context: str, stream: bool = False):
        """Generates content using the Gemini model."""
        logging.info("Generating content with Gemini...")
        try:
            response = self._model.generate_content(context, stream=stream)
            logging.info("Successfully started content generation stream with Gemini.")
            return response
        except Exception as e:
            logging.error(f"Error generating response from Gemini: {e}")
            raise RuntimeError(f"Error generating response: {e}") from e

    def extract_filters(self, query: str) -> dict:
        """
        Extracts structured filters (min_price, max_price) from the query using Gemini.
        Returns a dictionary like {'min_price': 100, 'max_price': 500} or empty dict if none.
        """
        import json
        import re

        prompt = f"""
        Analyze the following user query for hotel searches. 
        Extract any price constraints. 
        Return ONLY a JSON object with keys 'min_price' and 'max_price'.
        Values should be numbers (floats or integers).
        If no constraint is found for a key, use null.
        
        Query: "{query}"
        
        JSON:
        """
        
        try:
            logging.info("Extracting filters with Gemini...")
            response = self._model.generate_content(prompt)
            text_response = response.text.strip()
            
            # Remove potential markdown code blocks ```json ... ```
            text_response = re.sub(r'```json\s*', '', text_response)
            text_response = re.sub(r'```', '', text_response)
            
            filters = json.loads(text_response)
            
            logging.info(f"Extracted filters: {filters}")
            return filters
            
        except Exception as e:
            logging.error(f"Error extraction filters: {e}")
            return {}
