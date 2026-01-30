"""
Abstractions and implementations for external services.
"""
import logging
import time
from abc import ABC, abstractmethod
import psycopg2
import google.generativeai as genai
import ollama
from psycopg2.extras import RealDictCursor
from config import DB_HOST, DB_NAME, DB_USER, DB_PASSWORD, DB_PORT, GOOGLE_API_KEY, EMBEDDING_MODEL_NAME

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
            # Truncate to avoid exceeding 512 token limit. Reduced to 500 chars.
            truncated_text = text[:500]
            response = ollama.embeddings(model=self._model_name, prompt=truncated_text)
            logging.info("Successfully embedded text with Ollama.")
            return response["embedding"]
        except Exception as e:
            logging.error(f"Error embedding query with Ollama: {e}")
            raise RuntimeError(f"Error embedding query with Ollama: {e}") from e

class PostgresVectorStore(VectorStore):
    """Implementation of the vector store using Local PostgreSQL + pgvector."""
    def __init__(self):
        logging.info("Initializing PostgresVectorStore...")
        try:
            self._connection_params = {
                "host": DB_HOST,
                "database": DB_NAME,
                "user": DB_USER,
                "password": DB_PASSWORD,
                "port": DB_PORT
            }
            # Test connection
            conn = psycopg2.connect(**self._connection_params)
            conn.close()
            logging.info("PostgresVectorStore initialized successfully (connection test passed).")
        except Exception as e:
            logging.error(f"Failed to connect to PostgreSQL: {e}")
            raise RuntimeError("Could not connect to database") from e

    def search(self, embedding: list[float], match_threshold: float, match_count: int, filters: dict = None) -> list[dict]:
        """Searches for similar properties using the match_properties_filtered function."""
        retries = 3
        delay = 2
        
        # Prepare arguments for the SQL function
        min_price = filters.get('min_price') if filters else None
        max_price = filters.get('max_price') if filters else None
        
        logging.info(f"Searching Postgres with filters: min={min_price}, max={max_price}")
        
        for i in range(retries):
            conn = None
            try:
                conn = psycopg2.connect(**self._connection_params)
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    # Convert list to string format for pgvector casting
                    embedding_str = f"[{','.join(map(str, embedding))}]"
                    
                    # Call the stored procedure
                    # match_properties_filtered(query_embedding, match_threshold, match_count, min_price, max_price)
                    cur.callproc('match_properties_filtered', (
                        embedding_str,
                        match_threshold,
                        match_count,
                        min_price,
                        max_price
                    ))
                    results = cur.fetchall()
                    
                logging.info(f"Postgres search successful, found {len(results)} matches.")
                return results

            except Exception as e:
                logging.warning(f"Postgres search attempt {i+1}/{retries} failed: {e}")
                if i < retries - 1:
                    time.sleep(delay)
                else:
                    logging.error("All Postgres search attempts failed.")
                    raise RuntimeError(f"Error searching database: {e}") from e
            finally:
                if conn:
                    conn.close()

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
