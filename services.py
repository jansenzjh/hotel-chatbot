"""
Abstractions and implementations for external services.
"""
from abc import ABC, abstractmethod
from supabase import create_client, Client
import google.generativeai as genai
import ollama
from config import SUPABASE_URL, SUPABASE_KEY, GOOGLE_API_KEY, EMBEDDING_MODEL_NAME

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
    def search(self, embedding: list[float], match_threshold: float, match_count: int) -> list[dict]:
        """Searches for similar documents in the vector store."""
        pass

class GenerativeModel(ABC):
    """Abstract base class for a generative model."""
    @abstractmethod
    def generate(self, context: str, stream: bool = False):
        """Generates content based on the given context."""
        pass

# --- Concrete Implementations ---

class OllamaEmbeddingModel(EmbeddingModel):
    """Implementation of the embedding model using Ollama."""
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self._model_name = model_name

    def embed(self, text: str) -> list[float]:
        """Embeds text using the Ollama model."""
        try:
            response = ollama.embeddings(model=self._model_name, prompt=text)
            return response["embedding"]
        except Exception as e:
            # It's often better to raise a custom exception here
            raise RuntimeError(f"Error embedding query with Ollama: {e}") from e

class SupabaseVectorStore(VectorStore):
    """Implementation of the vector store using Supabase."""
    def __init__(self):
        self._client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

    def search(self, embedding: list[float], match_threshold: float, match_count: int) -> list[dict]:
        """Searches for similar properties in Supabase."""
        try:
            results = self._client.rpc('match_properties_1024', {
                'query_embedding': embedding,
                'match_threshold': match_threshold,
                'match_count': match_count
            }).execute()
            return results.data
        except Exception as e:
            raise RuntimeError(f"Error searching database: {e}") from e

class GeminiGenerativeModel(GenerativeModel):
    """Implementation of the generative model using Google Gemini."""
    def __init__(self, model_name: str):
        genai.configure(api_key=GOOGLE_API_KEY)
        self._model = genai.GenerativeModel(model_name)

    def generate(self, context: str, stream: bool = False):
        """Generates content using the Gemini model."""
        try:
            return self._model.generate_content(context, stream=stream)
        except Exception as e:
            raise RuntimeError(f"Error generating response: {e}") from e
