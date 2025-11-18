"""
Core RAG (Retrieval-Augmented Generation) pipeline logic.
"""
from services import EmbeddingModel, VectorStore, GenerativeModel

class RagPipeline:
    """
    Encapsulates the logic for the RAG pipeline.
    """
    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_store: VectorStore,
        generative_model: GenerativeModel
    ):
        self._embedding_model = embedding_model
        self._vector_store = vector_store
        self._generative_model = generative_model

    def get_rag_response(self, user_query: str, chat_history: list, match_threshold: float, match_count: int):
        """
        Performs the full RAG pipeline and streams the response.
        """
        # 1. Embed the user's query
        yield "Embedding your query...\n\n"
        query_embedding = self._embedding_model.embed(user_query)

        # 2. Search for similar documents
        yield "Searching for relevant hotels...\n\n"
        matches = self._vector_store.search(query_embedding, match_threshold, match_count)

        if not matches:
            yield "I couldn't find any hotels that match your request. Try rephrasing your search."
            return

        # 3. Augment: Create the context
        rag_context = "--- CONTEXT ---\n"
        listing_urls = []
        for i, match in enumerate(matches):
            rag_context += f"Result {i+1}:\n{match['rag_document']}\n\n"
            listing_urls.append(match['listing_url'])
        rag_context += "--- END CONTEXT ---\n\n"

        # Build the full prompt with history
        prompt_with_history = (
            "You are a helpful Tokyo hotel assistant.\n"
            "1. Answer the user's question based *only* on the provided context.\n"
            "2. If the user asks a follow-up question, you can use the chat history to understand it.\n"
            "3. Do not make up information. Be concise and friendly.\n"
            "4. After providing the summary/answer, create a new section titled 'Hotels Found:'.\n"
            "5. In this new section, list the **names** of all hotels you used to answer the question with bullet points.\n\n"
        )

        # Add chat history to the prompt
        if chat_history:
            prompt_with_history += "--- CHAT HISTORY ---\n"
            for message in chat_history:
                prompt_with_history += f"{message['role']}: {message['content']}\n"
            prompt_with_history += "--- END CHAT HISTORY ---\n\n"

        prompt_with_history += rag_context
        prompt_with_history += "User Question: " + user_query

        # 4. Generate: Stream the response
        yield "Found matches! Asking the AI...\n\n"
        response_stream = self._generative_model.generate(prompt_with_history, stream=True)
        
        # Store listing_urls to be accessed by the UI
        self.listing_urls = listing_urls

        for chunk in response_stream:
            yield chunk.text
