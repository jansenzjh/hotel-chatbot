import os
import json
import time
from dotenv import load_dotenv
from supabase import create_client, Client
import torch
from sentence_transformers import SentenceTransformer

# --- Configuration ---
load_dotenv()  # Load environment variables from .env file

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")
# GOOGLE_API_KEY is no longer needed for sentence_transformers
# genai.configure(api_key=GOOGLE_API_KEY) is also no longer needed

INPUT_FILE = './data_process/clean_listings.jsonl'
SUPABASE_TABLE = 'properties'
BATCH_SIZE = 50  # Number of listings to process and upload at a time
EMBEDDING_MODEL = "google/embeddinggemma-300m" # Hugging Face model name
# --- End Configuration ---

# Initialize the embedding model globally
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device for embedding: {device}")
try:
    embedding_model_instance = SentenceTransformer(EMBEDDING_MODEL).to(device)
except Exception as e:
    print(f"Error loading SentenceTransformer model: {e}")
    print("Please ensure the model is downloaded and available, and you have accepted its license on Hugging Face.")
    embedding_model_instance = None # Set to None if loading fails

def get_embedding(text: str, model_instance: SentenceTransformer) -> list[float] | None:
    """Generates an embedding for the given text using the SentenceTransformer model."""
    if not text or not isinstance(text, str) or len(text.strip()) == 0:
        print("Warning: Skipping empty or invalid text.")
        return None
    if model_instance is None:
        print("Error: Embedding model not loaded.")
        return None
        
    try:
        # The model automatically handles specific prefixes like 'query:' and 'text:' for optimal performance
        # For RAG, we use 'text:' prefix
        embedding = model_instance.encode([f"text: {text}"])[0]
        return embedding.tolist() # Convert numpy array to list
    except Exception as e:
        print(f"Error generating embedding for text: {text[:50]}...")
        print(f"Error: {e}")
        return None


def upload_batch(supabase: Client, batch: list[dict], max_retries: int = 3):
    """Uploads a batch of listings to the Supabase table with retry logic."""
    if not batch:
        return 0
    
    delay = 2.0 # Initial delay
    for i in range(max_retries):
        try:
            supabase.table(SUPABASE_TABLE).insert(batch).execute()
            print(f"Successfully uploaded batch of {len(batch)} listings.")
            return len(batch)
        except Exception as e:
            # PostgREST can also throw errors that might be wrapped.
            # A 429 is often a sign of hitting an API gateway limit or db proxy limit.
            if "429" in str(e) or "timed out" in str(e).lower():
                 print(f"Error uploading to Supabase: {e}. Retrying in {delay:.1f} seconds... (Attempt {i+1}/{max_retries})")
                 time.sleep(delay)
                 delay *= 2
            else:
                print(f"Error uploading batch to Supabase: {e}")
                # Log the failed batch for debugging if necessary
                # with open('failed_batch.json', 'w') as f:
                #     json.dump(batch, f)
                return 0

    print(f"Failed to upload batch after {max_retries} retries.")
    return 0

def main():
    """Main script to process and upload data."""
    
    # 1. Initialize Clients
    if not all([SUPABASE_URL, SUPABASE_KEY]):
        print("Error: Missing environment variables.")
        print("Please create a .env file with SUPABASE_URL, SUPABASE_SERVICE_KEY.")
        return

    # Check if the embedding model was loaded successfully
    if embedding_model_instance is None:
        print("Error: Embedding model failed to load. Exiting.")
        return

    try:
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as e:
        print(f"Error initializing clients: {e}")
        return

    print("Clients initialized. Starting data processing...")

    # 2. Open input file and process in batches
    listings_batch = []
    total_success = 0
    total_failed = 0

    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                try:
                    listing = json.loads(line)
                    text_to_embed = listing.get('rag_document')
                    
                    if not text_to_embed:
                        print(f"Warning: Skipping listing {listing.get('id')} - no rag_document.")
                        total_failed += 1
                        continue

                    # 3. Generate Embedding
                    # No proactive sleep needed as the model is local
                    embedding = get_embedding(text_to_embed, embedding_model_instance)
                    
                    if embedding is None:
                        print(f"Warning: Failed to embed listing {listing.get('id')}.")
                        total_failed += 1
                        continue

                    # 4. Prepare the final DB record
                    # This maps our JSON keys to the exact SQL table column names
                    db_record = {
                        'id': listing.get('id'),
                        'name': listing.get('name'),
                        'rag_document': text_to_embed,
                        'price_cleaned': listing.get('price_cleaned'),
                        # IMPORTANT: Map the key from the JSONL to the table column name
                        'neighbourhood': listing.get('neighbourhood_cleansed'), 
                        'room_type': listing.get('room_type'),
                        # Cast to integer to match table schema
                        'accommodates': int(listing.get('accommodates', 0)),
                        'bedrooms': int(listing.get('bedrooms', 0)),
                        'latitude': listing.get('latitude'),
                        'longitude': listing.get('longitude'),
                        'embedding': embedding  # Add the generated embedding
                    }
                    
                    listings_batch.append(db_record)

                    # 5. Upload batch when full
                    if len(listings_batch) >= BATCH_SIZE:
                        print(f"\nProcessing line {i+1}...")
                        uploaded_count = upload_batch(supabase, listings_batch)
                        total_success += uploaded_count
                        total_failed += (BATCH_SIZE - uploaded_count)
                        listings_batch = []  # Clear the batch
                        time.sleep(1) # Be nice to the APIs

                except json.JSONDecodeError:
                    print(f"Error decoding JSON on line {i+1}. Skipping.")
                    total_failed += 1
                except Exception as e:
                    print(f"An unexpected error occurred processing line {i+1}: {e}")
                    total_failed += 1

            # 6. Upload any remaining listings in the last batch
            if listings_batch:
                print("\nUploading final batch...")
                uploaded_count = upload_batch(supabase, listings_batch)
                total_success += uploaded_count
                total_failed += (len(listings_batch) - uploaded_count)

    except FileNotFoundError:
        print(f"Error: Input file '{INPUT_FILE}' not found.")
        print("Did you run the 'process_tokyo_listings.py' script first?")
        return
    except Exception as e:
        print(f"A fatal error occurred: {e}")
        return

    print("\n--- Upload Complete ---")
    print(f"Total Successful: {total_success}")
    print(f"Total Failed: {total_failed}")
    print("------------------------")

if __name__ == "__main__":
    main()