import os
import json
import time
import psycopg2
import ollama
from dotenv import load_dotenv
from config import DB_HOST, DB_NAME, DB_USER, DB_PASSWORD, DB_PORT, EMBEDDING_MODEL_NAME

# --- Configuration ---
load_dotenv()  # Load environment variables from .env file

INPUT_FILE = 'clean_listings.jsonl'
TABLE_NAME = 'listings'
BATCH_SIZE = 50  # Number of listings to process and upload at a time
EMBEDDING_MODEL = EMBEDDING_MODEL_NAME # Use model from config
# --- End Configuration ---

def get_embedding(text: str, model_name: str) -> list[float] | None:
    """Generates an embedding for the given text using the Ollama model.
       Retries with truncation if the context length is exceeded.
    """
    if not text or not isinstance(text, str) or len(text.strip()) == 0:
        print("Warning: Skipping empty or invalid text.")
        return None

    # Try with different lengths: Full -> 4000 -> 1000 -> 500
    lengths_to_try = [8000, 4000, 1000, 500]

    for length in lengths_to_try:
        try:
            truncated_text = text[:length]
            response = ollama.embeddings(model=model_name, prompt=truncated_text)
            return response["embedding"]
        except Exception as e:
            # Check if it's a context length error (likely 500 status code)
            error_msg = str(e).lower()
            if "500" in error_msg or "context length" in error_msg:
                print(f"Warning: Embedding failed with length {length}. Retrying with truncation...")
                continue # Try next shorter length
            else:
                # Actual error (e.g. connection lost), stop trying
                print(f"Error generating embedding for text: {text[:50]}...")
                print(f"Error: {e}")
                return None
    
    print(f"Failed to embed even after truncation to {lengths_to_try[-1]} chars.")
    return None


def sanitize_record(record: dict) -> dict:
    """Replaces NaN/inf float values with None for JSON compliance."""
    for key, value in record.items():
        if isinstance(value, float) and (value != value or value == float('inf') or value == float('-inf')):
            record[key] = None
    return record


def upload_batch(conn, batch: list[dict]):
    """Uploads a batch of listings to the Postgres table."""
    if not batch:
        return 0
    
    try:
        with conn.cursor() as cur:
            # Assuming all records have the same keys (which they update below)
            # We get columns from the first record
            columns = list(batch[0].keys())
            
            # Construct the SQL query dynamically
            # INSERT INTO table (col1, col2) VALUES (%s, %s) ON CONFLICT (id) DO NOTHING
            cols_str = ', '.join(columns)
            placeholders = ', '.join(['%s'] * len(columns))
            sql = f"INSERT INTO {TABLE_NAME} ({cols_str}) VALUES ({placeholders}) ON CONFLICT (id) DO NOTHING"
            
            # Prepare values
            values = [tuple(record[c] for c in columns) for record in batch]
            
            cur.executemany(sql, values)
            conn.commit()
            
            print(f"Successfully uploaded batch of {len(batch)} listings.")
            return len(batch)

    except Exception as e:
        conn.rollback() # Rollback in case of error
        print(f"Error uploading batch to Postgres: {e}")
        return 0


def main():
    """Main script to process and upload data."""
    
    # 1. Connect to Database
    conn = None # Initialize conn to None
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            port=DB_PORT
        )
        print("Connected to PostgreSQL successfully.")
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return

    print("Starting data processing...")

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
                    embedding = get_embedding(text_to_embed, EMBEDDING_MODEL)
                    
                    if embedding is None:
                        print(f"Warning: Failed to embed listing {listing.get('id')}.")
                        total_failed += 1
                        continue

                    # 4. Prepare the final DB record
                    # This maps our JSON keys to the exact SQL table column names
                    db_record = {
                        'id': listing.get('id'),
                        'listing_url': listing.get('listing_url', ''),
                        'last_scraped': listing.get('last_scraped', ''),
                        'source': listing.get('source', ''),
                        'name': listing.get('name', ''),
                        'description': listing.get('description', ''),
                        'neighborhood_overview': listing.get('neighborhood_overview', ''),
                        'host_url': listing.get('host_url', ''),
                        'host_name': listing.get('host_name', ''),
                        'host_since': listing.get('host_since', ''),
                        'host_location': listing.get('host_location', ''),
                        'host_about': listing.get('host_about', ''),
                        'host_response_time': listing.get('host_response_time', ''),
                        'host_response_rate': listing.get('host_response_rate', ''),
                        'host_acceptance_rate': listing.get('host_acceptance_rate', ''),
                        'host_is_superhost': listing.get('host_is_superhost', ''),
                        'host_neighbourhood': listing.get('host_neighbourhood', ''),
                        'host_listings_count': int(listing.get('host_listings_count', 0)),
                        'host_total_listings_count': int(listing.get('host_total_listings_count', 0)),
                        'host_verifications': listing.get('host_verifications', ''),
                        'host_has_profile_pic': listing.get('host_has_profile_pic', ''),
                        'host_identity_verified': listing.get('host_identity_verified', ''),
                        'neighbourhood': listing.get('neighbourhood', ''),
                        'neighbourhood_cleansed': listing.get('neighbourhood_cleansed', ''),
                        'neighbourhood_group_cleansed': listing.get('neighbourhood_group_cleansed', ''),
                        'latitude': float(listing.get('latitude', 0.0)),
                        'longitude': float(listing.get('longitude', 0.0)),
                        'property_type': listing.get('property_type', ''),
                        'room_type': listing.get('room_type', ''),
                        'accommodates': int(listing.get('accommodates', 0)),
                        'bathrooms': float(listing.get('bathrooms', 0.0)),
                        'bathrooms_text': listing.get('bathrooms_text', ''),
                        'bedrooms': float(listing.get('bedrooms', 0.0)),
                        'beds': float(listing.get('beds', 0.0)),
                        'amenities': listing.get('amenities', ''),
                        'price_cleaned': float(listing.get('price_cleaned', 0.0)),
                        'minimum_nights': int(listing.get('minimum_nights', 0)),
                        'maximum_nights': int(listing.get('maximum_nights', 0)),
                        'minimum_minimum_nights': int(listing.get('minimum_minimum_nights', 0)),
                        'maximum_minimum_nights': int(listing.get('maximum_minimum_nights', 0)),
                        'minimum_maximum_nights': int(listing.get('minimum_maximum_nights', 0)),
                        'maximum_maximum_nights': int(listing.get('maximum_maximum_nights', 0)),
                        'minimum_nights_avg_ntm': float(listing.get('minimum_nights_avg_ntm', 0.0)),
                        'maximum_nights_avg_ntm': float(listing.get('maximum_nights_avg_ntm', 0.0)),
                        'calendar_updated': listing.get('calendar_updated', ''),
                        'has_availability': listing.get('has_availability', ''),
                        'availability_30': int(listing.get('availability_30', 0)),
                        'availability_60': int(listing.get('availability_60', 0)),
                        'availability_90': int(listing.get('availability_90', 0)),
                        'availability_365': int(listing.get('availability_365', 0)),
                        'calendar_last_scraped': listing.get('calendar_last_scraped', ''),
                        'number_of_reviews': int(listing.get('number_of_reviews', 0)),
                        'number_of_reviews_ltm': int(listing.get('number_of_reviews_ltm', 0)),
                        'number_of_reviews_l30d': int(listing.get('number_of_reviews_l30d', 0)),
                        'review_scores_rating': float(listing.get('review_scores_rating', 0.0)),
                        'review_scores_accuracy': float(listing.get('review_scores_accuracy', 0.0)),
                        'review_scores_cleanliness': float(listing.get('review_scores_cleanliness', 0.0)),
                        'review_scores_checkin': float(listing.get('review_scores_checkin', 0.0)),
                        'review_scores_communication': float(listing.get('review_scores_communication', 0.0)),
                        'review_scores_location': float(listing.get('review_scores_location', 0.0)),
                        'review_scores_value': float(listing.get('review_scores_value', 0.0)),
                        'license': listing.get('license', ''),
                        'instant_bookable': listing.get('instant_bookable', ''),
                        'calculated_host_listings_count': int(listing.get('calculated_host_listings_count', 0)),
                        'calculated_host_listings_count_entire_homes': int(listing.get('calculated_host_listings_count_entire_homes', 0)),
                        'calculated_host_listings_count_private_rooms': int(listing.get('calculated_host_listings_count_private_rooms', 0)),
                        'calculated_host_listings_count_shared_rooms': int(listing.get('calculated_host_listings_count_shared_rooms', 0)),
                        'reviews_per_month': float(listing.get('reviews_per_month', 0.0)),
                        'rag_document': text_to_embed,
                        'embedding': embedding
                    }
                    
                    listings_batch.append(sanitize_record(db_record))

                    # 5. Upload batch when full
                    if len(listings_batch) >= BATCH_SIZE:
                        print(f"\nProcessing line {i+1}...")
                        uploaded_count = upload_batch(conn, listings_batch)
                        total_success += uploaded_count
                        total_failed += (BATCH_SIZE - uploaded_count)
                        listings_batch = []  # Clear the batch
                        time.sleep(0.5) # Reduced sleep as local DB is faster

                except json.JSONDecodeError:
                    print(f"Error decoding JSON on line {i+1}. Skipping.")
                    total_failed += 1
                except Exception as e:
                    print(f"An unexpected error occurred processing line {i+1}: {e}")
                    total_failed += 1

            # 6. Upload any remaining listings in the last batch
            if listings_batch:
                print("\nUploading final batch...")
                uploaded_count = upload_batch(conn, listings_batch)
                total_success += uploaded_count
                total_failed += (len(listings_batch) - uploaded_count)

    except FileNotFoundError:
        print(f"Error: Input file '{INPUT_FILE}' not found.")
        print("Did you run the 'process_tokyo_listings.py' script first?")
    except Exception as e:
        print(f"A fatal error occurred: {e}")
    finally:
        if conn:
            conn.close()

    print("\n--- Upload Complete ---")
    print(f"Total Successful: {total_success}")
    print(f"Total Failed: {total_failed}")
    print("------------------------")

if __name__ == "__main__":
    main()