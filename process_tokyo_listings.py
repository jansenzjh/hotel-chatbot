import pandas as pd
import re
import json

def clean_price(price_str):
    """
    Cleans a price string like '$1,500.00' into a float '1500.0'.
    Handles NaNs (empty values) by returning 0.0.
    """
    if pd.isna(price_str):
        return 0.0
    
    # Remove '$', ',', and whitespace
    try:
        # Remove characters that are not digits or a decimal point
        cleaned_str = re.sub(r"[^0-9.]", "", str(price_str))
        if cleaned_str:
            return float(cleaned_str)
        else:
            return 0.0
    except ValueError:
        return 0.0

def create_rag_document(row):
    """
    Combines structured data into a single text document
    for embedding.
    """
    doc_parts = []

    # Name and description
    doc_parts.append(f"Name: {row.get('name', 'N/A')}")
    if pd.notna(row.get('description')):
        doc_parts.append(f"Description: {row.get('description')}")
    if pd.notna(row.get('neighborhood_overview')):
        doc_parts.append(f"Neighborhood Overview: {row.get('neighborhood_overview')}")

    # Location
    location_parts = []
    if pd.notna(row.get('neighbourhood_cleansed')):
        location_parts.append(row.get('neighbourhood_cleansed'))
    if pd.notna(row.get('neighbourhood_group_cleansed')):
        location_parts.append(row.get('neighbourhood_group_cleansed'))
    if location_parts:
        doc_parts.append(f"Location: {', '.join(location_parts)}")

    # Property details
    property_details = []
    if pd.notna(row.get('property_type')):
        property_details.append(f"Property Type: {row.get('property_type')}")
    if pd.notna(row.get('room_type')):
        property_details.append(f"Room Type: {row.get('room_type')}")
    if property_details:
        doc_parts.append(". ".join(property_details) + ".")
    
    doc_parts.append(f"Accommodates: {row.get('accommodates', 1)} guest(s), with {row.get('bedrooms', 0)} bedroom(s) and {row.get('beds', 0)} bed(s). It has {row.get('bathrooms_text', 'a bathroom')}.")

    # Amenities
    amenities = row.get('amenities', '[]')
    if amenities and amenities != '[]':
        try:
            amenities_list = eval(amenities)
            if isinstance(amenities_list, list) and len(amenities_list) > 0:
                amenities_str = ", ".join(amenities_list)
                doc_parts.append(f"Amenities include: {amenities_str}.")
        except:
            pass

    # Host information
    host_info = []
    if pd.notna(row.get('host_name')):
        host_info.append(f"The host is {row.get('host_name')}")
    if pd.notna(row.get('host_since')):
        host_info.append(f"hosting since {row.get('host_since')}")
    if host_info:
        doc_parts.append(", ".join(host_info) + ".")
    
    if row.get('host_is_superhost') == 't':
        doc_parts.append("The host is a Superhost.")
    
    if pd.notna(row.get('host_about')):
        doc_parts.append(f"About the host: {row.get('host_about')}")

    # Reviews
    review_score = row.get('review_scores_rating', 0)
    review_count = row.get('number_of_reviews', 0)
    if review_count > 0:
        doc_parts.append(f"It has a rating of {review_score:.2f}/5.00 from {review_count} reviews.")

    # Price
    doc_parts.append(f"Price: Around {row.get('price_cleaned', 0)} JPY per night.")

    # Booking details
    booking_details = []
    booking_details.append(f"Minimum stay: {row.get('minimum_nights', 1)} night(s).")
    if row.get('instant_bookable') == 't':
        booking_details.append("Instant bookable is available.")
    doc_parts.append(" ".join(booking_details))

    return "\n".join(doc_parts)


def process_listings(input_csv='listings.csv', output_jsonl='clean_listings.jsonl'):
    """
    Main function to load, clean, and prepare data for Supabase.
    """
    print(f"Loading data from {input_csv}...")
    try:
        df = pd.read_csv(input_csv)
    except FileNotFoundError:
        print(f"Error: {input_csv} not found.")
        print("Please download the 'listings.csv' from Kaggle and place it in this directory.")
        return

    print(f"Loaded {len(df)} listings.")

    # 1. Select the columns we care about
    columns_to_keep = [
        'id', 'listing_url', 'last_scraped', 'source', 'name', 'description', 
        'neighborhood_overview', 'host_url', 'host_name', 'host_since', 
        'host_location', 'host_about', 'host_response_time', 'host_response_rate', 
        'host_acceptance_rate', 'host_is_superhost', 'host_neighbourhood', 
        'host_listings_count', 'host_total_listings_count', 'host_verifications', 
        'host_has_profile_pic', 'host_identity_verified', 'neighbourhood', 
        'neighbourhood_cleansed', 'neighbourhood_group_cleansed', 'latitude', 
        'longitude', 'property_type', 'room_type', 'accommodates', 'bathrooms', 
        'bathrooms_text', 'bedrooms', 'beds', 'amenities', 'price', 'minimum_nights', 
        'maximum_nights', 'minimum_minimum_nights', 'maximum_minimum_nights', 
        'minimum_maximum_nights', 'maximum_maximum_nights', 'minimum_nights_avg_ntm', 
        'maximum_nights_avg_ntm', 'calendar_updated', 'has_availability', 
        'availability_30', 'availability_60', 'availability_90', 'availability_365', 
        'calendar_last_scraped', 'number_of_reviews', 'number_of_reviews_ltm', 
        'number_of_reviews_l30d', 'review_scores_rating', 'review_scores_accuracy', 
        'review_scores_cleanliness', 'review_scores_checkin', 
        'review_scores_communication', 'review_scores_location', 'review_scores_value', 
        'license', 'instant_bookable', 'calculated_host_listings_count', 
        'calculated_host_listings_count_entire_homes', 
        'calculated_host_listings_count_private_rooms', 
        'calculated_host_listings_count_shared_rooms', 'reviews_per_month'
    ]
    
    available_columns = [col for col in columns_to_keep if col in df.columns]
    print(f"Keeping {len(available_columns)} columns.")
    df = df[available_columns]

    # 2. Clean the structured data
    print("Cleaning data...")
    df['price_cleaned'] = df['price'].apply(clean_price)
    
    # Handle missing values (NaNs)
    for col in ['name', 'description', 'neighborhood_overview', 'host_about', 'host_name', 'host_location', 'host_neighbourhood', 'bathrooms_text', 'license', 'host_since', 'host_response_time', 'neighbourhood', 'neighbourhood_cleansed', 'neighbourhood_group_cleansed', 'property_type', 'room_type', 'amenities', 'host_verifications']:
        if col in df.columns:
            df[col] = df[col].fillna('')
            
    for col in ['host_response_rate', 'host_acceptance_rate']:
        if col in df.columns:
            df[col] = df[col].fillna('N/A')

    for col in ['host_is_superhost', 'host_has_profile_pic', 'host_identity_verified', 'has_availability', 'instant_bookable']:
        if col in df.columns:
            df[col] = df[col].fillna('f')

    numeric_cols = [
        'host_listings_count', 'host_total_listings_count', 'accommodates', 'bathrooms', 
        'bedrooms', 'beds', 'minimum_nights', 'maximum_nights', 'minimum_minimum_nights', 
        'maximum_minimum_nights', 'minimum_maximum_nights', 'maximum_maximum_nights', 
        'minimum_nights_avg_ntm', 'maximum_nights_avg_ntm', 'availability_30', 
        'availability_60', 'availability_90', 'availability_365', 'number_of_reviews', 
        'number_of_reviews_ltm', 'number_of_reviews_l30d', 'review_scores_rating', 
        'review_scores_accuracy', 'review_scores_cleanliness', 'review_scores_checkin', 
        'review_scores_communication', 'review_scores_location', 'review_scores_value', 
        'calculated_host_listings_count', 'calculated_host_listings_count_entire_homes', 
        'calculated_host_listings_count_private_rooms', 
        'calculated_host_listings_count_shared_rooms', 'reviews_per_month'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # 3. Create the RAG document for each listing
    print("Creating RAG documents...")
    df['rag_document'] = df.apply(create_rag_document, axis=1)
    
    # 4. Save the prepared data to a JSONL file
    # JSONL (JSON Lines) is a great format for this: one JSON object per line.
    
    # Select final columns for the output file
    final_columns = available_columns + ['rag_document', 'price_cleaned']
    if 'price' in final_columns:
        final_columns.remove('price') # use price_cleaned instead
    
    # Ensure all final columns exist
    final_df = df[[col for col in final_columns if col in df.columns]]
    
    print(f"Saving {len(final_df)} processed listings to {output_jsonl}...")
    
    # Convert DataFrame to a list of dictionaries
    records = final_df.to_dict('records')
    
    # Write to JSONL file
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for record in records:
            json.dump(record, f)
            f.write('\n')
            
    print("\nDone! Your data is prepared.")
    print(f"See '{output_jsonl}' for the clean data.")
    print("\n--- Example Record ---")
    if records:
        print(json.dumps(records[0], indent=2))
    print("------------------------")

if __name__ == "__main__":
    # Make sure you have downloaded 'listings.csv' and placed it
    # in the same directory as this script.
    process_listings()
