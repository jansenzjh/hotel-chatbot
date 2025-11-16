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
    # Start with the name and description
    doc_parts = [
        f"Name: {row.get('name', 'N/A')}",
        f"Description: {row.get('description', 'No description provided.')}"
    ]
    
    # Add neighborhood and room type
    doc_parts.append(f"Location: Located in {row.get('neighbourhood_cleansed', 'Tokyo')}.")
    doc_parts.append(f"Room Type: This is a {row.get('room_type', 'standard room')}.")
    
    # Add amenities
    amenities = row.get('amenities', '[]') # Amenities are often a string representation of a list
    if amenities:
        try:
            # Safely evaluate the string as a Python list
            amenities_list = eval(amenities)
            if isinstance(amenities_list, list) and len(amenities_list) > 0:
                amenities_str = ", ".join(amenities_list[:10]) # Get first 10
                doc_parts.append(f"Amenities include: {amenities_str}.")
        except:
            pass # Ignore errors if 'amenities' is not a valid list string

    # Add pricing and capacity
    doc_parts.append(f"Price: Around {row.get('price_cleaned', 0)} JPY per night.")
    doc_parts.append(f"Accommodates: {row.get('accommodates', 1)} guest(s).")
    
    # Join all parts with a newline
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
    # (You can add/remove columns here)
    columns_to_keep = [
        'id',
        'name',
        'description',
        'neighbourhood_cleansed',
        'room_type',
        'price',
        'accommodates',
        'bedrooms',
        'beds',
        'amenities',
        'latitude',
        'longitude'
    ]
    
    # Filter for only the columns we need
    # Check which columns are actually in the file
    available_columns = [col for col in columns_to_keep if col in df.columns]
    print(f"Keeping {len(available_columns)} columns.")
    df = df[available_columns]

    # 2. Clean the structured data
    print("Cleaning data...")
    # Clean the price column
    df['price_cleaned'] = df['price'].apply(clean_price)
    
    # Handle missing values (NaNs)
    df['name'] = df['name'].fillna('')
    df['description'] = df['description'].fillna('')
    df['bedrooms'] = df['bedrooms'].fillna(0)
    df['beds'] = df['beds'].fillna(0)

    # 3. Create the RAG document for each listing
    print("Creating RAG documents...")
    df['rag_document'] = df.apply(create_rag_document, axis=1)
    
    # 4. Save the prepared data to a JSONL file
    # JSONL (JSON Lines) is a great format for this: one JSON object per line.
    
    # Select final columns for the output file
    final_columns = [
        'id',
        'name',
        'rag_document',
        'price_cleaned',
        'neighbourhood_cleansed',
        'room_type',
        'accommodates',
        'bedrooms',
        'beds',
        'latitude',
        'longitude'
    ]
    
    # Ensure all final columns exist (some might have been dropped if not in original CSV)
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
    print(json.dumps(records[0], indent=2))
    print("------------------------")

if __name__ == "__main__":
    # Make sure you have downloaded 'listings.csv' and placed it
    # in the same directory as this script.
    process_listings()