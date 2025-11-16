-- 1. Enable the pgvector extension
-- (You only need to run this once per project)
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Create the properties table
-- This table will store your hotel listings and their embeddings
CREATE TABLE properties (
    id BIGINT PRIMARY KEY,
    name TEXT,
    rag_document TEXT,
    price_cleaned FLOAT,
    neighbourhood TEXT,
    room_type TEXT,
    accommodates INT,
    bedrooms INT,
    latitude FLOAT,
    longitude FLOAT,
    
    -- IMPORTANT: Use the correct dimension for your embedding model
    -- 768 is for Google's 'embedding-gemma-300m'
    embedding VECTOR(768)
);

-- 3. Create a function to search for matching properties
-- This function will perform the vector similarity search (RAG)
CREATE OR REPLACE FUNCTION match_properties (
    -- The query vector must have the same dimension
    query_embedding VECTOR(768), 
    match_threshold FLOAT,
    match_count INT
)
RETURNS TABLE (
    id BIGINT,
    rag_document TEXT,
    -- We calculate similarity as 1 - distance
    similarity FLOAT 
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        properties.id,
        properties.rag_document,
        1 - (properties.embedding <=> query_embedding) AS similarity
    FROM properties
    -- The <=> operator calculates the cosine distance (0=identical, 2=opposite)
    -- So, 1 - distance gives us similarity (1=identical, -1=opposite)
    WHERE 1 - (properties.embedding <=> query_embedding) > match_threshold
    ORDER BY similarity DESC
    LIMIT match_count;
END;
$$;