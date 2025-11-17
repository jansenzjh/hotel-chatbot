-- 1. Enable the pgvector extension
create extension if not exists vector with schema public;

-- 2. Create a table to store your listings
create table if not exists listings_1024 (
  id bigint primary key,
  listing_url text,
  last_scraped text,
  source text,
  name text,
  description text,
  neighborhood_overview text,
  host_url text,
  host_name text,
  host_since text,
  host_location text,
  host_about text,
  host_response_time text,
  host_response_rate text,
  host_acceptance_rate text,
  host_is_superhost text,
  host_neighbourhood text,
  host_listings_count integer,
  host_total_listings_count integer,
  host_verifications text,
  host_has_profile_pic text,
  host_identity_verified text,
  neighbourhood text,
  neighbourhood_cleansed text,
  neighbourhood_group_cleansed text,
  latitude float,
  longitude float,
  property_type text,
  room_type text,
  accommodates integer,
  bathrooms float,
  bathrooms_text text,
  bedrooms float,
  beds float,
  amenities text,
  price_cleaned float,
  minimum_nights integer,
  maximum_nights integer,
  minimum_minimum_nights integer,
  maximum_minimum_nights integer,
  minimum_maximum_nights integer,
  maximum_maximum_nights integer,
  minimum_nights_avg_ntm float,
  maximum_nights_avg_ntm float,
  calendar_updated text,
  has_availability text,
  availability_30 integer,
  availability_60 integer,
  availability_90 integer,
  availability_365 integer,
  calendar_last_scraped text,
  number_of_reviews integer,
  number_of_reviews_ltm integer,
  number_of_reviews_l30d integer,
  review_scores_rating float,
  review_scores_accuracy float,
  review_scores_cleanliness float,
  review_scores_checkin float,
  review_scores_communication float,
  review_scores_location float,
  review_scores_value float,
  license text,
  instant_bookable text,
  calculated_host_listings_count integer,
  calculated_host_listings_count_entire_homes integer,
  calculated_host_listings_count_private_rooms integer,
  calculated_host_listings_count_shared_rooms integer,
  reviews_per_month float,
  rag_document text,
  embedding vector(1024)
);
-- 3. Create a function to search for listings
create or replace function match_properties_1024 (
  query_embedding vector(1024),
  match_threshold float,
  match_count int
)
returns table (
  id bigint,
  name text,
  listing_url text,
  latitude float,
  longitude float,
  rag_document text,
  price_cleaned float,
  similarity float
)
language sql stable
as $$
  select
    listings_1024.id,
    listings_1024.name,
    listings_1024.listing_url,
    listings_1024.latitude,
    listings_1024.longitude,
    listings_1024.rag_document,
    listings_1024.price_cleaned,
    1 - (listings_1024.embedding <=> query_embedding) as similarity
  from listings_1024
  where 1 - (listings_1024.embedding <=> query_embedding) > match_threshold
  order by similarity desc
  limit match_count;
$$;