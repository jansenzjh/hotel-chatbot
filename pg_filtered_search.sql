create or replace function match_properties_filtered (
  query_embedding vector(1024),
  match_threshold float,
  match_count int,
  min_price float default null,
  max_price float default null
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
  and (min_price is null or listings_1024.price_cleaned >= min_price)
  and (max_price is null or listings_1024.price_cleaned <= max_price)
  order by similarity desc
  limit match_count;
$$;
