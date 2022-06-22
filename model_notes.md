# Content-Based Filtering Model Notes

## Approach
Given a movie, find top-N similar movies using cosine similarity on tag vectors.

## Core Function Logic
1. Get movie index from dataframe
2. Fetch cosine distances from precomputed similarity matrix
3. Sort by distance descending, return top 5 (excluding itself)

## Test Results
Tested with: The Dark Knight, Inception, Avatar
Recommendations appeared thematically consistent

## Next Steps
Integrate TMDB API for movie poster display
Build Streamlit UI for the web app
