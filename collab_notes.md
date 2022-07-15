# Collaborative Filtering Research Notes

## Methods Evaluated
1. User-Based CF - requires user rating data (not available in TMDB dataset)
2. Item-Based CF - can work with implicit feedback
3. Matrix Factorization (SVD) - evaluated using surprise library

## Decision
Proceeding with Content-Based + Popularity-based hybrid approach
Rationale: No explicit user ratings available in dataset

## References
Koren, Bell, Volinsky (2009). Matrix Factorization for Recommender Systems.
TMDB API Docs: https://developers.themoviedb.org/3
