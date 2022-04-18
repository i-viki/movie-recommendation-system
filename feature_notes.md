# Feature Engineering Notes

## Tags Column Construction
Combined: overview, genres, keywords, top 3 cast, director

## Text Normalization
Lowercased all text
Removed spaces within multi-word tags
Applied Porter Stemmer for dimensionality reduction

## Vectorization
Used CountVectorizer with max_features=5000
Stop words: English
Output: (4806, 5000) sparse matrix

## Similarity
Cosine similarity computed on tag vectors
Saved as similarity.pkl (~96MB)
