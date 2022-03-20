# Exploratory Data Analysis Notes

## Dataset Shape
tmdb_5000_movies.csv: 4803 rows x 20 columns
tmdb_5000_credits.csv: 4803 rows x 4 columns

## Key Observations
Missing values in: homepage, tagline, runtime
genres, keywords, cast, crew stored as JSON strings -> need parsing
Top genres: Drama, Comedy, Thriller, Action, Romance

## Preprocessing Plan
Parse JSON columns using ast.literal_eval
Extract director from crew
Merge both datasets on title
Create unified tags column for content-based filtering
