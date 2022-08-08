# TMDB API Integration Notes

## Endpoint Used
GET https://api.themoviedb.org/3/movie/{movie_id}?api_key={key}

## Poster Path
Base URL: https://image.tmdb.org/t/p/w500
Append poster_path from API response

## Rate Limiting
40 requests per 10 seconds on free tier
Added retry logic with exponential backoff

## Testing
Tested with movie_id 27205 (Inception) - returned correct poster
Fallback: show placeholder image if API call fails
