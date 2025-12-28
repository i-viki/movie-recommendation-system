import os
import requests
from dotenv import load_dotenv
from app_logging import get_logger

logger = get_logger(__name__)

load_dotenv()

TMDB_API_KEY = os.getenv("TMDB_API_KEY")
TMDB_BASE_URL = os.getenv("TMDB_BASE_URL", "https://api.themoviedb.org/3")
TMDB_IMAGE_BASE_URL = os.getenv(
    "TMDB_IMAGE_BASE_URL",
    "https://image.tmdb.org/t/p/w500"
)


def fetch_poster(movie_id):
    """
    Fetch movie poster URL from TMDB.
    Fails gracefully if network / SSL issues occur.
    """

    # Safety: no API key, no call
    if not TMDB_API_KEY:
        return None

    url = (
        f"{TMDB_BASE_URL}/movie/{movie_id}"
        f"?api_key={TMDB_API_KEY}&language=en-US"
    )

    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()

        poster_path = data.get("poster_path")
        if not poster_path:
            return None

        return TMDB_IMAGE_BASE_URL + poster_path

    except requests.exceptions.RequestException:
        logger.warning("TMDB request failed for movie_id=%s", movie_id)
        return None
