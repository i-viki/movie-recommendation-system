from app_logging import get_logger

logger = get_logger(__name__)


def recommend(movie_title, movies_df, similarity_matrix, fetch_poster_fn):
    """
    Generate movie recommendations based on similarity scores.

    Fails safely if input is invalid or data is missing.
    """

    # ---------- Validation ----------
    if movie_title is None or not isinstance(movie_title, str):
        return [], []

    if movies_df is None or similarity_matrix is None:
        return [], []

    if "title" not in movies_df.columns:
        return [], []

    # ---------- Find movie index ----------
    matches = movies_df[movies_df["title"] == movie_title]

    if matches.empty:
        logger.warning("Movie title not found: %s", movie_title)
        return [], []

    index = matches.index[0]

    # ---------- Similarity calculation ----------
    try:
        distances = list(enumerate(similarity_matrix[index]))
    except Exception:
        return [], []

    distances = sorted(distances, reverse=True, key=lambda x: x[1])

    # ---------- Build recommendations ----------
    recommended_movie_names = []
    recommended_movie_posters = []

    for i in distances[1:6]:
        try:
            movie_row = movies_df.iloc[i[0]]
            movie_id = movie_row.movie_id

            recommended_movie_names.append(movie_row.title)
            recommended_movie_posters.append(fetch_poster_fn(movie_id))

        except Exception:
            logger.debug("Skipping broken row during recommendation build", exc_info=e)
            continue

    return recommended_movie_names, recommended_movie_posters