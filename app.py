import pickle
import streamlit as st
import pandas as pd
import subprocess
import sys
from services.recommender import recommend
from services.tmdb_client import fetch_poster
import json
from pathlib import Path
from sklearn.decomposition import PCA
import numpy as np
import time



STAGE_PROGRESS = {
    "STARTED": 0.05,
    "DATA_LOADING": 0.15,
    "MERGING": 0.25,
    "COLUMN_SELECTION": 0.30,
    "CLEANING": 0.35,
    "PARSING": 0.45,
    "NORMALIZATION": 0.55,
    "FEATURE_BUILDING": 0.65,
    "VECTORIZATION": 0.75,
    "CLUSTERING": 0.82,
    "SIMILARITY": 0.85,
    "SAVING": 0.95,
    "COMPLETED": 1.0,
    "FAILED": 1.0,
}


STATUS_FILE = Path("training/status.json")

def read_training_status():
    if not STATUS_FILE.exists():
        return None
    try:
        with open(STATUS_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return None


# --------------------
# Sidebar Navigation
# --------------------
st.sidebar.title("🎬 Movie Recommender")
if "page" not in st.session_state:
    st.session_state.page = "Recommend"
page = st.sidebar.radio(
    "Navigate",
    [
        "Recommend",
        "Model Info",
        "Data (Admin)",
        "Training (Admin)"
    ],
    index=[
        "Recommend",
        "Model Info",
        "Data (Admin)",
        "Training (Admin)"
    ].index(st.session_state.page)
)
st.session_state.page = page




# --------------------
# Load Data & Models
# --------------------
clusters = pickle.load(open("data/clusters.pkl", "rb"))
movies = pickle.load(open("data/movie_list.pkl", "rb"))
similarity = pickle.load(open("data/similarity.pkl", "rb"))
vectors = pickle.load(open("data/vectors.pkl", "rb"))


movies["cluster"] = clusters["labels"]
credits = pd.read_csv("dataset/tmdb_credits.csv")


# --------------------
# UI Configuration
# --------------------
st.set_page_config(
    page_title="Movie Recommender System",
    layout="wide"
)

st.title("🎬 Movie Recommender System")

st.write(
    "A content-based movie recommendation system that suggests similar movies "
    "based on user-selected preferences."
)


# --------------------
# Movie Selection
# --------------------
if page == "Recommend":
    st.header("Get Movie Recommendations")

    st.write(
        "Select a movie you like and the system will recommend similar movies "
        "based on content similarity."
    )

    movie_list = movies["title"].values
    selected_movie = st.selectbox(
        "Select a movie",
        movie_list
    )

    if st.button("Recommend Movies"):
        recommended_names, recommended_posters = recommend(
            selected_movie,
            movies,
            similarity,
            fetch_poster
        )

        if not recommended_names:
            st.warning(
                "No recommendations could be generated at the moment. "
                "Please try a different movie."
            )
        else:
            cols = st.columns(len(recommended_names))
            for i in range(len(recommended_names)):
                with cols[i]:
                    st.subheader(recommended_names[i])
                    if recommended_posters[i]:
                        st.image(recommended_posters[i])
                    else:
                        st.caption("Poster unavailable")
elif page == "Model Info":
    st.header("Model Information")

    # -------------------------------------------------
    # Recommendation explanation
    # -------------------------------------------------
    st.markdown("""
    ### Recommendation Approach

    This system uses a **content-based recommendation approach**.

    Movies are represented using textual features derived from:
    - Overview (plot summary)
    - Genres
    - Keywords
    - Cast (top 3 actors)
    - Director

    These features are combined into a single text representation and
    vectorized using **CountVectorizer**.
    Movie similarity is computed using **cosine similarity**.
    """)

    st.markdown("""
    ### Why Content-Based?

    - No user history required
    - Deterministic and explainable recommendations
    - Handles cold-start scenarios well
    - Easy to debug and reason about
    """)

    # -------------------------------------------------
    # Clustering section (Fuzzy C-Means)
    # -------------------------------------------------
    st.subheader("Clustering Analysis (Fuzzy C-Means)")

    st.markdown("""
    During training, movies are grouped using **Fuzzy C-Means clustering**
    based on the same vectorized metadata features.

    Unlike hard clustering, Fuzzy C-Means assigns **degrees of membership**
    to each movie across multiple clusters.
    For visualization and exploration, the **dominant cluster assignment**
    is shown.
    """)

    cluster_counts = movies["cluster"].value_counts().sort_index()
    st.bar_chart(cluster_counts)

    selected_cluster = st.selectbox(
        "Explore movies in a cluster",
        sorted(movies["cluster"].unique())
    )

    cluster_movies = (
        movies[movies["cluster"] == selected_cluster]["title"]
        .head(20)
    )

    st.write(cluster_movies)

    # -------------------------------------------------
    # PCA Visualization
    # -------------------------------------------------
    st.subheader("2D PCA Visualization")

    st.markdown("""
    This visualization projects the high-dimensional movie feature vectors
    into a **2D space using Principal Component Analysis (PCA)**.

    Each point represents a movie and is colored by its dominant fuzzy cluster.
    PCA is used **only for visualization and interpretability**, not for
    recommendation or clustering itself.
    """)

    pca = PCA(n_components=2, random_state=42)
    pca_result = pca.fit_transform(vectors)

    pca_df = pd.DataFrame(
        {
            "PC1": pca_result[:, 0],
            "PC2": pca_result[:, 1],
            "Cluster": movies["cluster"]
        }
    )

    st.scatter_chart(
        pca_df,
        x="PC1",
        y="PC2",
        color="Cluster"
    )

    st.caption(
        f"PCA explained variance: "
        f"{pca.explained_variance_ratio_[0]:.2%} (PC1), "
        f"{pca.explained_variance_ratio_[1]:.2%} (PC2)"
    )

    # -------------------------------------------------
    # Model lifecycle & limitations
    # -------------------------------------------------
    st.markdown("""
    ### Model Lifecycle

    - Training is performed **offline** using a dedicated training script
    - Vectorization, clustering, and similarity computation happen during training
    - Model artifacts are persisted as pickle files
    - The Streamlit application **only loads and serves** trained artifacts
    - Retraining can be triggered through an admin-only action
    """)

    st.markdown("""
    ### Limitations

    - No collaborative filtering or user behavior learning
    - Recommendations depend entirely on metadata quality
    - Clustering is used for **analysis and explainability**, not ranking
    - PCA visualization is an approximation and not a lossless projection
    """)




# --------------------
# Dataset & Info Section
# --------------------
elif page == "Data (Admin)":
    st.header("Data Exploration (Admin)")

    st.info(
        "This section is intended for development and analysis purposes only. "
        "It is not part of the end-user recommendation experience."
    )

    with st.expander("Movies Dataset Preview"):
        st.dataframe(movies.head(20))

    with st.expander("Movies Dataset Shape"):
        st.write(movies.shape)

    with st.expander("Credits Dataset Preview"):
        st.dataframe(credits.head(10))

elif page == "Training (Admin)":
    st.header("Model Training (Admin)")

    # Read current training status
    status = read_training_status()


    if status:
        stage = status.get("stage", "UNKNOWN")
        message = status.get("message", "")
        timestamp = status.get("timestamp", "")

        progress = STAGE_PROGRESS.get(stage, 0.0)

        st.subheader("Training Progress")
        st.progress(progress)

        if stage == "COMPLETED":
            st.success("Training completed successfully.")
        elif stage == "FAILED":
            st.error("Training failed. Check logs for details.")
        else:
            st.info(
            f"**Stage:** {stage}\n\n"
            f"**Message:** {message}\n\n"
            f"**Last update:** {timestamp}"
        )

    st.warning(
        "⚠️ Training is a resource-intensive operation.\n\n"
        "- Training runs in a background process\n"
        "- The application will continue using the current model\n"
        "- Restart the app after training completes to load the new model\n\n"
        "This action is intended for administrators only."
    )

    if st.button("Train Model"):
        with st.spinner("Starting training in background..."):
            subprocess.Popen(
                [sys.executable, "training/train_model.py"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

        st.success(
            "Training process started successfully.\n\n"
            "Please restart the application after completion "
            "to use the updated model."
        )



