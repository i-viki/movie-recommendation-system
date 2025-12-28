import os
import ast
import pickle
import skfuzzy as fuzz
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import json
from datetime import datetime

STATUS_FILE = "training/status.json"

def update_status(stage, message):
    status = {
        "stage": stage,
        "message": message,
        "timestamp": datetime.utcnow().isoformat()
    }
    with open(STATUS_FILE, "w") as f:
        json.dump(status, f)


# ------------------------------
# Paths
# ------------------------------
DATASET_DIR = "dataset"
MOVIES_CSV = os.path.join(DATASET_DIR, "tmdb_movies.csv")
CREDITS_CSV = os.path.join(DATASET_DIR, "tmdb_credits.csv")

OUTPUT_DIR = "data"
MOVIE_LIST_PATH = os.path.join(OUTPUT_DIR, "movie_list.pkl")
SIMILARITY_PATH = os.path.join(OUTPUT_DIR, "similarity.pkl")


# ------------------------------
# Helper functions
# ------------------------------
def convert(text):
    return [i["name"] for i in ast.literal_eval(text)]


def convert_top_3(text):
    return [i["name"] for i in ast.literal_eval(text)[:3]]


def fetch_director(text):
    for i in ast.literal_eval(text):
        if i["job"] == "Director":
            return [i["name"]]
    return []


def collapse(words):
    return [i.replace(" ", "") for i in words]


# ------------------------------
# Training pipeline
# ------------------------------
def train():
    try:
        update_status("STARTED", "Training started")

        update_status("DATA_LOADING", "Loading datasets")
        movies = pd.read_csv(MOVIES_CSV)
        credits = pd.read_csv(CREDITS_CSV)

        update_status("MERGING", "Merging movie and credits data")
        movies = movies.merge(credits, on="title")

        update_status("COLUMN_SELECTION", "Selecting relevant columns")
        movies = movies[
            ["movie_id", "title", "overview", "genres", "keywords", "cast", "crew"]
        ]

        update_status("CLEANING", "Dropping missing values")
        movies.dropna(inplace=True)

        update_status("PARSING", "Parsing JSON-like columns")
        movies["genres"] = movies["genres"].apply(convert)
        movies["keywords"] = movies["keywords"].apply(convert)
        movies["cast"] = movies["cast"].apply(convert_top_3)
        movies["crew"] = movies["crew"].apply(fetch_director)

        update_status("NORMALIZATION", "Normalizing text features")
        movies["overview"] = movies["overview"].apply(lambda x: x.split())

        movies["genres"] = movies["genres"].apply(collapse)
        movies["keywords"] = movies["keywords"].apply(collapse)
        movies["cast"] = movies["cast"].apply(collapse)
        movies["crew"] = movies["crew"].apply(collapse)

        update_status("FEATURE_BUILDING", "Building combined tags")
        movies["tags"] = (
            movies["overview"]
            + movies["genres"]
            + movies["keywords"]
            + movies["cast"]
            + movies["crew"]
        )

        new_df = movies[["movie_id", "title", "tags"]]
        new_df["tags"] = new_df["tags"].apply(lambda x: " ".join(x))

        update_status("VECTORIZATION", "Vectorizing text features")
        cv = CountVectorizer(max_features=5000, stop_words="english")
        vectors = cv.fit_transform(new_df["tags"]).toarray()
        VECTORS_PATH = os.path.join(OUTPUT_DIR, "vectors.pkl")
        with open(VECTORS_PATH, "wb") as f:
            pickle.dump(vectors, f)
            
        update_status("CLUSTERING", "Performing fuzzy c-means clustering")

        N_CLUSTERS = 8  # reasonable default

        # scikit-fuzzy expects: features x samples
        data = vectors.T

        cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
            data=data,
            c=N_CLUSTERS,
            m=2.0,
            error=0.005,
            maxiter=1000,
            init=None
        )

        # Dominant cluster for each movie
        cluster_labels = u.argmax(axis=0)

        # Attach cluster info to dataframe
        new_df["cluster"] = cluster_labels

        

        update_status("SIMILARITY", "Computing cosine similarity")
        similarity = cosine_similarity(vectors)

        update_status("SAVING", "Saving model artifacts")
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        with open(MOVIE_LIST_PATH, "wb") as f:
            pickle.dump(new_df, f)

        with open(SIMILARITY_PATH, "wb") as f:
            pickle.dump(similarity, f)
        
        CLUSTER_PATH = os.path.join(OUTPUT_DIR, "clusters.pkl")

        with open(CLUSTER_PATH, "wb") as f:
            pickle.dump(
                {
                    "centers": cntr,
                    "membership": u,
                    "labels": cluster_labels
                },
                f
            )


        update_status("COMPLETED", "Training completed successfully")

    except Exception as e:
        update_status("FAILED", f"Training failed: {str(e)}")
        raise


if __name__ == "__main__":
    train()
