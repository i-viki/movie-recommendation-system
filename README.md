# 🎬 Movie Recommendation System

A content-based movie recommendation system built using **Python** and **Streamlit**.  
The system recommends movies based on metadata similarity and provides analytical insights through clustering and visualization.

This project focuses on **clear separation between training and serving**, **explainable ML**, and **clean engineering practices**.

---

## ✨ Features

- 🎥 Content-based movie recommendations using cosine similarity  
- 🧠 Offline ML training pipeline  
- 🔍 Fuzzy C-Means clustering for analysis and explainability  
- 📊 PCA-based 2D visualization of movie feature space  
- ⚙️ Admin-controlled model training with live progress tracking  
- 🧪 Clear separation of training, services, and UI layers  

---

## 🏗️ Project Structure

```text
.
├── app.py                     # Streamlit application (serving & UI)
├── app_logging.py             # Logging configuration
├── requirements.txt           # Python dependencies
├── .env.example               # Environment variable template
│
├── assets/                    # Static assets (images)
│
├── data/                      # Trained model artifacts
│   ├── movie_list.pkl
│   ├── similarity.pkl
│   ├── clusters.pkl
│   └── vectors.pkl
│
├── dataset/                   # Raw datasets (TMDB)
│   ├── tmdb_movies.csv
│   └── tmdb_credits.csv
│
├── notebooks/                 # Exploratory notebooks
│   └── model_training.ipynb
│
├── services/                  # Reusable business logic
│   ├── recommender.py
│   └── tmdb_client.py
│
├── training/                  # Offline training pipeline
│   ├── train_model.py
│   └── status.json
│
└── tests/                     # Reserved for future tests
````

---

## 🧠 Recommendation Approach

This system uses a **content-based filtering** approach.

Each movie is represented using textual metadata:

* Overview (plot summary)
* Genres
* Keywords
* Cast (top 3 actors)
* Director

These features are combined and vectorized using **CountVectorizer**.
Movie similarity is computed using **cosine similarity**, and the most similar movies are recommended.

### Why Content-Based?

* No user history required
* Deterministic and explainable results
* Works well for cold-start scenarios
* Easy to debug and extend

---

## 📊 Clustering & Visualization

During training, the system performs **Fuzzy C-Means clustering** on movie feature vectors.

* Each movie belongs to multiple clusters with varying membership strength
* Clustering is used **only for analysis and explainability**
* It does **not** influence recommendation ranking

To visualize the high-dimensional feature space, **PCA (Principal Component Analysis)** is used to project movie vectors into 2D.

---

## ⚙️ Model Training

Training is performed **offline** using a dedicated script:

```bash
python training/train_model.py
```

### Training steps include:

* Data preprocessing
* Feature engineering
* Vectorization
* Clustering
* Similarity computation
* Artifact persistence

The Streamlit application loads **only pre-trained artifacts** and does not retrain models during normal usage.

An admin-only UI allows triggering training and monitoring progress live.

---

## 🔐 Environment Variables

This project uses environment variables for sensitive configuration.

### Required variable

```env
TMDB_API_KEY=your_tmdb_api_key_here
```

### Steps

1. Copy `.env.example` to `.env`
2. Add your API key
3. Ensure `.env` is **not committed**

---

## 🚀 Running the Application

### 1️⃣ Create virtual environment

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run Streamlit app

```bash
streamlit run app.py
```

---

## 📦 Dependencies

Key libraries used:

* Streamlit
* Pandas / NumPy
* scikit-learn
* scikit-fuzzy
* SciPy
* python-dotenv

Exact versions are pinned in `requirements.txt`.

---

## ⚠️ Limitations

* No collaborative filtering
* No user behavior learning
* Recommendations depend on metadata quality
* PCA is a lossy projection used only for visualization
* Not intended for production-scale deployment

---

## 📌 Notes

* Exploratory notebooks are kept separate from production code
* Authentication and persistent user data are intentionally excluded
* The project prioritizes clarity, correctness, and explainability

---

## 📜 License

This project is for learning and demonstration purposes.
