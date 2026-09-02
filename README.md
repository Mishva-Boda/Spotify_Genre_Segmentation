# 🎵 Spotify Songs' Genre Segmentation & Music Recommendation

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vinutasalagur017-cmyk/Cardiovascular-Disease-Prediction/blob/main/Spotify_Genre_Segmentation.ipynb)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![Machine Learning](https://img.shields.io/badge/ML-Unsupervised%20Clustering-orange.svg)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-yellow.svg)
![Status](https://img.shields.io/badge/Project-Academic%20Minor%20Project-green.svg)

---

## 📌 Project Overview

Spotify uses advanced machine learning algorithms to analyze the acoustic and auditory properties of songs to comprehend diverse music styles. This project explores the **Spotify Songs Dataset** to perform **unsupervised genre segmentation** and build a **content-based song recommendation engine**.

By applying **K-Means Clustering** and **Principal Component Analysis (PCA)** on multi-dimensional audio characteristics (e.g., *Danceability, Energy, Valence, Acousticness, Tempo, and Loudness*), this project groups tracks with similar auditory profiles together and demonstrates how audio features serve as the foundation for modern music recommendation systems.

---

## 🎯 Key Objectives

1. **Data Preprocessing & Cleaning:** Handle missing metadata, manage duplicate records, extract temporal and duration metrics, and standardize feature distributions.
2. **Exploratory Data Analysis (EDA):** Visualize genre distributions, subgenre patterns, feature distributions, and genre-wise feature comparisons.
3. **Correlation Analysis:** Examine associations among audio properties using Pearson correlation heatmaps.
4. **Unsupervised Clustering:** Determine the optimal number of clusters ($k$) using the **Elbow Method** and **Silhouette Score**, and segment songs via **K-Means**.
5. **Dimensionality Reduction (PCA):** Project 12-dimensional audio space into 2D for cluster visualization.
6. **Genre & Playlist Alignment:** Cross-tabulate discovered clusters against Spotify's curated playlist genres, subgenres, and top playlists.
7. **Content-Based Recommendation:** Build a similarity-based recommendation engine using **Cosine Similarity** with cluster-informed boosting.

---

## 📂 Dataset Description

The project analyzes the Spotify songs dataset consisting of:
* **Initial Records:** 32,833 tracks
* **Cleaned Records:** 32,828 tracks (5 missing metadata rows removed)
* **Total Columns:** 23

### Feature Breakdown

| Category | Columns | Description |
|---|---|---|
| **Metadata & Identifiers** | `track_id`, `track_name`, `track_artist`, `track_popularity`, `track_album_id`, `track_album_name`, `track_album_release_date` | Song titles, artists, album information, and popularity score (0–100) |
| **Playlist Taxonomy** | `playlist_name`, `playlist_id`, `playlist_genre`, `playlist_subgenre` | 6 playlist genres, 24 subgenres, and 449 unique playlists |
| **Acoustic Audio Features** | `danceability`, `energy`, `key`, `loudness`, `mode`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`, `duration_ms` | Extracted numerical audio properties describing timbre, rhythm, and mood |
| **Engineered Features** | `duration_minutes`, `release_year` | Human-interpretable duration and extracted 4-digit release years |

### 6 Major Genres
* **EDM:** 6,043 tracks (18.4%)
* **Rap:** 5,746 tracks (17.5%)
* **Pop:** 5,507 tracks (16.8%)
* **R&B:** 5,431 tracks (16.5%)
* **Latin:** 5,155 tracks (15.7%)
* **Rock:** 4,951 tracks (15.1%)

---

## 🛠️ Technologies & Libraries Used

* **Python 3.9+**
* **Pandas & NumPy:** Data cleaning, matrix computations, and feature engineering
* **Scikit-Learn:** `StandardScaler`, `KMeans`, `PCA`, `silhouette_score`, `cosine_similarity`
* **Matplotlib & Seaborn:** Statistical plotting, heatmaps, boxplots, and PCA scatter plots

---

## 🚀 Machine Learning & Data Science Methodology

```
┌─────────────────────────────────────────────────────────┐
│                   1. Data Preprocessing                 │
│   • Missing values handling    • Feature engineering    │
│   • Metadata separation        • StandardScaler (μ=0,σ=1)│
└───────────────────────────┬─────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────┐
│             2. Exploratory Data Analysis (EDA)          │
│   • Feature distributions      • Boxplots by genre      │
│   • Correlation heatmap        • Popularity analysis    │
└───────────────────────────┬─────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────┐
│             3. Cluster Optimization & Modeling          │
│   • Elbow Method (Inertia)     • Silhouette Analysis    │
│   • K-Means Clustering (k=6)   • 2D PCA Visualization   │
└───────────────────────────┬─────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────┐
│             4. Cluster Interpretation & Profiling       │
│   • Mean feature profiles      • Genre/Playlist Crosstab│
│   • Data-driven cluster descriptive naming              │
└───────────────────────────┬─────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────┐
│             5. Content-Based Recommendation             │
│   • Vectorized Cosine Similarity                        │
│   • Same-cluster coherence boost (+0.05)                │
└─────────────────────────────────────────────────────────┘
```

---

## 📈 Key Experimental Results

### 1. Correlation Insights
* **Energy $\leftrightarrow$ Loudness ($+0.68$):** Strong positive correlation — louder recordings exhibit higher acoustic energy.
* **Energy $\leftrightarrow$ Acousticness ($-0.54$):** Strong negative correlation — acoustic instruments characterize softer, lower-energy tracks.
* **Danceability $\leftrightarrow$ Valence ($+0.33$):** Moderate positive correlation — happier/positive tracks correlate with rhythmic danceability.

### 2. Dimensionality Reduction (PCA)
* **PC1:** $17.94\%$ explained variance
* **PC2:** $12.96\%$ explained variance
* **Total 2D Variance Explained:** **$30.90\%$**

### 3. Discovered Cluster Profiles ($k = 6$)

| Cluster | Songs Count | Dominant Genre (% of Cluster) | Key Acoustic Characteristics | Descriptive Profile |
|:---:|:---:|:---:|:---|:---|
| **Cluster 0** | 7,360 | Latin ($22.5\%$), Pop ($19.5\%$) | Danceability: 0.718, Energy: 0.721, Mode: 0.000 (Minor), Valence: 0.598 | **Danceable & Energetic (Minor Key)** |
| **Cluster 1** | 8,544 | Latin ($23.3\%$), Pop ($19.9\%$) | Danceability: 0.716, Energy: 0.725, Mode: 1.000 (Major), Valence: 0.624 | **Upbeat & Dance-Oriented (Major Key)** |
| **Cluster 2** | 6,400 | Rock ($32.4\%$), EDM ($29.8\%$) | Energy: 0.814, Loudness: -5.24 dB, Tempo: 135.8 BPM, Danceability: 0.503 | **High-Energy, Fast-Tempo Rock & EDM** |
| **Cluster 3** | 4,192 | R&B ($33.0\%$), Pop ($17.1\%$) | Acousticness: 0.548, Energy: 0.413, Loudness: -10.72 dB, Valence: 0.385 | **Acoustic, Mellow & Downtempo** |
| **Cluster 4** | 2,509 | EDM ($58.5\%$) | Instrumentalness: 0.747, Energy: 0.787, Loudness: -6.93 dB | **Electronic & Instrumental Focused** |
| **Cluster 5** | 3,823 | Rap ($54.4\%$) | Speechiness: 0.331, Danceability: 0.715, Energy: 0.657 | **Speech-Heavy & Rhythmic Rap** |

---

## 🎵 Content-Based Recommendation Engine

The recommendation system calculates the **Cosine Similarity** between the query song's scaled audio vector ($\vec{u}$) and all candidate tracks ($\vec{v}_i$):

$$\text{Similarity}(\vec{u}, \vec{v}_i) = \frac{\vec{u} \cdot \vec{v}_i}{\|\vec{u}\|_2 \|\vec{v}_i\|_2}$$

* **Cluster-Informed Boosting:** Candidate songs in the same cluster receive a $+0.05$ bonus to enhance genre and acoustic coherence.
* **Deduplication:** Repeated song titles across playlists are automatically deduplicated.
* **Output:** Returns top-$N$ similar tracks displaying Title, Artist, Genre, Subgenre, Cluster, and Similarity Score.

---

## 💻 How to Run the Project

### Option 1: Run on Google Colab (Recommended)
1. Click the **Open in Colab** badge at the top or open [`Spotify_Genre_Segmentation.ipynb`](Spotify_Genre_Segmentation.ipynb) in Google Colab.
2. Upload the CSV dataset (`spotify_songs.csv` or `upload_1cc29b48-7982-4cdb-a0c8-c647c3cd2aa1.csv`) to Colab's file storage.
3. Click **Runtime** $\rightarrow$ **Run all** (`Ctrl + F9`).

### Option 2: Run Locally (Jupyter Notebook / VS Code)
1. Clone or download this project folder:
   ```powershell
   cd P:\corizo\Spotify
   ```
2. Install required packages:
   ```powershell
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```
3. Launch Jupyter Notebook:
   ```powershell
   jupyter notebook Spotify_Genre_Segmentation.ipynb
   ```

---

## 📁 Repository Structure

```
P:\corizo\Spotify\
├── Spotify_Genre_Segmentation.ipynb        # 📓 Standalone Google Colab / Jupyter Notebook
├── upload_1cc29b48-7982-4cdb-a0c8-c647c3cd2aa1.csv # 📊 Spotify Songs CSV Dataset
├── upload_2003450c-1808-42ba-9795-5176f30b50e1.docx # 📄 Project Requirements Document
└── README.md                               # 📖 Comprehensive Project Documentation
```

---

## 🎓 Academic Conclusions

1. **Acoustic Coherence:** Unsupervised clustering effectively groups songs by auditory properties without relying on manual tagging.
2. **Genre Overlap Realism:** Modern genres often share production attributes (e.g., electronic pop tracks clustering with EDM, melodic rap clustering with R&B).
3. **Recommender Foundation:** Audio feature vector similarity enables a cold-start-free recommendation baseline suitable for real-time music discovery.

---

## 🔮 Future Scope

* Integrating collaborative filtering with user listening histories.
* Experimenting with Gaussian Mixture Models (GMM) and hierarchical clustering.
* Using deep learning autoencoders for non-linear feature embeddings.
