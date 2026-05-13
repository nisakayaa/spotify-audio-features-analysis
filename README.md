# 🎧 Spotify Audio Features – What Makes a Hit Song?

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-orange.svg)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Featured-purple.svg)]()

> **⭐ Featured Niche Project** – A data-driven look at the audio DNA of hit songs.

## 🎯 Project Overview

Why do some songs blow up while others don't? Using Spotify's audio feature API data (danceability, energy, valence, tempo, etc.), this project explores the **sonic signature of popular tracks** across genres and decades.

**Decode the recipe of a hit.** This is not a typical sales/EDA dataset — it's a music analytics project blending data science with art.

## 🔍 Research Questions

1. **Is there a "hit formula"?** Which audio features correlate most with popularity?
2. **Has popular music sound changed over decades?** Quantify the evolution.
3. **Genre fingerprints**: Can we separate hip-hop, rock, and EDM tracks purely by audio features?
4. **Mood map**: Plot songs in a valence×energy space — sad/calm vs happy/hype quadrants.
5. **Can we predict popularity?** Build a regression model and inspect its top features.

## 🎼 The Features Explained

| Feature | What It Measures | Range |
|---------|------------------|-------|
| 💃 Danceability | How suitable for dancing | 0–1 |
| ⚡ Energy | Intensity & activity | 0–1 |
| 🎤 Speechiness | Presence of spoken words | 0–1 |
| 🎻 Acousticness | Acoustic confidence | 0–1 |
| 🎹 Instrumentalness | No vocals likelihood | 0–1 |
| 🎺 Liveness | Audience presence | 0–1 |
| 😊 Valence | Musical positiveness | 0–1 |
| 🥁 Tempo | BPM | 50–200 |
| 🔊 Loudness | Overall dB | -60 to 0 |

## 📁 Project Structure

```
spotify-audio-features-analysis/
├── data/
│   └── spotify_tracks.csv
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_genre_clustering.ipynb
│   └── 03_popularity_model.ipynb
├── src/
│   ├── eda.py
│   ├── clustering.py
│   ├── popularity_model.py
│   └── generate_data.py
├── outputs/
├── images/
├── requirements.txt
└── README.md
```

## 🛠️ Tech Stack

| Layer | Tools |
|-------|-------|
| **Data** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn, Plotly |
| **ML** | Scikit-learn (KMeans, RandomForest), UMAP |
| **Notebooks** | Jupyter Lab |


## 📈 Key Insights

### 🎯 The "Hit Formula" 
Top predictors of popularity:
1. **Loudness** (newer hits are louder)
2. **Energy**
3. **Danceability**
4. **Year of release** (recency matters)

### 📅 Sound Evolution
- 🎵 **2000s**: Higher acousticness, lower energy
- 🎵 **2010s**: Energy spike, loudness war intensifies
- 🎵 **2020s**: Tempo slows, valence drops slightly ("sad hits")

### 🎨 Genre Clusters
K-Means (k=5) on audio features cleanly separates:
- Hip-Hop (high speechiness, low acousticness)
- EDM (high energy, high danceability)
- Acoustic/Folk (high acousticness, low energy)
- Rock (high energy + loudness, mid valence)
- Pop (balanced across features)

### 😊 Mood Quadrants
| Quadrant | Energy | Valence | Vibe |
|----------|--------|---------|------|
| 🔥 Hype | High | High | Party, anthems |
| 😢 Angry/Tense | High | Low | Aggressive |
| 🌙 Chill | Low | High | Acoustic, mellow |
| 💧 Sad | Low | Low | Ballads, ambient |

## 🧠 Methodology

1. **Data Collection** – Spotify Web API audio features (synthetic stand-in here)
2. **Feature Engineering** – Decade buckets, mood quadrants
3. **EDA** – Distributions, correlations, time-series
4. **Unsupervised Learning** – K-Means clustering, UMAP visualization
5. **Supervised Modeling** – Random Forest regression on popularity
6. **Interpretation** – SHAP-style feature importance

## 📊 Sample Visualizations

The `images/` directory contains:
- `feature_distributions.png` – Audio feature histograms
- `correlation_heatmap.png` – Inter-feature correlations
- `evolution_over_decades.png` – Time series of key features
- `genre_clusters_umap.png` – 2D projection of clusters
- `mood_quadrants.png` – Valence × Energy scatter
- `feature_importance.png` – Top predictors of popularity

## 🔗 Real-World Extension

This project is built to plug into the actual **Spotify Web API**. Replace `data/spotify_tracks.csv` with real pulls via `spotipy`:

```python
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials())
features = sp.audio_features(track_ids)
```

## 📝 License

[MIT](LICENSE)

## 👤 Author

**Your Name**
- GitHub: nisakayaa

---

⭐ If you found this useful, star the repo!
