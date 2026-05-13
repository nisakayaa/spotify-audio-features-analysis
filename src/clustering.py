"""
Genre Clustering with K-Means
==============================
Unsupervised: can audio features alone reveal genre groups?
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

sns.set_style("whitegrid")

CLUSTER_FEATURES = [
    "danceability", "energy", "speechiness", "acousticness",
    "instrumentalness", "valence", "loudness", "tempo",
]


def cluster_tracks(df: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    X = df[CLUSTER_FEATURES].values
    X_scaled = StandardScaler().fit_transform(X)

    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    df = df.copy()
    df["cluster"] = km.fit_predict(X_scaled)

    # PCA for 2D viz
    pca = PCA(n_components=2)
    df[["pc1", "pc2"]] = pca.fit_transform(X_scaled)
    print(f"✅ K-Means complete. Explained variance: {pca.explained_variance_ratio_.sum():.2%}")

    return df


def label_clusters(df: pd.DataFrame) -> dict:
    """Auto-label clusters with their dominant feature."""
    profiles = df.groupby("cluster")[CLUSTER_FEATURES].mean()
    labels = {}
    for c in profiles.index:
        row = profiles.loc[c]
        if row["speechiness"] > 0.15:
            labels[c] = "🎤 Hip-Hop-like"
        elif row["acousticness"] > 0.6:
            labels[c] = "🎻 Acoustic/Folk"
        elif row["energy"] > 0.75 and row["danceability"] > 0.65:
            labels[c] = "🕺 EDM/Dance"
        elif row["energy"] > 0.7:
            labels[c] = "🎸 Rock"
        else:
            labels[c] = "🎵 Pop/Mid"
    return labels


def plot_clusters_pca(df: pd.DataFrame, labels: dict, save_path: str = None):
    df_plot = df.copy()
    df_plot["cluster_label"] = df_plot["cluster"].map(labels)
    fig, ax = plt.subplots(figsize=(11, 8))
    sns.scatterplot(
        data=df_plot.sample(min(3000, len(df_plot))),
        x="pc1", y="pc2", hue="cluster_label", alpha=0.7, s=18, ax=ax,
    )
    ax.set_title("Track Clusters in PCA Space", fontweight="bold")
    ax.legend(title="Cluster", loc="best")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def main():
    root = Path(__file__).resolve().parents[1]
    df = pd.read_csv(root / "data" / "spotify_tracks.csv")

    clustered = cluster_tracks(df, k=5)
    labels = label_clusters(clustered)

    print("\n🏷️  CLUSTER LABELS:")
    for c, l in labels.items():
        print(f"  Cluster {c}: {l}")

    print("\n📊 CLUSTER PROFILES (mean values):")
    print(clustered.groupby("cluster")[CLUSTER_FEATURES].mean().round(2))

    images = root / "images"
    images.mkdir(exist_ok=True)
    plot_clusters_pca(clustered, labels, str(images / "genre_clusters_umap.png"))

    out = root / "outputs" / "clustered_tracks.csv"
    out.parent.mkdir(exist_ok=True)
    clustered.to_csv(out, index=False)
    print(f"\n✅ Saved → {out}")


if __name__ == "__main__":
    main()
