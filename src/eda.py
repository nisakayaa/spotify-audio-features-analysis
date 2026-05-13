"""
Spotify Audio Features – EDA
=============================
Explore relationships between audio features and popularity.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set_style("whitegrid")
sns.set_palette("husl")

AUDIO_FEATURES = [
    "danceability", "energy", "speechiness", "acousticness",
    "instrumentalness", "liveness", "valence", "tempo", "loudness",
]


def load(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df["decade"] = (df["year"] // 10) * 10
    print(f"✅ Loaded {len(df):,} tracks across {df['decade'].nunique()} decades")
    return df


def feature_correlations_with_popularity(df: pd.DataFrame) -> pd.Series:
    """Pearson correlation of each feature with popularity."""
    return df[AUDIO_FEATURES + ["popularity"]].corr()["popularity"].drop("popularity").sort_values(key=abs, ascending=False)


def features_by_decade(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("decade")[AUDIO_FEATURES].mean()


def mood_quadrant(row) -> str:
    if row["energy"] >= 0.5 and row["valence"] >= 0.5:
        return "🔥 Hype"
    if row["energy"] >= 0.5 and row["valence"] < 0.5:
        return "😠 Angry/Tense"
    if row["energy"] < 0.5 and row["valence"] >= 0.5:
        return "🌙 Chill"
    return "💧 Sad"


def add_mood(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["mood"] = df.apply(mood_quadrant, axis=1)
    return df


def plot_feature_distributions(df: pd.DataFrame, save_path: str = None):
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    for ax, feat in zip(axes.flatten(), AUDIO_FEATURES):
        sns.histplot(df[feat], bins=40, ax=ax, color="steelblue")
        ax.set_title(feat.capitalize(), fontweight="bold")
    plt.suptitle("Audio Feature Distributions", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_correlation_heatmap(df: pd.DataFrame, save_path: str = None):
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = df[AUDIO_FEATURES + ["popularity"]].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax)
    ax.set_title("Audio Features vs Popularity – Correlation", fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_mood_quadrants(df: pd.DataFrame, save_path: str = None):
    df = add_mood(df)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.scatterplot(
        data=df.sample(min(3000, len(df))),
        x="valence", y="energy", hue="mood",
        alpha=0.6, s=15, ax=ax,
    )
    ax.axhline(0.5, color="black", linestyle="--", alpha=0.4)
    ax.axvline(0.5, color="black", linestyle="--", alpha=0.4)
    ax.set_title("Mood Map: Valence × Energy", fontweight="bold")
    ax.set_xlabel("Valence (positivity)")
    ax.set_ylabel("Energy")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_evolution(df: pd.DataFrame, save_path: str = None):
    decade_means = features_by_decade(df)
    fig, ax = plt.subplots(figsize=(12, 6))
    features_to_plot = ["danceability", "energy", "valence", "acousticness"]
    for feat in features_to_plot:
        ax.plot(decade_means.index, decade_means[feat], marker="o", label=feat, linewidth=2)
    ax.set_title("How Pop Music Has Evolved", fontweight="bold")
    ax.set_xlabel("Decade")
    ax.set_ylabel("Avg Feature Value")
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def main():
    root = Path(__file__).resolve().parents[1]
    df = load(str(root / "data" / "spotify_tracks.csv"))

    print("\n🔝 FEATURES MOST CORRELATED WITH POPULARITY:")
    print(feature_correlations_with_popularity(df))

    print("\n📅 AVERAGE FEATURES BY DECADE:")
    print(features_by_decade(df))

    df_moods = add_mood(df)
    print("\n🎭 MOOD DISTRIBUTION:")
    print(df_moods["mood"].value_counts())

    images = root / "images"
    images.mkdir(exist_ok=True)
    plot_feature_distributions(df, str(images / "feature_distributions.png"))
    plot_correlation_heatmap(df, str(images / "correlation_heatmap.png"))
    plot_mood_quadrants(df, str(images / "mood_quadrants.png"))
    plot_evolution(df, str(images / "evolution_over_decades.png"))


if __name__ == "__main__":
    main()
