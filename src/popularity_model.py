"""
Popularity Prediction Model
============================
Random Forest regression to predict track popularity from audio features.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

sns.set_style("whitegrid")

FEATURES = [
    "danceability", "energy", "speechiness", "acousticness",
    "instrumentalness", "liveness", "valence", "tempo", "loudness", "year",
]


def train(df: pd.DataFrame):
    X = df[FEATURES]
    y = df["popularity"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    rf = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    preds = rf.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    importance = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=False)

    return rf, mae, r2, importance


def plot_importance(importance: pd.Series, save_path: str = None):
    fig, ax = plt.subplots(figsize=(9, 6))
    importance.plot(kind="barh", ax=ax, color="purple")
    ax.invert_yaxis()
    ax.set_title("What Predicts Spotify Popularity?", fontweight="bold")
    ax.set_xlabel("Feature Importance")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def main():
    root = Path(__file__).resolve().parents[1]
    df = pd.read_csv(root / "data" / "spotify_tracks.csv")

    print("🤖 Training Random Forest...")
    model, mae, r2, importance = train(df)

    print(f"\n📊 MAE: {mae:.2f}")
    print(f"📊 R²:  {r2:.3f}")
    print("\n🔝 FEATURE IMPORTANCE:")
    print(importance)

    images = root / "images"
    images.mkdir(exist_ok=True)
    plot_importance(importance, str(images / "feature_importance.png"))


if __name__ == "__main__":
    main()
