"""
Generate synthetic Spotify-like audio feature data.
Distributions mimic real Spotify API distributions across genres and decades.
"""
import pandas as pd
import numpy as np
from pathlib import Path

np.random.seed(42)

GENRE_PROFILES = {
    "Hip-Hop":  {"dance": (0.70, 0.15), "energy": (0.65, 0.15),
                 "speech": (0.18, 0.08), "acoustic": (0.15, 0.10),
                 "valence": (0.50, 0.20), "tempo": (95, 15)},
    "EDM":      {"dance": (0.75, 0.10), "energy": (0.85, 0.10),
                 "speech": (0.05, 0.03), "acoustic": (0.05, 0.05),
                 "valence": (0.60, 0.20), "tempo": (128, 8)},
    "Rock":     {"dance": (0.50, 0.15), "energy": (0.75, 0.15),
                 "speech": (0.05, 0.03), "acoustic": (0.20, 0.15),
                 "valence": (0.45, 0.20), "tempo": (120, 20)},
    "Pop":      {"dance": (0.65, 0.12), "energy": (0.70, 0.15),
                 "speech": (0.06, 0.04), "acoustic": (0.20, 0.15),
                 "valence": (0.60, 0.20), "tempo": (115, 18)},
    "Acoustic": {"dance": (0.40, 0.15), "energy": (0.35, 0.15),
                 "speech": (0.04, 0.03), "acoustic": (0.75, 0.15),
                 "valence": (0.50, 0.20), "tempo": (100, 20)},
    "Jazz":     {"dance": (0.45, 0.15), "energy": (0.40, 0.15),
                 "speech": (0.07, 0.05), "acoustic": (0.55, 0.20),
                 "valence": (0.50, 0.20), "tempo": (110, 25)},
    "R&B":      {"dance": (0.65, 0.10), "energy": (0.55, 0.15),
                 "speech": (0.10, 0.05), "acoustic": (0.25, 0.15),
                 "valence": (0.55, 0.20), "tempo": (100, 15)},
    "Country":  {"dance": (0.55, 0.12), "energy": (0.60, 0.15),
                 "speech": (0.05, 0.03), "acoustic": (0.40, 0.20),
                 "valence": (0.60, 0.18), "tempo": (115, 15)},
}


def _year_weights():
    """Tilt toward recent years."""
    years = np.arange(1970, 2025)
    w = np.linspace(0.3, 3, len(years))
    return w / w.sum()


def generate(n=12000):
    rows = []
    year_probs = _year_weights()
    years = np.arange(1970, 2025)

    for i in range(n):
        genre = np.random.choice(list(GENRE_PROFILES.keys()))
        p = GENRE_PROFILES[genre]
        year = int(np.random.choice(years, p=year_probs))

        danceability = np.clip(np.random.normal(*p["dance"]), 0, 1)
        energy = np.clip(np.random.normal(*p["energy"]), 0, 1)
        # Energy creeps up by decade (loudness war effect)
        energy = np.clip(energy + (year - 1990) / 200, 0, 1)
        speechiness = np.clip(np.random.normal(*p["speech"]), 0, 1)
        acousticness = np.clip(np.random.normal(*p["acoustic"]), 0, 1)
        # Acousticness decreases by decade
        acousticness = np.clip(acousticness - (year - 1990) / 250, 0, 1)
        instrumentalness = np.clip(np.random.beta(0.5, 5), 0, 1)
        liveness = np.clip(np.random.beta(1.5, 8), 0, 1)
        valence = np.clip(np.random.normal(*p["valence"]), 0, 1)
        tempo = np.clip(np.random.normal(*p["tempo"]), 50, 200)
        loudness = np.clip(
            -15 + energy * 10 + (year - 1990) / 10
            + np.random.normal(0, 2), -30, 0
        )

        # Popularity model
        pop = (
            20
            + (year - 1970) * 0.4
            + danceability * 15
            + energy * 10
            + loudness * 0.5
            + valence * 5
            + np.random.normal(0, 12)
        )
        pop = int(np.clip(pop, 0, 100))

        rows.append({
            "track_id": f"sp_{i:06d}",
            "track_name": f"Track {i}",
            "artist": f"Artist {np.random.randint(1, 4000)}",
            "year": year,
            "genre": genre,
            "popularity": pop,
            "danceability": round(danceability, 3),
            "energy": round(energy, 3),
            "speechiness": round(speechiness, 3),
            "acousticness": round(acousticness, 3),
            "instrumentalness": round(instrumentalness, 3),
            "liveness": round(liveness, 3),
            "valence": round(valence, 3),
            "tempo": round(tempo, 1),
            "loudness": round(loudness, 2),
        })

    return pd.DataFrame(rows)


def main():
    df = generate()
    out = Path(__file__).resolve().parents[1] / "data" / "spotify_tracks.csv"
    out.parent.mkdir(exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Generated {len(df):,} tracks -> {out}")


if __name__ == "__main__":
    main()
