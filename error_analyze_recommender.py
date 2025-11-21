"""
Combined script for unsupervised analysis of album recommendations.
Defines prompts, generates recommendations, and runs all analysis in one file.
"""

import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from album_recommender_model import EnhancedRecommender

# --------- Define Prompts ---------
prompts = [
    "upbeat and energetic",
    "dark and moody",
    "experimental jazz",
    "classic rock",
    "ambient electronic",
    "folk storytelling",
    "dance party",
    "chill study music",
    "melancholic indie",
    "instrumental piano"
]

# --------- Generate Recommendations ---------
recommender = EnhancedRecommender()
if not recommender.load_models():
    recommender.build_models()

data = []
for prompt in prompts:
    recs = recommender.recommend_diverse(prompt, top_n=5)
    for r in recs:
        if 'artist' not in r:
            r['artist'] = r.get('author', 'Unknown')
    data.append({
        'prompt': prompt,
        'recommended_albums': recs
    })
df = pd.DataFrame(data)

# --------- Analysis Functions ---------


def analyze_recommendation_diversity(df, k=5):
    genre_counts, album_counts = [], []
    genre_sets, album_sets = [], []
    for _, row in df.iterrows():
        recs = row['recommended_albums'][:k]
        if recs and isinstance(recs[0], dict):
            genres = [r.get('genre', None) for r in recs if r.get('genre')]
            albums = [r.get('album', None) for r in recs if r.get('album')]
        else:
            genres, albums = [], recs
        genre_set = set(genres)
        album_set = set(albums)
        genre_counts.append(len(genre_set))
        album_counts.append(len(album_set))
        genre_sets.append(genre_set)
        album_sets.append(album_set)

    print(
        f"Mean unique genres per prompt: {pd.Series(genre_counts).mean():.2f}")
    print(
        f"Mean unique albums per prompt: {pd.Series(album_counts).mean():.2f}")
    print(
        f"Median unique genres per prompt: {pd.Series(genre_counts).median():.2f}")
    print(
        f"Median unique albums per prompt: {pd.Series(album_counts).median():.2f}")
    print(
        f"Min/Max unique genres per prompt: {min(genre_counts)}/{max(genre_counts)}")
    print(
        f"Min/Max unique albums per prompt: {min(album_counts)}/{max(album_counts)}")

    # Inter-prompt diversity: how much overlap in albums/genres between prompts?
    if len(album_sets) > 1:
        overlap_counts = []
        for i in range(len(album_sets)):
            for j in range(i+1, len(album_sets)):
                overlap = len(album_sets[i] & album_sets[j])
                overlap_counts.append(overlap)


def analyze_recommendation_overlap(df, k=5):
    all_recs = []
    for _, row in df.iterrows():
        recs = row['recommended_albums'][:k]
        if recs and isinstance(recs[0], dict):
            albums = [r.get('album', None) for r in recs if r.get('album')]
        else:
            albums = recs
        all_recs.extend(albums)
    counter = Counter(all_recs)
    print("Most common recommended albums:")
    for album, count in counter.most_common(10):
        print(f"{album}: {count} times")
    plt.figure(figsize=(10, 4))
    pd.Series(counter).value_counts().sort_index().plot(kind='bar')
    plt.title('Frequency of Album Recommendations in Top-K')
    plt.xlabel('Times recommended in top-K')
    plt.ylabel('Number of albums')
    plt.show()


def plot_recommendation_feature_distribution(df, feature='genre', k=5):
    all_features = []
    for _, row in df.iterrows():
        recs = row['recommended_albums'][:k]
        if recs and isinstance(recs[0], dict):
            feats = [r.get(feature, None) for r in recs if r.get(feature)]
        else:
            feats = []
        all_features.extend(feats)
    counter = Counter(all_features)
    print(f"Most common {feature}s in recommendations:")
    for feat, count in counter.most_common(10):
        print(f"{feat}: {count} times")
    plt.figure(figsize=(10, 4))
    pd.Series(counter).head(20).plot(kind='bar')
    plt.title(f'Top {feature.title()}s in Recommendations')
    plt.xlabel(feature.title())
    plt.ylabel('Count')
    plt.show()


def analyze_recommendation_bias(df, group_feature='genre', k=5):
    plot_recommendation_feature_distribution(df, feature=group_feature, k=k)


# --------- Run Analyses ---------
analyze_recommendation_diversity(df, k=5)
analyze_recommendation_overlap(df, k=5)
analyze_recommendation_bias(df, group_feature='genre', k=5)

# Print top 20 artists by recommendation count
all_artists = []
for recs in df['recommended_albums']:
    all_artists.extend([r['artist'] for r in recs])
print("\nTop 20 artists by recommendation count:")
for artist, count in Counter(all_artists).most_common(20):
    print(f"{artist}: {count} times")

# Plot artist distribution (top 20)
plot_recommendation_feature_distribution(df, feature='artist', k=5)