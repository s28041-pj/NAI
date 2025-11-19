#!/usr/bin/env python3
"""
Movie Recommendation System

This script loads user movie ratings from a JSON file, retrieves
missing movie metadata (via OMDb API with caching), builds genre-based
profiles for each user using fuzzy scoring, and generates personalized
recommendations and anti-recommendations.

Rules:
See the README.md file in this repository.
https://github.com/s28041-pj/NAI/blob/main/Zad3/readme.md

Authors:
- Mikołaj Gurgul, Łukasz Aleksandrowicz

Requirements:
    - Python 3.13
    - requests
    - A `ratings_en.json` file with user ratings
    - OMDb API key (can be set in environment under OMDB_API_KEY)

## How to run the code:
1. Install Python (>=3.13)
2. Clone this repository
3. Run the code:
    ```bash
    python NAI_03.py
    ```
"""

import json
import requests
import urllib.parse
import os
import random
from collections import defaultdict
import statistics

# CONFIGURATION
OMDB_API_KEY = os.environ.get("OMDB_API_KEY") or "22949d7"  # Default if env var set

# CACHE HANDLING HELPERS
def load_cache(cache_file="movie_cache.json"):
    """Load movie data cache from a JSON file."""
    try:
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def save_cache(cache, cache_file="movie_cache.json"):
    """Save movie data cache to a JSON file."""
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

# OMDb API FUNCTION
def fetch_movie_info_omdb(title):
    """
    Fetch movie info from OMDb API.
    Returns a dict with title, genre, year, plot, and imdb rating.
    """
    params = {"t": title, "apikey": OMDB_API_KEY, "r": "json", "plot": "short"}
    url = "http://www.omdbapi.com/?" + urllib.parse.urlencode(params)
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("Response") == "True":
                return {
                    "title": title,
                    "genre": data.get("Genre", "Unknown"),
                    "year": data.get("Year", "?"),
                    "plot": data.get("Plot", ""),
                    "imdb": data.get("imdbRating", "?")
                }
    except Exception as e:
        print(f"Error fetching {title}:", e)

    return {"title": title, "genre": "Unknown"}

# USER AND MOVIE DATA HANDLING
def load_ratings(filepath="ratings_en.json"):
    """Load user ratings from JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def extract_genres_for_all_movies(user_data, cache_file="movie_cache.json"):
    """
    Extract all movie titles from user data, and fetch
    corresponding movie metadata from OMDb/cached sources.
    """
    all_movies = set()
    for ratings in user_data.values():
        for entry in ratings:
            all_movies.add(entry["title"])

    movie_cache = load_cache(cache_file)
    movie_info = {}

    for movie in all_movies:
        if movie in movie_cache:
            movie_info[movie] = movie_cache[movie]
        else:
            print(f"Fetching data from OMDb for: {movie}")
            info = fetch_movie_info_omdb(movie)
            movie_info[movie] = info
            movie_cache[movie] = info

    save_cache(movie_cache, cache_file)
    return movie_info

# GENRE PREFERENCES
def build_user_genre_preferences(user_data, movie_info):
    """
    Build a genre preference profile for each user based on their
    ratings and the genres associated with each movie.
    Returns: {username: {genre: average_rating}}
    """
    user_genres = {}
    for user, ratings in user_data.items():
        genre_scores = defaultdict(list)
        for entry in ratings:
            title = entry["title"]
            rating = entry["rating"]
            info = movie_info.get(title)
            if not info:
                continue
            genres = [g.strip() for g in info["genre"].split(",")]
            for g in genres:
                genre_scores[g].append(rating)

        # Compute mean score per genre
        user_genres[user] = {g: statistics.mean(scores)
                             for g, scores in genre_scores.items()}

    return user_genres

def get_fuzzy_preferences_for_new_user(movie_info, num_questions=5):
    """
    Interactive mode: If the user is new, ask them to rate a few
    randomly selected movies in order to estimate genre preferences.
    """
    sample_titles = random.sample(list(movie_info.keys()), min(num_questions, len(movie_info)))
    print("\nUser not found. Let's generate your movie taste profile!")
    print("Rate the following movies from 1 to 10 (or leave empty if unknown):\n")

    genre_scores = defaultdict(list)

    for title in sample_titles:
        info = movie_info[title]
        print(f"{title} ({info.get('genre')})")
        score = input("   Your rating (1-10): ")
        try:
            score = float(score)
            if 1 <= score <= 10:
                genres = [g.strip() for g in info['genre'].split(",")]
                for g in genres:
                    genre_scores[g].append(score)
        except:
            print(" Skipped (no rating or invalid input).")

    preferences = {g: statistics.mean(scores) for g, scores in genre_scores.items()}
    print("\nYour genre profile has been generated.")
    return preferences

# RECOMMENDATION ENGINE
def recommend_for_user(target_user, user_data, movie_info, user_genres, top_n=5):
    """
    Generate movie recommendations and anti-recommendations for the given user:
    - Recommendations: Highly rated movies matching liked genres.
    - Anti-recommendations: Poorly rated or disliked genre films.
    """
    if target_user not in user_genres:
        raise ValueError(f"Preferences UNDEFINED for user: {target_user}")

    seen = {entry["title"] for entry in user_data.get(target_user, [])}
    target_genres = user_genres[target_user]

    liked_genres = {g for g, s in target_genres.items() if s >= 7}
    disliked_genres = {g for g, s in target_genres.items() if s <= 4}

    # RECOMMENDATIONS
    candidate_movies = []
    for other, ratings in user_data.items():
        if other == target_user:
            continue
        for entry in ratings:
            title = entry["title"]
            score = entry["rating"]
            if title not in seen:
                genres = [g.strip() for g in movie_info.get(title, {}).get("genre", "Unknown").split(",")]
                if any(g in liked_genres for g in genres) and score >= 7:
                    candidate_movies.append((title, score))

    candidate_movies = sorted(candidate_movies, key=lambda x: x[1], reverse=True)[:top_n]

    # ANTI-RECOMMENDATIONS
    bad_candidates = []
    for other, ratings in user_data.items():
        if other == target_user:
            continue
        for entry in ratings:
            title = entry["title"]
            score = entry["rating"]
            if title not in seen:
                genres = [g.strip() for g in movie_info.get(title, {}).get("genre", "Unknown").split(",")]
                if any(g in disliked_genres for g in genres) or score <= 4:
                    bad_candidates.append((title, score))

    bad_candidates = sorted(bad_candidates, key=lambda x: x[1])[:top_n]
    return candidate_movies, bad_candidates

# MAIN PROGRAM
if __name__ == "__main__":
    # Load user ratings
    user_data = load_ratings("ratings_en.json")

    # Fetch movie metadata from OMDb (with cache support)
    print("Fetching movie metadata from OMDb (or cache)...")
    movie_info = extract_genres_for_all_movies(user_data)

    # Build genre-based preference profiles for existing users
    user_genres = build_user_genre_preferences(user_data, movie_info)

    # Ask for username
    user_name = input("Enter username: ").strip()

    # If the user isn't found, generate a new fuzzy profile
    if user_name not in user_genres:
        print(f"\nUser '{user_name}' not found — creating new profile.")
        user_genres[user_name] = get_fuzzy_preferences_for_new_user(movie_info)
        user_data[user_name] = []  # Initialize as empty list (no seen movies yet)

    # Generate recommendations and anti-recommendations
    recs, anti = recommend_for_user(user_name, user_data, movie_info, user_genres, top_n=5)

    # Print results
    print(f"\nRecommendations for {user_name}:")
    for title, score in recs:
        info = movie_info.get(title)
        print(f" - {title} ({info.get('genre')}) | IMDb: {info.get('imdb')} | {info.get('plot')[:80]}...")

    print(f"\nAnti-recommendations for {user_name}:")
    for title, score in anti:
        info = movie_info.get(title)
        print(f" - {title} ({info.get('genre')}) | IMDb: {info.get('imdb')}")
