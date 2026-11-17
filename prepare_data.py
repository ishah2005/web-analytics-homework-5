"""Prepare the combined Rotten Tomatoes review dataset for Homework 5.

Pass ``--parser-input`` to clean a CSV produced by collect_reviews.py. Without
that option, the script rebuilds the committed cached snapshot from its public
archive source.
"""

import argparse

from pathlib import Path

import pandas as pd


RANDOM_STATE = 4506
N_MOVIES = 1000
MAX_REVIEWS_PER_MOVIE = 50

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_FILE = DATA_DIR / "rotten_tomatoes_hw5.csv"

# Direct download of the exact CSV used for the submitted analysis.
DATA_DOWNLOAD_URL = (
    "https://raw.githubusercontent.com/ishah2005/web-analytics-homework-5/"
    "main/data/rotten_tomatoes_hw5.csv"
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parser-input", type=Path)
    return parser.parse_args()


def prepare_parser_output(path):
    selected = pd.read_csv(path)
    required = {
        "rotten_tomatoes_link",
        "movie_title",
        "review_type",
        "review_content",
    }
    if not required.issubset(selected.columns):
        raise ValueError(f"Parser output is missing: {sorted(required - set(selected))}")
    selected = selected.dropna(subset=["review_content", "review_type"]).copy()
    selected = selected[selected["review_type"].isin(["Fresh", "Rotten"])]
    selected["review_content"] = selected["review_content"].astype(str).str.strip()
    selected = selected[selected["review_content"].str.len() >= 10]
    selected = selected.drop_duplicates(
        subset=["rotten_tomatoes_link", "review_content"]
    )
    selected["label"] = (selected["review_type"] == "Fresh").astype(int)
    selected["review_date"] = pd.to_datetime(selected.get("review_date"), errors="coerce")
    selected["review_length_words"] = selected["review_content"].str.split().str.len()
    if len(selected) < 10_000 or selected["rotten_tomatoes_link"].nunique() < 500:
        raise ValueError("Parser output must contain at least 10,000 reviews from 500 movies")
    # Preserve the collector's stable row order so fixed train/test and LLM
    # samples remain identical when this cleaned CSV is prepared again.
    return selected.reset_index(drop=True)


def main() -> None:
    args = parse_args()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if args.parser_input:
        selected = prepare_parser_output(args.parser_input)
        selected.to_csv(OUTPUT_FILE, index=False)
        print(f"Saved {len(selected):,} parser-collected reviews to {OUTPUT_FILE}")
        print(f"Distinct movies: {selected['rotten_tomatoes_link'].nunique():,}")
        print(f"Submitted dataset download: {DATA_DOWNLOAD_URL}")
        return

    # Only the cached-snapshot rebuild route needs Kaggle Hub.
    import kagglehub

    dataset_dir = Path(
        kagglehub.dataset_download(
            "stefanoleone992/rotten-tomatoes-movies-and-critic-reviews-dataset"
        )
    )
    reviews_path = dataset_dir / "rotten_tomatoes_critic_reviews.csv"
    movies_path = dataset_dir / "rotten_tomatoes_movies.csv"

    reviews = pd.read_csv(reviews_path)
    movies = pd.read_csv(
        movies_path,
        usecols=[
            "rotten_tomatoes_link",
            "movie_title",
            "genres",
            "original_release_date",
            "tomatometer_rating",
        ],
    )

    reviews = reviews[
        [
            "rotten_tomatoes_link",
            "critic_name",
            "top_critic",
            "publisher_name",
            "review_type",
            "review_score",
            "review_date",
            "review_content",
        ]
    ].copy()
    reviews = reviews.dropna(subset=["review_content", "review_type"])
    reviews = reviews[reviews["review_type"].isin(["Fresh", "Rotten"])]
    reviews["review_content"] = reviews["review_content"].astype(str).str.strip()
    reviews = reviews[reviews["review_content"].str.len() >= 10]
    reviews = reviews.drop_duplicates(
        subset=["rotten_tomatoes_link", "review_content"]
    )

    movie_counts = reviews.groupby("rotten_tomatoes_link").size()
    eligible_movies = movie_counts[
        movie_counts >= MAX_REVIEWS_PER_MOVIE
    ].index.to_series()
    selected_movies = eligible_movies.sample(
        n=N_MOVIES, random_state=RANDOM_STATE
    ).tolist()

    selected = reviews[reviews["rotten_tomatoes_link"].isin(selected_movies)].copy()
    selected = (
        selected.groupby("rotten_tomatoes_link", group_keys=False)
        .sample(n=MAX_REVIEWS_PER_MOVIE, random_state=RANDOM_STATE)
        .reset_index(drop=True)
    )

    selected = selected.merge(movies, on="rotten_tomatoes_link", how="left")
    selected["label"] = (selected["review_type"] == "Fresh").astype(int)
    selected["review_date"] = pd.to_datetime(
        selected["review_date"], errors="coerce"
    )
    selected["review_length_words"] = selected["review_content"].str.split().str.len()
    selected = selected.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    selected.to_csv(OUTPUT_FILE, index=False)

    print(f"Saved {len(selected):,} reviews to {OUTPUT_FILE}")
    print(f"Distinct movies: {selected['rotten_tomatoes_link'].nunique():,}")
    print(selected["review_type"].value_counts().to_string())
    print(f"Submitted dataset download: {DATA_DOWNLOAD_URL}")


if __name__ == "__main__":
    main()
