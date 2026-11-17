"""Collect Rotten Tomatoes critic reviews with the Homework 4 Selenium parser.

The script saves a checkpoint after every movie, so a long collection run can be
stopped and resumed without losing completed work.
"""

import argparse
from pathlib import Path

import pandas as pd

from rt_scraper import build_browser, scrape_movie


ROOT = Path(__file__).resolve().parent
DEFAULT_MOVIE_LIST = ROOT / "data" / "rotten_tomatoes_hw5.csv"
DEFAULT_OUTPUT = ROOT / "data" / "parser_collected_reviews.csv"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--movie-list", type=Path, default=DEFAULT_MOVIE_LIST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pages", type=int, default=3)
    parser.add_argument("--minimum-movies", type=int, default=500)
    parser.add_argument("--max-movies", type=int)
    parser.add_argument("--visible", action="store_true")
    return parser.parse_args()


def save_checkpoint(rows, output):
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.drop_duplicates(
            subset=["rotten_tomatoes_link", "review_content"]
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    return frame


def main():
    args = parse_args()
    movies = pd.read_csv(args.movie_list)
    if "movie_slug" not in movies and "rotten_tomatoes_link" in movies:
        movies["movie_slug"] = movies["rotten_tomatoes_link"].str.replace(
            r"^m/", "", regex=True
        )
    movies = movies.drop_duplicates("movie_slug").reset_index(drop=True)
    required = {"movie_slug", "movie_title"}
    if not required.issubset(movies.columns):
        raise ValueError(f"Movie list must contain columns: {sorted(required)}")
    if args.max_movies:
        movies = movies.head(args.max_movies)

    existing = pd.read_csv(args.output) if args.output.exists() else pd.DataFrame()
    rows = existing.to_dict("records")
    completed = set(existing.get("rotten_tomatoes_link", []))

    browser = build_browser(headless=not args.visible)
    try:
        for number, movie in movies.iterrows():
            link = f"m/{movie['movie_slug']}"
            if link in completed:
                continue
            try:
                reviews = scrape_movie(
                    browser, movie["movie_slug"], page_num=args.pages
                )
                for review in reviews:
                    review_type = str(review.get("rating", "")).strip().title()
                    text = str(review.get("text", "")).strip()
                    if review_type not in {"Fresh", "Rotten"} or not text:
                        continue
                    rows.append(
                        {
                            "rotten_tomatoes_link": link,
                            "movie_title": movie["movie_title"],
                            "critic_name": review.get("critic"),
                            "publisher_name": review.get("source"),
                            "review_type": review_type,
                            "review_date": review.get("date"),
                            "review_content": text,
                            "label": int(review_type == "Fresh"),
                            "review_length_words": len(text.split()),
                        }
                    )
                completed.add(link)
                frame = save_checkpoint(rows, args.output)
                print(
                    f"Processed {number + 1}/{len(movies)}: "
                    f"{frame['rotten_tomatoes_link'].nunique()} movies, "
                    f"{len(frame):,} reviews"
                )
            except Exception as error:
                print(f"Skipped {link}: {error}")
    finally:
        browser.quit()

    frame = save_checkpoint(rows, args.output)
    movie_count = frame["rotten_tomatoes_link"].nunique() if not frame.empty else 0
    if movie_count < args.minimum_movies or len(frame) < 10_000:
        raise RuntimeError(
            f"Collection has {len(frame):,} reviews from {movie_count} movies; "
            "continue the resumable run to reach 10,000 reviews from 500 movies."
        )
    print(f"Complete: {len(frame):,} reviews from {movie_count} movies")


if __name__ == "__main__":
    main()
