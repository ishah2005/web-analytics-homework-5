"""Classify the held-out sample with OpenAI and calculate cost per review."""

import json
import os
from pathlib import Path

import pandas as pd
from openai import OpenAI
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
SAMPLE_FILE = RESULTS_DIR / "llm_sample.csv"
OUTPUT_FILE = RESULTS_DIR / "llm_predictions.csv"
SUMMARY_FILE = RESULTS_DIR / "llm_summary.json"

# Official standard pricing checked November 17, 2026.
MODEL = "gpt-6-luna"
INPUT_PRICE_PER_MILLION = 0.10
OUTPUT_PRICE_PER_MILLION = 0.50


def classify_review(client, text):
    response = client.responses.create(
        model=MODEL,
        reasoning={"effort": "none"},
        input=(
            "Classify this movie critic review as fresh or rotten. "
            "Return exactly one lowercase word: fresh or rotten.\n\n"
            f"Review: {text}"
        ),
        max_output_tokens=16,
    )
    answer = response.output_text.strip().lower()
    if answer not in {"fresh", "rotten"}:
        raise ValueError(f"Unexpected model response: {answer!r}")
    usage = response.usage
    return answer, usage.input_tokens, usage.output_tokens


def main():
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "Set the OPENAI_API_KEY environment variable before running this script."
        )

    client = OpenAI()
    sample = pd.read_csv(SAMPLE_FILE)
    completed = []

    for number, row in sample.iterrows():
        answer, input_tokens, output_tokens = classify_review(
            client, row["review_content"]
        )
        completed.append(
            {
                **row.to_dict(),
                "llm_prediction_text": answer,
                "llm_prediction": 1 if answer == "fresh" else 0,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            }
        )
        print(f"Classified {number + 1}/{len(sample)}")

    results = pd.DataFrame(completed)
    results["cost_usd"] = (
        results["input_tokens"] / 1_000_000 * INPUT_PRICE_PER_MILLION
        + results["output_tokens"] / 1_000_000 * OUTPUT_PRICE_PER_MILLION
    )
    results.to_csv(OUTPUT_FILE, index=False)

    summary = {
        "model": MODEL,
        "reviews_classified": int(len(results)),
        "accuracy": float(accuracy_score(results["label"], results["llm_prediction"])),
        "precision": float(
            precision_score(results["label"], results["llm_prediction"])
        ),
        "recall": float(recall_score(results["label"], results["llm_prediction"])),
        "f1": float(f1_score(results["label"], results["llm_prediction"])),
        "total_input_tokens": int(results["input_tokens"].sum()),
        "total_output_tokens": int(results["output_tokens"].sum()),
        "total_cost_usd": float(results["cost_usd"].sum()),
        "average_cost_per_review_usd": float(results["cost_usd"].mean()),
        "input_price_per_million": INPUT_PRICE_PER_MILLION,
        "output_price_per_million": OUTPUT_PRICE_PER_MILLION,
    }
    SUMMARY_FILE.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
