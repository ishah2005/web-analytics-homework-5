"""Train, evaluate, and visualize Rotten Tomatoes review classifiers."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


RANDOM_STATE = 4506
ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "data" / "rotten_tomatoes_hw5.csv"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"


def metrics_for(name, y_true, predictions, scores=None):
    result = {
        "model": name,
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(y_true, predictions),
        "recall": recall_score(y_true, predictions),
        "f1": f1_score(y_true, predictions),
    }
    if scores is not None:
        result["roc_auc"] = roc_auc_score(y_true, scores)
    return result


def save_eda(df):
    sns.set_theme(style="whitegrid", palette="deep")

    plt.figure(figsize=(7, 4.5))
    order = ["Fresh", "Rotten"]
    ax = sns.countplot(data=df, x="review_type", order=order)
    ax.set(title="Distribution of Review Ratings", xlabel="Rating", ylabel="Reviews")
    for container in ax.containers:
        ax.bar_label(container, fmt="{:,.0f}")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "rating_distribution.png", dpi=180)
    plt.close()

    clipped = df[df["review_length_words"] <= df["review_length_words"].quantile(0.99)]
    plt.figure(figsize=(8, 4.8))
    sns.boxplot(
        data=clipped,
        x="review_type",
        y="review_length_words",
        order=order,
        showfliers=False,
    )
    plt.title("Review Length by Rating")
    plt.xlabel("Rating")
    plt.ylabel("Words per review")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "review_length.png", dpi=180)
    plt.close()

    dated = df.dropna(subset=["review_date"]).copy()
    dated = dated.sort_values(["rotten_tomatoes_link", "review_date"])
    dated["review_order_fraction"] = dated.groupby("rotten_tomatoes_link").cumcount() / (
        dated.groupby("rotten_tomatoes_link")["review_date"].transform("size") - 1
    ).replace(0, 1)
    dated["review_period"] = np.where(
        dated["review_order_fraction"] < 0.5, "Earlier half", "Later half"
    )
    period_rates = (
        dated.groupby("review_period")["label"]
        .agg(["mean", "count"])
        .reindex(["Earlier half", "Later half"])
        .reset_index()
    )
    period_rates.to_csv(RESULTS_DIR / "early_late_rates.csv", index=False)

    plt.figure(figsize=(7, 4.5))
    ax = sns.barplot(data=period_rates, x="review_period", y="mean")
    ax.set(
        title="Fresh-Review Rate: Earlier vs. Later Reviews",
        xlabel="Position in each movie's review timeline",
        ylabel="Share rated Fresh",
        ylim=(0, 1),
    )
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "early_vs_late.png", dpi=180)
    plt.close()

    monthly = dated.set_index("review_date").resample("YE")["label"].agg(["mean", "count"])
    monthly = monthly[monthly["count"] >= 100]
    plt.figure(figsize=(9, 4.8))
    sns.lineplot(data=monthly, x=monthly.index, y="mean", marker="o")
    plt.title("Fresh-Review Rate over Time")
    plt.xlabel("Review year")
    plt.ylabel("Share rated Fresh")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "rating_over_time.png", dpi=180)
    plt.close()

    return period_rates


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_FILE)
    df = df.dropna(subset=["review_content", "label", "rotten_tomatoes_link"])
    df["label"] = df["label"].astype(int)
    df["review_date"] = pd.to_datetime(df["review_date"], errors="coerce")

    summary = {
        "reviews": int(len(df)),
        "movies": int(df["rotten_tomatoes_link"].nunique()),
        "fresh_reviews": int(df["label"].sum()),
        "rotten_reviews": int((1 - df["label"]).sum()),
        "fresh_rate": float(df["label"].mean()),
        "median_review_words": float(df["review_length_words"].median()),
    }

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=RANDOM_STATE)
    train_idx, test_idx = next(
        splitter.split(df, y=df["label"], groups=df["rotten_tomatoes_link"])
    )
    train = df.iloc[train_idx].copy()
    test = df.iloc[test_idx].copy()

    train.to_csv(RESULTS_DIR / "train_split.csv", index=False)
    test.to_csv(RESULTS_DIR / "held_out_test_reviews.csv", index=False)

    models = {
        "Baseline: unigram TF-IDF + logistic regression": Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        ngram_range=(1, 1), min_df=3, max_df=0.95, sublinear_tf=True
                    ),
                ),
                ("model", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
            ]
        ),
        "Unigram + bigram TF-IDF + logistic regression": Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        ngram_range=(1, 2),
                        min_df=3,
                        max_df=0.95,
                        sublinear_tf=True,
                        max_features=150000,
                    ),
                ),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000, C=2.0, class_weight="balanced", random_state=RANDOM_STATE
                    ),
                ),
            ]
        ),
        "Unigram + bigram TF-IDF + linear SVM": Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        ngram_range=(1, 2),
                        min_df=3,
                        max_df=0.95,
                        sublinear_tf=True,
                        max_features=150000,
                    ),
                ),
                ("model", LinearSVC(C=0.7, class_weight="balanced", random_state=RANDOM_STATE)),
            ]
        ),
        "Unigram TF-IDF + multinomial Naive Bayes": Pipeline(
            [
                ("tfidf", TfidfVectorizer(ngram_range=(1, 1), min_df=3, max_df=0.95)),
                ("model", MultinomialNB(alpha=0.5)),
            ]
        ),
    }

    results = []
    predictions_by_model = {}
    for name, model in models.items():
        print(f"Training {name}")
        model.fit(train["review_content"], train["label"])
        predictions = model.predict(test["review_content"])
        if hasattr(model, "predict_proba"):
            scores = model.predict_proba(test["review_content"])[:, 1]
        elif hasattr(model, "decision_function"):
            scores = model.decision_function(test["review_content"])
        else:
            scores = None
        results.append(metrics_for(name, test["label"], predictions, scores))
        predictions_by_model[name] = predictions

    # Accuracy is the primary comparison metric; F1 and ROC-AUC are also reported.
    results_df = pd.DataFrame(results).sort_values("accuracy", ascending=False)
    results_df.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
    best_name = results_df.iloc[0]["model"]
    best_predictions = predictions_by_model[best_name]

    report = classification_report(
        test["label"], best_predictions, target_names=["Rotten", "Fresh"], digits=4
    )
    (RESULTS_DIR / "best_model_classification_report.txt").write_text(report)

    cm = confusion_matrix(test["label"], best_predictions)
    plt.figure(figsize=(5.5, 4.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt=",",
        cmap="Blues",
        xticklabels=["Rotten", "Fresh"],
        yticklabels=["Rotten", "Fresh"],
    )
    plt.title(f"Best Model Confusion Matrix\n{best_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=180)
    plt.close()

    melted = results_df.melt(
        id_vars="model", value_vars=["accuracy", "precision", "recall", "f1"]
    )
    plt.figure(figsize=(10, 5.5))
    sns.barplot(data=melted, y="model", x="value", hue="variable")
    plt.xlim(0.5, 1)
    plt.xlabel("Score")
    plt.ylabel("")
    plt.title("Classifier Performance on Held-Out Movies")
    plt.legend(title="Metric", loc="lower right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "model_comparison.png", dpi=180)
    plt.close()

    period_rates = save_eda(df)
    summary.update(
        {
            "train_reviews": int(len(train)),
            "test_reviews": int(len(test)),
            "train_movies": int(train["rotten_tomatoes_link"].nunique()),
            "test_movies": int(test["rotten_tomatoes_link"].nunique()),
            "best_model": best_name,
            "best_accuracy": float(results_df.iloc[0]["accuracy"]),
            "best_f1": float(results_df.iloc[0]["f1"]),
            "early_fresh_rate": float(period_rates.iloc[0]["mean"]),
            "late_fresh_rate": float(period_rates.iloc[1]["mean"]),
        }
    )
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    llm_sample = test.sample(n=min(200, len(test)), random_state=RANDOM_STATE)
    llm_sample[
        ["rotten_tomatoes_link", "movie_title", "review_content", "label"]
    ].to_csv(RESULTS_DIR / "llm_sample.csv", index=False)

    print("\nDataset summary:")
    print(json.dumps(summary, indent=2))
    print("\nModel comparison:")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
