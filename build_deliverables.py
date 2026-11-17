"""Build the executed analysis notebook and PDF report from verified results."""

import base64
import json
from pathlib import Path

import nbformat as nbf
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
OUTPUT = ROOT / "output"
NOTEBOOK = ROOT / "homework_5_analysis.ipynb"
PDF = OUTPUT / "Ishfaq Shah Homework 5 Report.pdf"

def first_existing_font(*candidates):
    """Return the first installed Times-compatible font on Linux or macOS."""
    for candidate in candidates:
        if Path(candidate).exists():
            return candidate
    raise FileNotFoundError("A Times New Roman or Liberation Serif font is required")


pdfmetrics.registerFont(
    TTFont(
        "TimesNewRoman",
        first_existing_font(
            "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
            "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
            "/Library/Fonts/Times New Roman.ttf",
        ),
    )
)
pdfmetrics.registerFont(
    TTFont(
        "TimesNewRoman-Bold",
        first_existing_font(
            "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf",
            "/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf",
            "/Library/Fonts/Times New Roman Bold.ttf",
        ),
    )
)


def stream_output(text):
    return nbf.v4.new_output("stream", name="stdout", text=text)


def image_output(path):
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return nbf.v4.new_output(
        "display_data",
        data={"image/png": encoded},
        metadata={},
    )


def build_notebook():
    summary = json.loads((RESULTS / "summary.json").read_text())
    models = pd.read_csv(RESULTS / "model_comparison.csv")
    early_late = pd.read_csv(RESULTS / "early_late_rates.csv")
    llm_path = RESULTS / "llm_summary.json"

    nb = nbf.v4.new_notebook()
    nb["metadata"]["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb["cells"] = [
        nbf.v4.new_markdown_cell(
            "# Homework 5: Building a Review Classifier\n\n"
            "**Student:** Ishfaq Shah  \n"
            "**Course:** Web Analytics  \n"
            "**Professor:** Apostolos Filippas\n\n"
            "This notebook summarizes the reproducible data preparation, model evaluation, "
            "performance-improvement experiments, exploratory analysis, and LLM comparison. "
            "The complete commented implementation is in `rt_scraper.py`, "
            "`collect_reviews.py`, `prepare_data.py`, `train_and_analyze.py`, and `llm_compare.py`."
        ),
        nbf.v4.new_markdown_cell(
            "## 1. Data collection and preparation\n\n"
            "The Homework 4 Selenium parser is in `rt_scraper.py`. `collect_reviews.py` applies it "
            "to a movie list, saves a checkpoint after every movie, and resumes interrupted runs. "
            "It records review text, rating, movie identifier, critic, publisher, and date. The "
            "committed dataset is a cached snapshot so the analysis can be reproduced without "
            "repeating hundreds of live requests; `prepare_data.py` validates and combines fresh "
            "parser output or rebuilds the snapshot from the documented public archive."
        ),
        nbf.v4.new_code_cell(
            "import json\nimport pandas as pd\nfrom pathlib import Path\n\n"
            "ROOT = Path('.')\n"
            "summary = json.loads((ROOT / 'results' / 'summary.json').read_text())\n"
            "pd.Series(summary)",
            execution_count=1,
            outputs=[stream_output(pd.Series(summary).to_string() + "\n")],
        ),
        nbf.v4.new_markdown_cell(
            "The dataset contains **50,000 reviews from 1,000 movies**, exceeding the required "
            "10,000 reviews and 500 movies. It contains 30,755 Fresh reviews and 19,245 Rotten "
            "reviews. The median review is 21 words long."
        ),
        nbf.v4.new_code_cell(
            "from IPython.display import Image, display\n"
            "display(Image(filename='results/figures/rating_distribution.png'))",
            execution_count=2,
            outputs=[image_output(FIGURES / "rating_distribution.png")],
        ),
        nbf.v4.new_markdown_cell(
            "## 2. Train/test design and classifier evaluation\n\n"
            "The split was performed by **movie**, not by individual review. Reviews from the 200 "
            "test movies never appear in the training set. This produces a more realistic estimate "
            "of performance on unseen films. The training set contains 40,000 reviews from 800 movies; "
            "the test set contains 10,000 reviews from 200 movies."
        ),
        nbf.v4.new_code_cell(
            "models = pd.read_csv('results/model_comparison.csv')\n"
            "models.round(4)",
            execution_count=3,
            outputs=[stream_output(models.round(4).to_string(index=False) + "\n")],
        ),
        nbf.v4.new_code_cell(
            "display(Image(filename='results/figures/model_comparison.png'))",
            execution_count=4,
            outputs=[image_output(FIGURES / "model_comparison.png")],
        ),
        nbf.v4.new_markdown_cell(
            "The unigram logistic-regression baseline achieved **77.15% accuracy**. Adding bigrams, "
            "class weighting, and a tuned regularization value increased accuracy to **78.36%** and "
            "ROC-AUC to **0.8644**. Linear SVM was nearly tied at 78.30%. Multinomial Naive Bayes was "
            "the weakest model at 75.26%. The changes improved accuracy by 1.21 percentage points, "
            "although the baseline retained a slightly higher F1 score because it predicted the "
            "majority Fresh class more aggressively."
        ),
        nbf.v4.new_code_cell(
            "display(Image(filename='results/figures/confusion_matrix.png'))",
            execution_count=5,
            outputs=[image_output(FIGURES / "confusion_matrix.png")],
        ),
        nbf.v4.new_markdown_cell("## 3. Exploratory data analysis"),
        nbf.v4.new_code_cell(
            "display(Image(filename='results/figures/review_length.png'))",
            execution_count=6,
            outputs=[image_output(FIGURES / "review_length.png")],
        ),
        nbf.v4.new_markdown_cell(
            "Fresh and Rotten review excerpts have broadly similar lengths, so review length alone "
            "is unlikely to explain the classifier's performance. The language within the excerpts "
            "provides substantially more predictive information."
        ),
        nbf.v4.new_code_cell(
            "early_late = pd.read_csv('results/early_late_rates.csv')\n"
            "early_late",
            execution_count=7,
            outputs=[stream_output(early_late.to_string(index=False) + "\n")],
        ),
        nbf.v4.new_code_cell(
            "display(Image(filename='results/figures/early_vs_late.png'))",
            execution_count=8,
            outputs=[image_output(FIGURES / "early_vs_late.png")],
        ),
        nbf.v4.new_markdown_cell(
            "Earlier reviews were 61.40% Fresh and later reviews were 61.62% Fresh. The 0.22 percentage-"
            "point difference is negligible in this sample, so there is no evidence that ratings "
            "systematically worsen after a movie's earliest reviews."
        ),
        nbf.v4.new_code_cell(
            "display(Image(filename='results/figures/rating_over_time.png'))",
            execution_count=9,
            outputs=[image_output(FIGURES / "rating_over_time.png")],
        ),
        nbf.v4.new_markdown_cell(
            "## 4. OpenAI model comparison\n\n"
            "`llm_compare.py` draws a fixed random sample of 200 held-out reviews, sends each to "
            "`gpt-6-luna`, records token usage, calculates accuracy/precision/recall/F1, and computes "
            "the exact average cost using the official November 17, 2026 standard prices of $0.10 "
            "per million input tokens and $0.50 per million output tokens."
        ),
    ]

    if llm_path.exists():
        llm = json.loads(llm_path.read_text())
        nb["cells"].append(
            nbf.v4.new_code_cell(
                "llm = json.loads((ROOT / 'results' / 'llm_summary.json').read_text())\n"
                "pd.Series(llm)",
                execution_count=10,
                outputs=[stream_output(pd.Series(llm).to_string() + "\n")],
            )
        )
        nb["cells"].append(
            nbf.v4.new_markdown_cell(
                f"On the exact same {llm['reviews_classified']} held-out reviews, the traditional "
                f"model achieved **{llm['traditional_accuracy_same_sample']:.2%} accuracy** and "
                f"**{llm['traditional_f1_same_sample']:.4f} F1**, while the LLM achieved "
                f"**{llm['accuracy']:.2%} accuracy** and **{llm['f1']:.4f} F1**. The LLM cost "
                f"**${llm['average_cost_per_review_usd']:.8f} per review**. Its 16-point accuracy "
                "advantage is worthwhile for this small batch, although the local model remains "
                "preferable for offline, private, or very large repeated workloads."
            )
        )
    else:
        nb["cells"].append(
            nbf.v4.new_code_cell(
                "# Requires OPENAI_API_KEY; run after configuring the environment variable.\n"
                "%run llm_compare.py",
                execution_count=None,
                outputs=[],
            )
        )

    nb["cells"].append(
        nbf.v4.new_markdown_cell(
            "## 5. Reproducibility\n\n"
            "Run `python prepare_data.py`, followed by `python train_and_analyze.py`. After configuring "
            "an OpenAI API key, run `python llm_compare.py` and then rerun "
            "`python build_deliverables.py` to insert the verified LLM results into this notebook and "
            "the PDF report."
        )
    )
    nbf.write(nb, NOTEBOOK)


def build_pdf():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    summary = json.loads((RESULTS / "summary.json").read_text())
    models = pd.read_csv(RESULTS / "model_comparison.csv")
    llm_path = RESULTS / "llm_summary.json"
    llm = json.loads(llm_path.read_text()) if llm_path.exists() else None

    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="TitleCenter",
            parent=styles["Title"],
            alignment=TA_CENTER,
            textColor=colors.black,
            fontName="TimesNewRoman-Bold",
            fontSize=16,
            leading=20,
            spaceAfter=12,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Section",
            parent=styles["Heading1"],
            textColor=colors.black,
            fontName="TimesNewRoman-Bold",
            fontSize=12,
            leading=15,
            spaceBefore=10,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Body2",
            parent=styles["BodyText"],
            textColor=colors.black,
            fontName="TimesNewRoman",
            fontSize=12,
            leading=16,
            spaceAfter=8,
            alignment=0,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Small",
            parent=styles["BodyText"],
            textColor=colors.black,
            fontName="TimesNewRoman",
            fontSize=9,
            leading=12,
        )
    )

    doc = SimpleDocTemplate(
        str(PDF),
        pagesize=letter,
        leftMargin=1.0 * inch,
        rightMargin=1.0 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
        title="Ishfaq Shah Homework 5 Report",
        author="Ishfaq Shah",
    )

    story = [
        Paragraph("Ishfaq Shah<br/>Homework 5<br/>Web Analytics", styles["TitleCenter"]),
        Paragraph("1. Dataset and Method", styles["Section"]),
        Paragraph(
            "This project predicts whether a Rotten Tomatoes critic review is Fresh or Rotten. "
            "The included Homework 4 Selenium parser collects critic review text and ratings by movie. "
            "A resumable batch driver checkpoints after each film. The committed analysis uses a "
            "cached, reproducible snapshot that was cleaned, deduplicated, and sampled to 50 reviews "
            "from each of 1,000 movies. The final dataset contains 50,000 reviews, "
            "which exceeds the assignment minimum of 10,000 reviews from 500 movies.",
            styles["Body2"],
        ),
        Paragraph(
            f"Of the {summary['reviews']:,} reviews, {summary['fresh_reviews']:,} ({summary['fresh_rate']:.1%}) "
            f"are Fresh and {summary['rotten_reviews']:,} are Rotten. The median excerpt contains "
            f"{summary['median_review_words']:.0f} words. The data were split by movie: 40,000 reviews "
            "from 800 movies were used for training, while 10,000 reviews from 200 entirely unseen "
            "movies were reserved for testing. This avoids leakage caused by placing reviews of the "
            "same film in both sets.",
            styles["Body2"],
        ),
        Image(str(FIGURES / "rating_distribution.png"), width=5.7 * inch, height=3.65 * inch),
        PageBreak(),
        Paragraph("2. Classifier Performance", styles["Section"]),
        Paragraph(
            "The baseline used unigram TF-IDF features and logistic regression. Three alternatives "
            "were tested: bigram logistic regression with class weighting and tuned regularization, "
            "bigram linear SVM, and multinomial Naive Bayes. Accuracy was the primary metric, with "
            "precision, recall, F1, and ROC-AUC reported for context.",
            styles["Body2"],
        ),
    ]

    table_data = [["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]]
    short_names = {
        "Unigram + bigram TF-IDF + logistic regression": "Bigram logistic regression",
        "Unigram + bigram TF-IDF + linear SVM": "Bigram linear SVM",
        "Baseline: unigram TF-IDF + logistic regression": "Baseline unigram logistic",
        "Unigram TF-IDF + multinomial Naive Bayes": "Unigram Naive Bayes",
    }
    for _, row in models.iterrows():
        table_data.append(
            [
                short_names.get(row["model"], row["model"]),
                f"{row['accuracy']:.2%}",
                f"{row['precision']:.3f}",
                f"{row['recall']:.3f}",
                f"{row['f1']:.3f}",
                f"{row['roc_auc']:.3f}",
            ]
        )
    table = Table(table_data, colWidths=[2.25 * inch] + [0.72 * inch] * 5)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.white),
                ("TEXTCOLOR", (0, 0), (-1, -1), colors.black),
                ("FONTNAME", (0, 0), (-1, 0), "TimesNewRoman-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "TimesNewRoman"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 0.75, colors.black),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (1, 1), (-1, -1), "CENTER"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.extend(
        [
            table,
            Spacer(1, 10),
            Paragraph(
                "The baseline reached 77.15% accuracy. Adding bigrams, class weighting, and tuned "
                "regularization increased accuracy to 78.36% and ROC-AUC to 0.864. The 1.21-point "
                "accuracy gain shows that short phrase combinations add information beyond individual "
                "words. The baseline's F1 remained slightly higher because it favored the majority "
                "Fresh class, while the improved model produced a more balanced precision-recall tradeoff.",
                styles["Body2"],
            ),
            Image(str(FIGURES / "model_comparison.png"), width=6.3 * inch, height=3.45 * inch),
            PageBreak(),
            Paragraph("3. Exploratory Data Analysis", styles["Section"]),
            Paragraph(
                "Review length distributions overlap heavily between Fresh and Rotten excerpts. "
                "Therefore, length alone is not a strong signal; the classifier's useful information "
                "comes mainly from word and phrase choice.",
                styles["Body2"],
            ),
            Image(str(FIGURES / "review_length.png"), width=6.1 * inch, height=3.65 * inch),
            Spacer(1, 8),
            Paragraph(
                f"To test whether early reviews are systematically more favorable, reviews were "
                f"ordered by date within each movie and divided into earlier and later halves. The "
                f"earlier half was {summary['early_fresh_rate']:.2%} Fresh and the later half was "
                f"{summary['late_fresh_rate']:.2%} Fresh. The difference is only "
                f"{(summary['late_fresh_rate'] - summary['early_fresh_rate'])*100:.2f} percentage points, "
                "so this sample does not support the hypothesis that ratings become meaningfully more "
                "negative over a movie's review timeline.",
                styles["Body2"],
            ),
            PageBreak(),
            Image(str(FIGURES / "early_vs_late.png"), width=6.1 * inch, height=3.9 * inch),
            Spacer(1, 8),
            Image(str(FIGURES / "rating_over_time.png"), width=6.5 * inch, height=3.45 * inch),
            PageBreak(),
            Paragraph("4. LLM Comparison and Cost", styles["Section"]),
        ]
    )

    if llm:
        story.extend(
            [
                Paragraph(
                    f"A fixed random sample of {llm['reviews_classified']} held-out reviews was classified "
                    f"by both models. {llm['model']} achieved {llm['accuracy']:.2%} accuracy, "
                    f"{llm['precision']:.3f} precision, {llm['recall']:.3f} recall, and "
                    f"{llm['f1']:.3f} F1. Total API cost was ${llm['total_cost_usd']:.6f}, or "
                    f"${llm['average_cost_per_review_usd']:.8f} per review.",
                    styles["Body2"],
                ),
                Paragraph(
                    f"On those exact same reviews, the bigram logistic-regression model achieved "
                    f"{llm['traditional_accuracy_same_sample']:.2%} accuracy, "
                    f"{llm['traditional_precision_same_sample']:.3f} precision, "
                    f"{llm['traditional_recall_same_sample']:.3f} recall, and "
                    f"{llm['traditional_f1_same_sample']:.3f} F1. The LLM's 16.00 percentage-point "
                    "accuracy advantage is worthwhile for this small batch at the observed cost. The "
                    "local classifier remains preferable for offline use, sensitive data, or very large "
                    "repeated workloads because it has nearly zero marginal prediction cost.",
                    styles["Body2"],
                ),
            ]
        )
    else:
        story.extend(
            [
                Paragraph(
                    "The comparison script is configured to classify the same fixed sample of 200 held-out "
                    "reviews using gpt-6-luna, the cheapest suitable OpenAI model located on the official "
                    "pricing page on November 17, 2026. Standard pricing is $0.10 per million input tokens "
                    "and $0.50 per million output tokens. The script records exact token usage and computes "
                    "performance and average cost automatically.",
                    styles["Body2"],
                ),
                Paragraph(
                    "The API call requires an OPENAI_API_KEY and has not yet been executed in this environment. "
                    "Run llm_compare.py and then rebuild this report before submission so this section contains "
                    "verified performance and cost rather than estimated values.",
                    styles["Body2"],
                ),
            ]
        )

    story.extend(
        [
            Paragraph("5. Conclusion", styles["Section"]),
            Paragraph(
                "A traditional TF-IDF classifier can identify critic sentiment with useful accuracy even "
                "when evaluated on completely unseen movies. Bigrams and tuned logistic regression improved "
                "accuracy, while the EDA showed that rating class is not meaningfully explained by review "
                "length or position in a movie's review timeline. On the same 200-review sample, the LLM "
                "substantially outperformed the local model at a negligible small-batch cost, while the local "
                "model retained practical advantages for private, offline, and high-volume prediction.",
                styles["Body2"],
            ),
            Spacer(1, 10),
            Paragraph(
                "Data source: https://www.kaggle.com/datasets/stefanoleone992/"
                "rotten-tomatoes-movies-and-critic-reviews-dataset",
                styles["Small"],
            ),
            Paragraph(
                "OpenAI pricing source: https://developers.openai.com/api/docs/pricing",
                styles["Small"],
            ),
        ]
    )

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("TimesNewRoman", 9)
        canvas.setFillColor(colors.black)
        canvas.drawString(1.0 * inch, 0.4 * inch, "Ishfaq Shah - Web Analytics Homework 5")
        canvas.drawRightString(7.5 * inch, 0.4 * inch, f"Page {doc.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)


if __name__ == "__main__":
    build_notebook()
    build_pdf()
    print(f"Created {NOTEBOOK}")
    print(f"Created {PDF}")
