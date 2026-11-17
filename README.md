# Homework 5: Rotten Tomatoes Review Classifier

**Student:** Ishfaq Shah  
**Course:** Web Analytics  
**Professor:** Apostolos Filippas

This project builds and evaluates machine-learning classifiers that predict whether a Rotten Tomatoes critic review is **Fresh** or **Rotten**. It also performs exploratory data analysis and compares the best traditional classifier with an OpenAI model on the same held-out reviews.

## Dataset

`rt_scraper.py` contains the Homework 4 Selenium parser. `collect_reviews.py` applies it to the supplied movie list, checkpoints after every movie, and can resume a long run. It collects the review text, rating, movie identifier, critic, publisher, and date. The committed `data/rotten_tomatoes_hw5.csv` is a cached, reproducible snapshot so the analysis can be graded without repeating hundreds of live web requests. `prepare_data.py` can clean either parser output or rebuild that snapshot from the linked public [Rotten Tomatoes Movies and Critic Reviews archive](https://www.kaggle.com/datasets/stefanoleone992/rotten-tomatoes-movies-and-critic-reviews-dataset).

The working dataset contains reviews from 1,000 randomly selected movies, with up to 50 reviews per movie. This exceeds the assignment minimum of 10,000 reviews from 500 distinct movies.

**Direct data download:** [rotten_tomatoes_hw5.csv](https://raw.githubusercontent.com/ishah2005/web-analytics-homework-5/main/data/rotten_tomatoes_hw5.csv)

## Files

- `rt_scraper.py`: Homework 4 Selenium parser for one movie.
- `collect_reviews.py`: batch collection, checkpointing, and resume support for 500+ movies.
- `prepare_data.py`: validates, cleans, and combines parser output; it can also rebuild the cached snapshot.
- `train_and_analyze.py`: performs EDA, creates a movie-level train/test split, trains four classifiers, evaluates them, and saves charts.
- `llm_compare.py`: classifies a held-out random sample with the cheapest suitable OpenAI model and calculates the exact cost per review.
- `results/`: saved metrics, held-out data, and visualizations.
- `homework_5_analysis.ipynb`: complete analysis notebook.
- `output/Ishfaq Shah Homework 5 Report.pdf`: final written report.

## Final model comparison

- Best traditional model on the full 10,000-review test set: TF-IDF unigram + bigram logistic regression, 78.36% accuracy and 0.8202 F1.
- On the exact same 200 held-out reviews, the traditional model achieved 76.50% accuracy and 0.7930 F1; GPT-6 Luna achieved 92.50% accuracy and 0.9321 F1.
- LLM cost: $0.001649 total, or $0.00000824 per review.

## Reproducing the analysis

```bash
python -m pip install -r requirements.txt
python prepare_data.py
python train_and_analyze.py
```

To perform a fresh parser-based collection instead of using the cached snapshot:

```bash
python collect_reviews.py
python prepare_data.py --parser-input data/parser_collected_reviews.csv
python train_and_analyze.py
```

For the LLM comparison, configure an OpenAI API key and run:

```bash
export OPENAI_API_KEY="your-key-here"
python llm_compare.py
```

The API key is never stored in this repository.
