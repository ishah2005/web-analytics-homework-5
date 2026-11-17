# Homework 5: Rotten Tomatoes Review Classifier

**Student:** Ishfaq Shah  
**Course:** Web Analytics  
**Professor:** Apostolos Filippas

This project builds and evaluates machine-learning classifiers that predict whether a Rotten Tomatoes critic review is **Fresh** or **Rotten**. It also performs exploratory data analysis and compares the best traditional classifier with an OpenAI model on the same held-out reviews.

## Dataset

The reproducible source dataset is the public [Rotten Tomatoes Movies and Critic Reviews dataset](https://www.kaggle.com/datasets/stefanoleone992/rotten-tomatoes-movies-and-critic-reviews-dataset). The preparation script downloads it automatically and creates `data/rotten_tomatoes_hw5.csv`.

The working dataset contains reviews from 1,000 randomly selected movies, with up to 50 reviews per movie. This exceeds the assignment minimum of 10,000 reviews from 500 distinct movies.

## Files

- `prepare_data.py`: downloads, cleans, samples, and combines the review data.
- `train_and_analyze.py`: performs EDA, creates a movie-level train/test split, trains four classifiers, evaluates them, and saves charts.
- `llm_compare.py`: classifies a held-out random sample with the cheapest suitable OpenAI model and calculates the exact cost per review.
- `results/`: saved metrics, held-out data, and visualizations.
- `homework_5_analysis.ipynb`: complete analysis notebook.
- `output/Ishfaq Shah Homework 5 Report.pdf`: final written report.

## Final model comparison

- Best traditional model: TF-IDF unigram + bigram logistic regression, 78.36% accuracy and 0.8202 F1.
- GPT-6 Luna: 92.50% accuracy and 0.9321 F1 on 200 held-out reviews.
- LLM cost: $0.001649 total, or $0.00000824 per review.

## Reproducing the analysis

```bash
python -m pip install -r requirements.txt
python prepare_data.py
python train_and_analyze.py
```

For the LLM comparison, configure an OpenAI API key and run:

```bash
export OPENAI_API_KEY="your-key-here"
python llm_compare.py
```

The API key is never stored in this repository.
