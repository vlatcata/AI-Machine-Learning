## 🎮 Best Video Game Analysis


📌 Abstract

This project analyzes a large dataset of video games from Steam and SteamSpy to identify the top-performing titles based on custom metrics. It combines gameplay, review, and popularity data into a single score (GameScore) that ranks games more reliably than raw sales or reviews alone.


🚀 Features

Custom GameScore formula that balances owners, retention, concurrent players, reviews, and playtime.

Data cleaning and exploratory analysis (distributions, correlations, sanity checks).

Interactive visualization (Plotly & Gradio).

Sensitivity analysis to test weight stability.

Reproducible pipeline (documented steps to rerun).


🔄 Data Collection (Custom Scraper)

This project includes a custom **scraper** (`scraper.py`) that automatically collects fresh data from the **Steam API** and **Steam Spy API**.  

To run it:
```bash
python scraper.py
```

The scraper works correctly, but fetching data from the Steam API takes a long time due to strict rate limits and Steam Spy availability.


📂 Dataset

The dataset is too large for GitHub.
Download it here: [Kaggle Dataset](https://www.kaggle.com/datasets/fronkongames/steam-games-dataset/data)

Place the file into the data/ folder before running.


🛠️ Installation

Clone the repo and install dependencies:

```
git clone https://github.com/vlatcata/AI-Machine-Learning.git
cd "Data Science/Project - Best Video Game Analysis"
pip install -r requirements.txt
```


▶️ Usage

Run the notebook to reproduce results:

```
jupyter notebook notebooks/Best\ Video\ Game\ Analysis.ipynb
```

Or launch the Gradio app for interactive exploration:

```
python app.py
```


📊 Results

The GameScore index provides a balanced ranking of games.

Top results remain stable under noise testing (9/10 overlap on average).

Visualizations confirm key trends in player engagement and review behavior.


📑 Reproducibility

Full preprocessing, analysis, and model evaluation steps are documented in the notebook.

All results can be reproduced by downloading the dataset and running the provided code.
