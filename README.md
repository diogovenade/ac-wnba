# AC Data Mining Project - WNBA Season Prediction

This repository contains a data mining project focused on WNBA season prediction tasks. It uses historical WNBA datasets to explore, preprocess, and model team rankings, coaching changes, and individual award winners. 

The project was developed as part of the [Machine Learning](https://sigarra.up.pt/feup/en/UCURR_GERAL.FICHA_UC_VIEW?pv_ocorrencia_id=560265) course.

**Group G63**
- Bernardo Costa - up202207579
- Diogo Venade - up202207805
- Vasco Costa - up202109923

## Project Overview

The work is organized around three prediction goals:

- Conference regular-season ranking prediction
- Coaching change prediction
- Individual award winner prediction

The analysis is implemented mostly in Jupyter notebooks, with small Python helper modules for missing-data imputation and outlier detection.

## Notebooks

- `src/data_understanding.ipynb` - business understanding, dataset loading, data description, quality checks, and exploratory analysis.
- `src/data_preprocessing.ipynb` - data cleaning, missing-value imputation, outlier analysis, and feature engineering.
- `src/ranking.ipynb` - regular-season conference ranking prediction.
- `src/coach_change.ipynb` - coaching change prediction using time-series validation and classification models.
- `src/awards.ipynb` - individual player and coach award winner prediction.

## Data

The main historical datasets are stored in `datasets/`:

- `teams.csv`
- `teams_post.csv`
- `series_post.csv`
- `players.csv`
- `players_teams.csv`
- `coaches.csv`
- `awards_players.csv`

The `Season_11/` directory contains the Season 11 input files used by the prediction notebooks:

- `teams.csv`
- `players_teams.csv`
- `coaches.csv`

## Helper Modules

- `logic/missing_data.py` provides Random Forest based imputation for missing player height and weight values.
- `logic/outliers.py` provides Z-score, IQR, and visualization utilities for detecting outliers in player, team, and player-team statistics.
