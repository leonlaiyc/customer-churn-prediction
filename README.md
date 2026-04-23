# Customer Churn Prediction

Kaggle Playground Series S6E3 portfolio project focused on leakage-aware validation, feature engineering, and practical tabular modeling decisions.

## Highlights
- Top 10% finish in Kaggle Playground Series S6E3
- Private Rank: 390 / 4142
- Public Rank: 404 / 4142
- Public Score: 0.91690
- Private Score: 0.91796
- Best OOF AUC: 0.919363
- Primary modeling pipeline: Ridge → XGBoost
- Secondary benchmark: RealMLP
- Final ensemble selected using OOF AUC and prediction correlation

## What this repo shows
This repository is not designed as a general-purpose machine learning framework.
It is a competition-specific portfolio project intended to demonstrate:
- validation discipline
- feature engineering judgment
- model comparison and ensemble decisions
- clean, interview-ready code

I used AI tools to accelerate parts of the coding workflow, but the modeling logic, validation design, feature ideas, and decision-making process were my own.

## Problem
The competition goal was to predict customer churn probability from a tabular customer dataset. From a portfolio perspective, I treated this project less as a leaderboard exercise and more as a chance to show how I approach a practical binary classification problem:
- structure a validation workflow carefully
- engineer features with clear hypotheses
- compare model families instead of relying on a single model
- make conservative, evidence-based ensemble decisions

## Approach
My workflow had four main parts.

### 1. Feature engineering
I built competition-specific features on top of the original tabular data, including:
- frequency-based signals
- arithmetic interaction features
- service usage summaries
- prior signals derived from the original Telco churn dataset
- distributional and percentile-based features
- categorical interaction features such as bigrams and trigrams

The goal was not to generate as many features as possible, but to create features that reflected customer behavior patterns, pricing consistency, and service configuration differences that could plausibly relate to churn.

### 2. Leakage-aware target encoding
A major focus of the project was validation discipline. I used inner-fold target encoding inside each training fold rather than computing encodings on the full fold directly. This was important because target encoding can easily introduce leakage if it is not built carefully.

### 3. Two-stage modeling
My primary pipeline was a two-stage setup:
- Stage 1: Ridge regression to capture simpler linear structure
- Stage 2: XGBoost trained with the Ridge prediction added as an input feature

The reasoning was practical rather than academic. I wanted a strong tabular baseline that could first absorb stable linear signal, then allow the tree model to focus on nonlinear interactions and residual structure.

### 4. Complementary benchmark and ensemble
I also trained a RealMLP model as a secondary benchmark. Rather than assuming a neural model would outperform tree-based methods, I treated it as a complementary candidate and compared it through out-of-fold behavior.

For the final blend, I used OOF predictions to inspect both AUC and prediction correlation before selecting the ensemble weights.

## Key Decisions
### Why I did not rely on a single model
A single strong model is often enough for a decent leaderboard result, but for portfolio purposes I wanted to show model comparison discipline. The Ridge → XGBoost pipeline represented my main modeling hypothesis, while RealMLP served as a check on whether a different model family added useful signal.

### Why leakage control mattered so much
This competition was simple enough that it would have been easy to overstate performance by building encoding features incorrectly. I intentionally chose the safer route, even though it made the pipeline more verbose, because I wanted the validation process to remain defensible in an interview setting.

### Why the final ensemble was conservative
The RealMLP model was useful, but its OOF predictions were highly correlated with the XGBoost pipeline. That meant the ensemble gain was incremental rather than dramatic. I kept the final blend simple and selected weights using OOF evidence instead of over-optimizing around a tiny leaderboard difference.

## Results
### Final leaderboard results
- Public Rank: 404 / 4142
- Private Rank: 390 / 4142
- Public Score: 0.91690
- Private Score: 0.91796

### OOF summary for the final version
- XGBoost OOF AUC: 0.919275
- RealMLP OOF AUC: 0.918700
- OOF Correlation: 0.995972
- Best Ensemble Weight: 0.75 XGBoost / 0.25 RealMLP
- Best Ensemble OOF AUC: 0.919363

The ensemble improvement was small but directionally consistent. I view that as a useful result rather than a disappointment: it showed that the secondary model added limited but real value, while also confirming that the main pipeline was already capturing most of the available signal.

## Learnings
This project reinforced four lessons that are directly relevant to data science work outside Kaggle.

### 1. Validation quality matters as much as model choice
Strong features and powerful models are not enough if the validation setup is careless. The most transferable part of this project was not the exact model stack, but the discipline around how features were created and evaluated.

### 2. Feature engineering is most valuable when it reflects problem structure
The best features were not always the most complicated ones. The most useful features were usually those tied to interpretable hypotheses about pricing behavior, service mix, and how a customer’s configuration differed from typical churn and non-churn patterns.

### 3. Model diversity only helps when it adds non-redundant signal
Trying multiple model families is useful, but only when their behavior is sufficiently different. In this case, RealMLP was competitive, but its predictions were still very close to the XGBoost pipeline, which limited ensemble upside.

### 4. For portfolio work, clarity matters more than novelty
For hiring purposes, the value of this project is not that it used an unusual architecture. The value is that it shows a clean thought process: careful validation, reasoned feature engineering, sensible model comparison, and honest interpretation of results.

## Repository Structure
```text
customer-churn-prediction/
├── data/
│   └── .gitkeep
├── src/
│   ├── feature_engineering.py
│   ├── train_twostage.py
│   ├── train_realmlp.py
│   └── ensemble.py
├── .gitignore
├── requirements.txt
└── README.md
```

## Notes
- The raw competition data is intentionally not included in this repository.
- This repo is meant to be readable and interview-ready, not packaged as a general-purpose framework.
- The code is structured to reflect the main modeling decisions I would want to discuss in a hiring conversation.
