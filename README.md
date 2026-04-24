# Customer Churn Prediction

Kaggle Playground Series S6E3 portfolio project focused on leakage-aware validation, feature engineering, and practical tabular modeling decisions.

## Highlights

- Top 10% finish in Kaggle Playground Series S6E3
- Private Rank: 390 / 4142
- Public Rank: 404 / 4142
- Public Score: 0.91690
- Private Score: 0.91796
- Best OOF AUC: 0.919363
- Primary modeling pipeline: Ridge -> XGBoost
- Secondary benchmark: RealMLP
- Final ensemble selected using OOF AUC and prediction correlation

## What this repo shows

This repository is not designed as a general-purpose machine learning framework. It is a competition-specific portfolio project intended to show:

- how I turn simple tabular data into richer feature representations
- how I handle validation carefully to reduce leakage risk and overfitting
- how I compare different model families and test whether they add complementary signal

I used AI tools to accelerate implementation and iteration. Final model selection, validation setup, feature selection, and repository framing were reviewed and decided by me.

## Problem

The competition goal was to predict customer churn probability from a tabular customer dataset. The underlying telco-style schema includes contract type, tenure, monthly charges, total charges, and service-related fields such as internet service, tech support, and streaming options.

For portfolio purposes, I used this project to show how I think through validation, feature engineering, and model design in a practical churn prediction problem.

## Approach

My workflow had four main parts.

### 1. Feature Engineering

I built competition-specific features on top of the raw tabular data, including:

- frequency-based signals on core numeric fields
- arithmetic interaction features
- service usage summaries
- prior-rate mappings derived from the original Telco churn dataset
- distributional and percentile-based features on TotalCharges
- categorical interaction features such as bigrams and trigrams

The goal was not to maximize feature count, but to make important relationships in the data more explicit — especially around billing consistency, service adoption, prior risk signals, and how a customer compares with similar segments.

Examples include:

- billing consistency features such as `charges_deviation`, `monthly_to_total_ratio`, and `avg_monthly_charges`. The benefit of these features is that they make the relationship between monthly price, accumulated charges, and tenure more explicit, rather than asking the model to infer that structure only from raw columns.

- service bundle summaries such as `service_count`, `has_internet`, and `has_phone`. These features compress multiple service flags into simpler usage signals, which makes it easier for the model to capture overall product adoption and bundle breadth.

- prior-rate mappings such as `ORIG_proba_Contract`, `ORIG_proba_InternetService`, and related `ORIG_proba_*` features. These features give the model a historical baseline risk signal from the original Telco dataset, instead of relying only on the raw category or numeric value itself.

- residual-style features such as `resid_IS_MC`, which compares a customer's monthly charge against the typical charge of similar internet-service users. The benefit here is that the model can evaluate a customer relative to a more comparable peer group instead of relying only on absolute price levels.

- conditional percentile features such as `cond_pctrank_C_TC`, which place a customer's total charges in the context of similar contract groups. This helps the model understand whether a customer looks unusually high or low within a comparable segment, rather than in the full population only.

### 2. Leakage-Aware Target Encoding

A major focus of the project was validation discipline. I used inner-fold target encoding inside each training fold rather than computing encodings on the full fold directly.

This mattered because target encoding can easily leak label information if it is built too broadly. In that case, validation results can look better than they really are, and the model may generalize less reliably to unseen data.

I chose the more careful approach because I wanted the validation process to stay defensible and the out-of-fold results to better reflect real generalization. I believe this discipline likely contributed to the pipeline holding up slightly better on the private leaderboard rather than overfitting to the public one.

### 3. Two-Stage Modeling

My primary pipeline was a two-stage setup:

- Stage 1: Ridge regression to absorb simpler linear structure
- Stage 2: XGBoost trained with the Ridge prediction added as an input feature

The intent was practical rather than academic. I wanted a strong tabular baseline that first captured stable linear signal, then allowed the tree model to focus on nonlinear interactions and residual structure.

### 4. Complementary Benchmark and Ensemble

I also trained a RealMLP model as a secondary benchmark. Rather than assuming a neural model would outperform tree-based methods, I treated it as a complementary candidate and compared it through out-of-fold behavior.

For the final blend, I used OOF predictions to inspect both AUC and prediction correlation before selecting the ensemble weights.

## Key Decisions

### Why I explored different model families and kept the final blend simple

For competition performance, I did not want to rely on a single model family alone. My main pipeline was Ridge -> XGBoost, while RealMLP was used as a secondary model family to test whether a different modeling perspective could add complementary signal.

After comparing the out-of-fold predictions, I found that RealMLP was competitive but highly correlated with the XGBoost pipeline. That meant the second model added some value, but the likely upside from ensembling was limited. Because of that, I kept the final blend simple and treated it as a small incremental improvement rather than building a more complex second-stage system.

### Why leakage control mattered

Target encoding can easily make validation results look better than they really are if it is computed too broadly. I used inner-fold target encoding to reduce that risk and keep the validation signal closer to real generalization on unseen data.

In practice, this matters because an overly optimistic validation setup can make a model look strong on seen data but fail to hold up on new data. I believe this discipline likely contributed to the private leaderboard holding up slightly better than the public leaderboard, rather than the pipeline overfitting to the public split.

## Results

### Final Leaderboard Results

- Public Rank: 404 / 4142
- Private Rank: 390 / 4142
- Public Score: 0.91690
- Private Score: 0.91796

### OOF Summary

- XGBoost OOF AUC: 0.919275
- RealMLP OOF AUC: 0.918700
- OOF Correlation: 0.995972
- Best Ensemble Weight: 0.75 XGBoost / 0.25 RealMLP
- Best Ensemble OOF AUC: 0.919363

The ensemble improvement was small but directionally consistent. I view that as a useful result: it showed the secondary model added limited but real value, while confirming that the main pipeline was already capturing most of the available signal.

## Learnings

### 1. A strong model on seen data is not enough

One of the biggest takeaways from this project was that a model can look very strong on visible data and still fail to generalize if the validation setup is not handled carefully. What matters in practice is not how impressive the model looks on data it has already seen, but whether it can stay stable on unseen data. That is why I treated validation discipline as part of the modeling work itself, not just a reporting step.

### 2. Feature engineering is where simple data becomes more informative

One of the most valuable parts of this project was seeing how seemingly simple tabular data could be expressed in many different ways. By re-framing raw columns into ratio features, service summaries, prior-rate mappings, and distribution-aware features, I was able to expose more structure than the original schema showed directly. That process of turning basic inputs into richer signals is one of the parts of data science I find most interesting.

### 3. Different model families can provide different ways of seeing the same data

A major takeaway for me was not just that multiple models can improve performance, but that they can view the same dataset through different lenses. The Ridge -> XGBoost pipeline was especially interesting because it combined a more stable linear representation with a nonlinear model that could pick up more complex interactions. That modeling perspective was one of the most useful things I took away from the project.

### 4. Good projects leave room for curiosity

What I found most rewarding in this project was that every stage — from understanding the data, to shaping features, to comparing models — opened up new questions to explore. Even with a relatively simple tabular dataset, there were many ways to uncover structure, reframe the information, and see the problem from a different angle. That sense of depth and discovery is part of what makes data analysis so engaging to me.

## Repository Structure

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