# Sports vs Politics Classifier

This page documents a binary text classification project that separates sports and politics articles using classic machine learning models and standard text features.

## Project Overview
- **Task:** Classify documents as Sports or Politics
- **Dataset:** 20 Newsgroups subset
  - Sports: `rec.sport.baseball`, `rec.sport.hockey`
  - Politics: `talk.politics.guns`, `talk.politics.mideast`, `talk.politics.misc`
- **Feature representations:** Bag of Words, TF-IDF (unigrams and bigrams)
- **Models compared:** Multinomial Naive Bayes, Logistic Regression, Linear SVM

## Summary of Results
Best configuration:
- **BoW bigrams + Naive Bayes**
- **Accuracy:** 0.9602 (test set)

Confusion matrix for the best model:
- Sports → Sports: 456
- Sports → Politics: 42
- Politics → Sports: 4
- Politics → Politics: 653

## How to Run
Open the notebook in [prob4/Q4.ipynb](prob4/Q4.ipynb) and run the single code cell. It will:
1. Download the dataset
2. Build features
3. Train and evaluate three models
4. Print a comparison table
5. Save plots for accuracy comparison and confusion matrix

## Files
- **Report:** [prob4/report.md](prob4/report.md)
- **Notebook:** [prob4/Q4.ipynb](prob4/Q4.ipynb)
- **Plots:** `comparison_barplot.png`, `confusion_matrix.png`

## Notes and Limitations
- The dataset is from 20 Newsgroups and may not reflect modern news or social media.
- Models use lexical features only; they do not capture deep semantics.
- Linear SVM showed a convergence warning for bigrams; increasing iterations may help.