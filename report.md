# Sports vs Politics Text Classification Report

### Name: Anurag Samota
### Roll No.: M25CE007

## 1. Abstract
This report describes a binary text classifier that separates sports posts from politics posts using a subset of the 20 Newsgroups dataset. I compare three standard models (Multinomial Naive Bayes, Logistic Regression, and Linear SVM) with classic lexical features. The workflow covers data collection, preprocessing, feature extraction, training, and evaluation. Based on the latest run of the provided code, the best configuration is Bag of Words with Multinomial Naive Bayes, achieving 0.9502 test accuracy and a weighted F1 of 0.9499. I summarize the setup, results, and limitations, then discuss likely error patterns and directions for improvement.

## 2. Introduction
Text classification is a core NLP task used for filtering, routing, and organizing large volumes of text. A binary topic task is a clean way to study how lexical features and linear classifiers behave when the classes are distinct in meaning but still share some vocabulary.

In this project, I build a sports vs politics classifier using 20 Newsgroups posts. These two topics are different in real-world meaning, but they overlap in terms like organizations, public events, and named entities. The goals are:
1. Build a reliable end-to-end pipeline for binary text classification.
2. Compare classic models under matched feature settings.
3. Identify the strongest configuration using measured results.
4. Provide a reproducible baseline for future improvements.

## 3. Background and Motivation
Topic classification is one of the earliest success stories of statistical NLP. It is a good testbed because simple lexical features often carry enough signal to separate topics, yet it still includes real-world issues such as class imbalance, noisy inputs, and topic overlap. The 20 Newsgroups dataset is a standard benchmark, so it allows results to be interpreted in relation to prior work.

The sports vs politics split is especially interesting because many posts contain named entities, public events, or global references. This creates realistic ambiguity and ensures the task is not trivial, even though the overall performance remains high with simple models.

## 4. Dataset and Preprocessing
**Source.** The dataset is drawn from the 20 Newsgroups corpus using scikit-learn. The selected categories are:
- Sports: `rec.sport.baseball`, `rec.sport.hockey`
- Politics: `talk.politics.guns`, `talk.politics.misc`, `talk.politics.mideast`

**Filtering.** The loader removes headers, footers, and quoted replies. This reduces boilerplate and keeps the focus on main message content rather than signatures or repeated quoted material.

**Text cleaning.** Each document is normalized by:
- Lowercasing
- Removing non-alphabetic characters
- Collapsing whitespace
- Dropping very short tokens (length <= 2)

**Dataset size and balance (latest run).**
- Total samples: **4,618**
- Sports: **1,993**
- Politics: **2,625**

The split is stratified to preserve these proportions.

## 5. Experimental Setup
### 5.1 Train/Test Split
- Train: **70%**
- Test: **30%**

A stratified split ensures the class proportions are consistent across train and test sets.

### 5.2 Feature Representations
Two classic lexical feature sets are used:
- **Bag of Words (BoW):** count-based features with English stop words removed and a max vocabulary size of 8,000.
- **TF-IDF:** unigrams and bigrams with max document frequency 0.95 and English stop words removed.

These choices balance simplicity and performance. BoW is effective for Naive Bayes, while TF-IDF often improves linear models by dampening overly common terms.

### 5.3 Models
Three classifiers are evaluated:
1. **Multinomial Naive Bayes**
2. **Logistic Regression**
3. **Linear SVM (LinearSVC)**

All models are trained on the same split for a fair comparison. Hyperparameters are kept close to defaults with minor adjustments to ensure convergence.

### 5.4 Evaluation Metrics
The script reports:
- Accuracy
- Weighted F1
- Precision, recall, and F1 per class

Accuracy summarizes overall correctness, while weighted F1 accounts for class imbalance by weighting class-specific F1 scores by support.

## 6. Results (Latest Execution)
The following metrics are taken directly from the most recent run of the script:

| Model | Features | Accuracy | Weighted F1 |
|---|---|---:|---:|
| Naive Bayes | BoW | **0.9502** | **0.9499** |
| Logistic Regression | TF-IDF (1-2 grams) | 0.9278 | 0.9268 |
| Linear SVM | TF-IDF (1-2 grams) | 0.9481 | 0.9477 |

**Best model.** The best performer is **Naive Bayes with Bag of Words**, which achieved 0.9502 accuracy and 0.9499 weighted F1.

### 6.1 Class-Level Metrics
From the latest run, the per-class results are:

**BoW + Naive Bayes**
- Sports: precision 0.98, recall 0.90, F1 0.94
- Politics: precision 0.93, recall 0.99, F1 0.96

**TF-IDF + Logistic Regression**
- Sports: precision 0.99, recall 0.84, F1 0.91
- Politics: precision 0.89, recall 1.00, F1 0.94

**TF-IDF + Linear SVM**
- Sports: precision 0.99, recall 0.89, F1 0.94
- Politics: precision 0.92, recall 0.99, F1 0.96

These numbers show that the models are consistently strong on politics recall, while sports recall tends to be slightly lower. This aligns with the dataset imbalance and topic overlap.

## 7. Analysis and Discussion
### 7.1 Why Naive Bayes Wins Here
Multinomial Naive Bayes works directly with count data and assumes conditional independence of features. For topic classification, this often performs surprisingly well because individual terms and short phrases carry strong class-specific signals. Since the data is relatively clean and the task is topical, Naive Bayes benefits from its simplicity and robustness.

### 7.2 Linear Models with TF-IDF
Both Logistic Regression and Linear SVM use TF-IDF features. SVM is slightly stronger here, likely because its margin-based objective handles sparse, high-dimensional data well. Logistic Regression is competitive but a bit lower on recall for sports. The difference is small, suggesting that hyperparameter tuning could narrow the gap.

### 7.3 Error Patterns
The main error trends likely include:
- **Mixed-topic posts:** Some posts discuss sports in political contexts or vice versa.
- **Overlapping entities:** Names of countries, organizations, or public events appear in both domains.
- **Short posts:** Short or informal messages provide few strong lexical cues.

Given the high politics recall, a common failure mode is sports posts being labeled as politics when there is insufficient evidence to support the sports class.

## 8. Limitations
1. **Domain specificity.** The dataset is a narrow slice of 20 Newsgroups and may not reflect modern news or social media text.
2. **Surface-level features.** BoW and TF-IDF focus on lexical cues and do not capture deeper semantics.
3. **Class imbalance.** Politics has more samples, which can bias predictions when evidence is weak.
4. **Mixed-topic posts.** Some documents naturally blend sports and politics, making strict separation difficult.
5. **Limited hyperparameter tuning.** Only light tuning is applied; more extensive search might improve performance.

## 9. Future Work
Possible extensions include:
- **Hyperparameter tuning.** Tune `alpha` for Naive Bayes and `C` for Logistic Regression and SVM.
- **Richer features.** Add character n-grams, part-of-speech patterns, or topic model features.
- **Semantic embeddings.** Use transformer embeddings with a lightweight classifier.
- **Stronger evaluation.** Add cross-validation, macro-F1, and ROC-AUC for a more balanced view.
- **Data expansion.** Include more categories or newer datasets to test generalization.

## 10. Conclusion
This project demonstrates that a clean pipeline with classic models can achieve strong results on a sports vs politics classification task. The best configuration from the current code is Bag of Words with Multinomial Naive Bayes, reaching 0.9502 accuracy and 0.9499 weighted F1 on the test set. These results provide a solid baseline and a starting point for future improvements using richer features or semantic embeddings.

## 11. Reproducibility Notes
- Dataset: 20 Newsgroups (`rec.sport.*`, `talk.politics.*`)
- Preprocessing: lowercasing, non-alphabetic removal, stop words, short-token removal
- Split: stratified 70/30 train/test
- Metrics: accuracy, weighted F1, per-class precision and recall

## 12. Appendix: Run Summary
The latest execution produced:
- Total Samples: 4,618
- Sports: 1,993
- Politics: 2,625
- Best Model Selected: Naive Bayes

This summary is included to make the report self-contained and consistent with the recorded outputs.
