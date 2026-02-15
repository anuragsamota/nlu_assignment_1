# Binary text classification for sports vs politics
# Compares three common ML models

import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


# Clean and normalize input text
def normalize_text(raw_text):
    lowered = raw_text.lower()
    cleaned = re.sub(r'[^a-z\s]', ' ', lowered)
    compact = re.sub(r'\s+', ' ', cleaned)
    tokens = compact.strip().split()
    tokens = [tok for tok in tokens if len(tok) > 2]
    return " ".join(tokens)


# Load the dataset and build labels
def build_corpus():

    topic_groups = [
        "rec.sport.baseball",
        "rec.sport.hockey",
        "talk.politics.guns",
        "talk.politics.misc",
        "talk.politics.mideast"
    ]

    dataset = fetch_20newsgroups(
        subset="all",
        categories=topic_groups,
        remove=("headers", "footers", "quotes"),
        random_state=50
    )

    documents = [normalize_text(text) for text in dataset.data]

    targets = np.array([
        0 if "sport" in dataset.target_names[label] else 1
        for label in dataset.target
    ])

    print("Total Samples:", len(documents))
    print("Sports:", np.sum(targets == 0))
    print("Politics:", np.sum(targets == 1))

    return documents, targets


# Train a model and report metrics
def score_model(title, featurizer, estimator, train_x, test_x, train_y, test_y):

    train_vec = featurizer.fit_transform(train_x)
    test_vec = featurizer.transform(test_x)

    estimator.fit(train_vec, train_y)
    predictions = estimator.predict(test_vec)

    acc = accuracy_score(test_y, predictions)
    f1 = f1_score(test_y, predictions, average="weighted")

    print(f"\n========== {title} ==========")
    print("Accuracy:", round(acc, 4))
    print("Weighted F1 Score:", round(f1, 4))
    print(classification_report(test_y, predictions,
                                target_names=["Sports", "Politics"]))

    matrix = confusion_matrix(test_y, predictions)

    plt.figure(figsize=(4, 3))
    sns.heatmap(matrix, annot=True, fmt="d",
                cmap="viridis",
                xticklabels=["Sports", "Politics"],
                yticklabels=["Sports", "Politics"])
    plt.title(title)
    plt.show()

    return acc


# End-to-end pipeline

texts, labels = build_corpus()

X_train, X_test, y_train, y_test = train_test_split(
    texts,
    labels,
    test_size=0.3,
    stratify=labels,
    random_state=50
)

scores = {}

# Model 1: Bag of Words + Naive Bayes
nb_acc = score_model(
    "BoW + Naive Bayes",
    CountVectorizer(stop_words="english", max_features=8000),
    MultinomialNB(alpha=0.7),
    X_train, X_test, y_train, y_test
)
scores["Naive Bayes"] = nb_acc


# Model 2: TF-IDF + Logistic Regression
lr_acc = score_model(
    "TF-IDF + Logistic Regression",
    TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_df=0.95),
    LogisticRegression(max_iter=1500),
    X_train, X_test, y_train, y_test
)
scores["Logistic Regression"] = lr_acc


# Model 3: TF-IDF + Linear SVM
svm_acc = score_model(
    "TF-IDF + Linear SVM",
    TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_df=0.95),
    LinearSVC(C=1.2),
    X_train, X_test, y_train, y_test
)
scores["Linear SVM"] = svm_acc


# Plot a simple accuracy comparison
plt.figure(figsize=(6, 4))
chart_bars = plt.bar(scores.keys(),
                     scores.values(),
                     color=["#4C72B0", "#55A868", "#C44E52"])

plt.ylabel("Accuracy")
plt.title("Comparison of Three ML Models")
plt.ylim(0.85, 1.0)

for bar in chart_bars:
    value = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2,
             value + 0.003,
             f"{value:.3f}",
             ha='center')

plt.show()


# Select and train the best model
best_name = max(scores, key=scores.get)
print("\nBest Model Selected:", best_name)

if best_name == "Naive Bayes":
    chosen_vectorizer = CountVectorizer(stop_words="english", max_features=8000)
    chosen_model = MultinomialNB(alpha=0.7)

elif best_name == "Logistic Regression":
    chosen_vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_df=0.95)
    chosen_model = LogisticRegression(max_iter=1500)

else:
    chosen_vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_df=0.95)
    chosen_model = LinearSVC(C=1.2)

full_matrix = chosen_vectorizer.fit_transform(texts)
chosen_model.fit(full_matrix, labels)


# Interactive prediction loop
print("\nInteractive Classification Mode")
print("Type 'exit' to stop")

feature_vocab = chosen_vectorizer.get_feature_names_out()

while True:

    user_text = input("\nEnter a sentence: ")

    if user_text.lower() == "exit":
        print("Exiting classifier...")
        break

    if not user_text.strip():
        print("Please enter valid text.")
        continue

    normalized = normalize_text(user_text)
    vectorized = chosen_vectorizer.transform([normalized])
    pred_label = chosen_model.predict(vectorized)[0]

    readable_label = "Sports" if pred_label == 0 else "Politics"
    print("Predicted Category:", readable_label)

    # Show influential words for linear models
    if hasattr(chosen_model, "coef_"):
        weights = chosen_model.coef_[0]
        nonzero_idx = vectorized.nonzero()[1]

        print("Influential Words:")
        for idx in nonzero_idx[:10]:
            weight = weights[idx]
            term = feature_vocab[idx]
            if weight > 0:
                print(f"{term} -> Politics")
            else:
                print(f"{term} -> Sports")
