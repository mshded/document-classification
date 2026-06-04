import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer, PorterStemmer
import nltk

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

from datasets import load_dataset
dataset = load_dataset("emotion")
train_texts = dataset["train"]["text"]
train_labels = dataset["train"]["label"]
test_texts = dataset["test"]["text"]
test_labels = dataset["test"]["label"]

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

# raw
def preprocess_raw(text):
    tokens = word_tokenize(text.lower())
    tokens = [t for t in tokens if t.isalpha()]
    return " ".join(tokens)

#lemmatization
def preprocess_lemma(text):
    tokens = word_tokenize(text.lower())
    tokens = [t for t in tokens if t.isalpha()]
    lemmas = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(lemmas)

# stemming
def preprocess_stem(text):
    tokens = word_tokenize(text.lower())
    tokens = [t for t in tokens if t.isalpha()]
    stems = [stemmer.stem(t) for t in tokens]
    return " ".join(stems)

# lemmatization + NJ
def preprocess_lemma_nj(text):
    tokens = word_tokenize(text.lower())
    tokens = [t for t in tokens if t.isalpha()]
    tagged = pos_tag(tokens)
    filtered = [word for word, tag in tagged if tag.startswith('N') or tag.startswith('J')]
    lemmas = [lemmatizer.lemmatize(t) for t in filtered]
    return " ".join(lemmas)

train_raw = [preprocess_raw(t) for t in train_texts]
test_raw = [preprocess_raw(t) for t in test_texts]
train_lemma = [preprocess_lemma(t) for t in train_texts]
test_lemma = [preprocess_lemma(t) for t in test_texts]
train_stem = [preprocess_stem(t) for t in train_texts]
test_stem = [preprocess_stem(t) for t in test_texts]
train_lemma_nj = [preprocess_lemma_nj(t) for t in train_texts]
test_lemma_nj = [preprocess_lemma_nj(t) for t in test_texts]


experiments = {}

# raw
vec_raw_bin = CountVectorizer(binary=True)
X_train_raw_bin = vec_raw_bin.fit_transform(train_raw)
X_test_raw_bin = vec_raw_bin.transform(test_raw)
experiments["raw_binary"] = (X_train_raw_bin, X_test_raw_bin)

vec_raw_count = CountVectorizer()
X_train_raw_count = vec_raw_count.fit_transform(train_raw)
X_test_raw_count = vec_raw_count.transform(test_raw)
experiments["raw_count"] = (X_train_raw_count, X_test_raw_count)

vec_raw_tfidf = TfidfVectorizer()
X_train_raw_tfidf = vec_raw_tfidf.fit_transform(train_raw)
X_test_raw_tfidf = vec_raw_tfidf.transform(test_raw)
experiments["raw_tfidf"] = (X_train_raw_tfidf, X_test_raw_tfidf)

# lemma
vec_lemma_bin = CountVectorizer(binary=True)
X_train_lemma_bin = vec_lemma_bin.fit_transform(train_lemma)
X_test_lemma_bin = vec_lemma_bin.transform(test_lemma)
experiments["lemma_binary"] = (X_train_lemma_bin, X_test_lemma_bin)

vec_lemma_count = CountVectorizer()
X_train_lemma_count = vec_lemma_count.fit_transform(train_lemma)
X_test_lemma_count = vec_lemma_count.transform(test_lemma)
experiments["lemma_count"] = (X_train_lemma_count, X_test_lemma_count)

vec_lemma_tfidf = TfidfVectorizer()
X_train_lemma_tfidf = vec_lemma_tfidf.fit_transform(train_lemma)
X_test_lemma_tfidf = vec_lemma_tfidf.transform(test_lemma)
experiments["lemma_tfidf"] = (X_train_lemma_tfidf, X_test_lemma_tfidf)

# stem
vec_stem_bin = CountVectorizer(binary=True)
X_train_stem_bin = vec_stem_bin.fit_transform(train_stem)
X_test_stem_bin = vec_stem_bin.transform(test_stem)
experiments["stem_binary"] = (X_train_stem_bin, X_test_stem_bin)

vec_stem_count = CountVectorizer()
X_train_stem_count = vec_stem_count.fit_transform(train_stem)
X_test_stem_count = vec_stem_count.transform(test_stem)
experiments["stem_count"] = (X_train_stem_count, X_test_stem_count)

vec_stem_tfidf = TfidfVectorizer()
X_train_stem_tfidf = vec_stem_tfidf.fit_transform(train_stem)
X_test_stem_tfidf = vec_stem_tfidf.transform(test_stem)
experiments["stem_tfidf"] = (X_train_stem_tfidf, X_test_stem_tfidf)

# lemma + NJ
vec_nj_bin = CountVectorizer(binary=True)
X_train_nj_bin = vec_nj_bin.fit_transform(train_lemma_nj)
X_test_nj_bin = vec_nj_bin.transform(test_lemma_nj)
experiments["lemma_NJ_binary"] = (X_train_nj_bin, X_test_nj_bin)

vec_nj_count = CountVectorizer()
X_train_nj_count = vec_nj_count.fit_transform(train_lemma_nj)
X_test_nj_count = vec_nj_count.transform(test_lemma_nj)
experiments["lemma_NJ_count"] = (X_train_nj_count, X_test_nj_count)

vec_nj_tfidf = TfidfVectorizer()
X_train_nj_tfidf = vec_nj_tfidf.fit_transform(train_lemma_nj)
X_test_nj_tfidf = vec_nj_tfidf.transform(test_lemma_nj)
experiments["lemma_NJ_tfidf"] = (X_train_nj_tfidf, X_test_nj_tfidf)


models = {
"DecisionTree": DecisionTreeClassifier(random_state=42),
"RandomForest": RandomForestClassifier(random_state=42),
"GradientBoosting": GradientBoostingClassifier(random_state=42),
"AdaBoost": AdaBoostClassifier(random_state=42)
}

def evaluate(model, X_train, X_test):
    model.fit(X_train, train_labels)
    preds = model.predict(X_test)
    f1_micro = f1_score(test_labels, preds, average="micro")
    f1_macro = f1_score(test_labels, preds, average="macro")
    f1_weighted = f1_score(test_labels, preds, average="weighted")
    return f1_micro, f1_macro, f1_weighted

results = []

for exp_name, (Xtr, Xte) in experiments.items():
    print(f"\nЭксперимент: {exp_name}")

    for model_name, model in models.items():
        micro, macro, weighted = evaluate(model, Xtr, Xte)

        results.append({
            "experiment": exp_name,
            "model": model_name,
            "f1_micro": micro,
            "f1_macro": macro,
            "f1_weighted": weighted
        })

        print(
            f"{model_name:17s}"
            f"micro = {micro:.4f} "
            f"macro = {macro:.4f} "
            f"weighted = {weighted:.4f}"
        )

df_results = pd.DataFrame(results)
print(df_results)
df_sorted_micro = df_results.sort_values(
    ["f1_micro", "f1_macro", "f1_weighted"],
    ascending=False
)
df_sorted_macro = df_results.sort_values(
    ["f1_macro", "f1_micro", "f1_weighted"],
    ascending=False
)
print(df_sorted_micro.head(10))
print(df_sorted_macro.head(10))
best_row = df_sorted_micro.iloc[0]
print(best_row)

# results = []

# for exp_name, (Xtr, Xte) in experiments.items():
#     for model_name, model in models.items():
#         micro, macro, weighted = evaluate(model, Xtr, Xte)

#         results.append({
#             "experiment": exp_name,
#             "model": model_name,
#             "f1_micro": micro,
#             "f1_macro": macro,
#             "f1_weighted": weighted
#         })

# df_results = pd.DataFrame(results)
# print(df_results.sort_values(["f1_micro", "f1_macro"], ascending=False))

# rf_results = []

# rf_params = [
#     {"n_estimators": 200, "max_depth": None, "max_features": "sqrt"},
#     {"n_estimators": 500, "max_depth": None, "max_features": "sqrt"},
#     {"n_estimators": 500, "max_depth": 50, "max_features": "sqrt"},
#     {"n_estimators": 500, "max_depth": None, "max_features": "log2"},
#     {"n_estimators": 700, "max_depth": None, "max_features": "sqrt"},
# ]

# for params in rf_params:
#     rf = RandomForestClassifier(
#         random_state=42,
#         n_jobs=-1,
#         **params
#     )

#     rf.fit(X_train_lemma_count, train_labels)
#     pred = rf.predict(X_test_lemma_count)

#     rf_results.append({
#         "params": str(params),
#         "f1_micro": f1_score(test_labels, pred, average="micro"),
#         "f1_macro": f1_score(test_labels, pred, average="macro"),
#         "f1_weighted": f1_score(test_labels, pred, average="weighted")
#     })

# df_rf = pd.DataFrame(rf_results)

# print("\nRandomForest parameter tuning:")
# print(df_rf.sort_values(["f1_micro", "f1_macro"], ascending=False))
