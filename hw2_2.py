import nltk
import string
from datasets import load_dataset
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords, wordnet

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('stopwords')

stop_words = set(stopwords.words("english"))

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

lemmatizer = nltk.WordNetLemmatizer()
stemmer = nltk.PorterStemmer()

# стемминг + удаление стоп-слов
def preprocess_with_stemming(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)

    stemmed_tokens = [
        stemmer.stem(token)
        for token in tokens
        if token not in string.punctuation
    ]

    filtered_tokens = [
        token for token in stemmed_tokens
        if token.isalpha() and token not in stop_words
    ]

    return filtered_tokens

# лемматизация + POS + стоп-слова
def preprocess_with_lemmatization(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)

    tagged = nltk.pos_tag(tokens)

    lemmatized_tokens = [
        lemmatizer.lemmatize(token, get_wordnet_pos(tag))
        for token, tag in tagged
        if token not in string.punctuation
    ]

    filtered_tokens = [
        token for token in lemmatized_tokens
        if token.isalpha() and token not in stop_words
    ]

    return filtered_tokens

# ЗАДАНИЕ 1 - сравнение стемминга и лемматизации

print("ЗАДАНИЕ 1")

dataset1 = load_dataset("emotion")

for i in range(3):
    text = dataset1["train"][i]["text"]
    print(f"\nТекст {i+1}:")
    print(text)

    stemmed = preprocess_with_stemming(text)
    lemmatized = preprocess_with_lemmatization(text)

    print("\nСтемминг:")
    print(stemmed)

    print("\nЛемматизация:")
    print(lemmatized)

# ЗАДАНИЕ 2 - фильтрация стоп-слов

print("ЗАДАНИЕ 2")

for i in range(3):
    text = dataset1["train"][i]["text"]
    result = preprocess_with_lemmatization(text)

    print(f"\nТекст {i+1}:")
    print(result)

# ЗАДАНИЕ 3 - векторизация 0/1

print("ЗАДАНИЕ 3")

raw_docs = [dataset1["train"][i]["text"] for i in range(200)]

# вариант 1 - лемматизация заранее, потом векторизация

lemmatized_docs = [preprocess_with_lemmatization(doc) for doc in raw_docs]
docs_as_strings = [' '.join(tokens) for tokens in lemmatized_docs]

vectorizer1 = CountVectorizer(binary=True)
X1 = vectorizer1.fit_transform(docs_as_strings)

print("\nВариант 1 (лемматизация заранее):")
print("Размер матрицы:", X1.shape)
print("Размер словаря:", len(vectorizer1.vocabulary_))

# вариант 2 - лемматизация внутри CountVectorizer

vectorizer2 = CountVectorizer(
    binary=True,
    tokenizer=preprocess_with_lemmatization,
    lowercase=False,
    token_pattern=None
)

X2 = vectorizer2.fit_transform(raw_docs)

print("\nВариант 2 (лемматизация внутри vectorizer):")
print("Размер матрицы:", X2.shape)
print("Размер словаря:", len(vectorizer2.vocabulary_))

# вариант 3 - без лемматизации, только удаление пунктуации и стоп-слов, но внутри CountVectorizer 

vectorizer3 = CountVectorizer(binary=True)
X3 = vectorizer3.fit_transform(raw_docs)

print("\nВариант 3 (без лемматизации):")
print("Размер матрицы:", X3.shape)
print("Размер словаря:", len(vectorizer3.vocabulary_))

print("\nСравнение размеров словаря:")
print("Вариант 1:", len(vectorizer1.vocabulary_))
print("Вариант 2:", len(vectorizer2.vocabulary_))
print("Вариант 3:", len(vectorizer3.vocabulary_))