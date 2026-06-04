import nltk
import string
from datasets import load_dataset
from nltk.corpus import wordnet

# ЗАДАНИЕ 1.1 - сравнение стемминга и лемматизации на одном тексте

# Загрузка нужных данных NLTK:
nltk.download('punkt') # Для токенизации
nltk.download('wordnet') # Для лемматизации
nltk.download('omw-1.4') # Часто нужно для корректной лемматизации (WordNet)
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')

# Загружаем датасет ag_news (Hugging Face Datasets)
dataset1 = load_dataset("Emotion")

# Берём первый пример из обучающей выборки
first_example = dataset1["train"][0]
first_text = first_example["text"]
print("Исходный текст:")
print(first_text)

# Шаг 1: Токенизация без доп. очистки
tokens = nltk.word_tokenize(first_text)
print("\nТокены (без дополнительной очистки):")
print(tokens)

# Функция предобработки со стеммингом
def preprocess_with_stemming(text):
    # Приведение к нижнему регистру
    text = text.lower()
    # Удаление знаков препинания
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Токенизация
    tokens = nltk.word_tokenize(text)
    # Применение стемминга
    stemmer = nltk.PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return stemmed_tokens

# Функция предобработки с лемматизацией
def preprocess_with_lemmatization(text):
    # Приведение к нижнему регистру
    text = text.lower()
    # Удаление знаков препинания
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Токенизация
    tokens = nltk.word_tokenize(text)
    # Применение лемматизации
    lemmatizer = nltk.WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return lemmatized_tokens

# Сравним результаты стемминга и лемматизации
stemmed_result = preprocess_with_stemming(first_text)
lemmatized_result = preprocess_with_lemmatization(first_text)
print("\nТокены после стемминга:")
print(stemmed_result)
print("\nТокены после лемматизации:")
print(lemmatized_result)


# ЗАДАНИЕ 1.2 - сравнение стемминга и лемматизации на 5 текстах

def find_differences(text):
    stemmed = preprocess_with_stemming(text)
    lemmatized = preprocess_with_lemmatization(text)

    differences = []

    for original, stem, lemma in zip(
        nltk.word_tokenize(text.lower().translate(str.maketrans('', '', string.punctuation))),
        stemmed,
        lemmatized
    ):
        if stem != lemma:
            differences.append((original, stem, lemma))

    return differences


for i in range(5):
    text = dataset1["train"][i]["text"]
    print(f"\nТекст {i+1}:")
    print(text)

    diffs = find_differences(text)

    print("\nРазличия (слово | стемминг | лемматизация):")
    for diff in diffs[:10]:  # выводим первые 10 различий
        print(diff)

# ЗАДАНИЕ 1.3 - сравнение удаления пунктуации до и после токенизации
texts = [dataset1["train"][i]["text"] for i in range(5)]

for i, text in enumerate(texts):
    print(f"\nТекст {i+1}:")
    print(text)

    tokens = nltk.word_tokenize(text)

    print("\nТокены после word_tokenize:")
    print(tokens)

print("\nУдаление пунктуации ДО токенизации")
for i, text in enumerate(texts):
    text_clean = text.translate(str.maketrans('', '', string.punctuation))
    tokens = nltk.word_tokenize(text_clean)

    print(f"\nТекст {i+1}:", text)
    print("После очистки:", tokens)

print("\nУдаление пунктуации ПОСЛЕ токенизации")
for i, text in enumerate(texts):
    tokens = nltk.word_tokenize(text)

    filtered_tokens = [
        token for token in tokens
        if token not in string.punctuation
    ]

    print(f"\nТекст {i+1}:", text)
    print("После фильтрации:", filtered_tokens)


# ЗАДАНИЕ 1.4 - то же самое для Newsgroups

categories = [
    'comp.sys.ibm.pc.hardware',
    'comp.sys.mac.hardware',
    'comp.graphics',
    'comp.windows.x'
]

dataset2 = load_dataset("SetFit/20_newsgroups")
print(dataset2)

train = dataset2["train"].filter(
    lambda example: example["label_text"] in categories
)

test = dataset2["test"].filter(
    lambda example: example["label_text"] in categories
)

for i in range(5):
    text = train[i]["text"]

    print(f"\nТекст {i+1}:")
    print(text)

    # 1️. Обычная токенизация
    tokens = nltk.word_tokenize(text)
    print("\nТокены (без очистки):")
    print(tokens)

    # 2️. Стемминг
    stemmed = preprocess_with_stemming(text)
    print("\nСтемминг:")
    print(stemmed[:15])

    # 3️. Лемматизация
    lemmatized = preprocess_with_lemmatization(text)
    print("\nЛемматизация:")
    print(lemmatized[:15])

    # 4️. Различия
    diffs = find_differences(text)
    print("\nРазличия (слово | стемминг | лемматизация):")
    for diff in diffs[:10]:
        print(diff)

    # 5️. Удаление пунктуации ДО токенизации
    text_clean = text.translate(str.maketrans('', '', string.punctuation))
    tokens_clean = nltk.word_tokenize(text_clean)
    print("\nБез пунктуации (до токенизации):")
    print(tokens_clean[:15])

    # 6️. Удаление пунктуации ПОСЛЕ токенизации
    filtered_tokens = [t for t in tokens if t not in string.punctuation]
    print("\nБез пунктуации (после токенизации):")
    print(filtered_tokens[:15])

