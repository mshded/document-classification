import nltk
import string
import re
from datasets import load_dataset

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt_tab')

url_pattern = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
html_tag_pattern = re.compile(r"<[^>]+>")   
entities = {
    "&quot;": " ",
    "&amp;": " ",
    "&lt;": " ",
    "&gt;": " ",
    "&#39;": " ",
    "&nbsp;": " "
}

dataset1 = load_dataset("emotion")
print("DATASET: emotion")

for i in range(5):
    text = dataset1["train"][i]["text"]

    print(f"\nТекст {i+1} (RAW):")
    print(text)

    # токенизация 
    tokens_raw = nltk.word_tokenize(text)
    print("\nТокены RAW:")
    print(tokens_raw)

    # чистка текста
    cleaned = text.lower()
    cleaned = url_pattern.sub(" ", cleaned)
    cleaned = html_tag_pattern.sub(" ", cleaned)

    for k, v in entities.items():
        cleaned = cleaned.replace(k, v)

    # удаление пунктуации ДО токенизации
    cleaned_no_punct = cleaned.translate(str.maketrans('', '', string.punctuation))
    tokens_clean = nltk.word_tokenize(cleaned_no_punct)
    tokens_clean = [t for t in tokens_clean if t.isalpha()]

    print("\nТекст (CLEANED):")
    print(cleaned_no_punct)

    print("\nТокены CLEANED:")
    print(tokens_clean)

print("DATASET: 20_newsgroups")

categories = [
    'comp.sys.ibm.pc.hardware',
    'comp.sys.mac.hardware',
    'comp.graphics',
    'comp.windows.x'
]

dataset2 = load_dataset("SetFit/20_newsgroups")

train = dataset2["train"].filter(
    lambda example: example["label_text"] in categories
)

for i in range(5):
    text = train[i]["text"]

    print(f"\nТекст {i+1} (RAW):")
    print(text[:800] + ("..." if len(text) > 800 else ""))

    # токенизация 
    tokens_raw = nltk.word_tokenize(text)
    print("\nТокены RAW:")
    print(tokens_raw[:60])

    # чистка текста
    cleaned = text.lower()
    cleaned = url_pattern.sub(" ", cleaned)
    cleaned = html_tag_pattern.sub(" ", cleaned)

    for k, v in entities.items():
        cleaned = cleaned.replace(k, v)

    # удаление пунктуации ДО токенизации
    cleaned_no_punct = cleaned.translate(str.maketrans('', '', string.punctuation))
    tokens_clean = nltk.word_tokenize(cleaned_no_punct)
    tokens_clean = [t for t in tokens_clean if t.isalpha()]

    print("\nТекст (CLEANED):")
    print(cleaned_no_punct[:800] + ("..." if len(cleaned_no_punct) > 800 else ""))

    print("\nТокены CLEANED:")
    print(tokens_clean[:60])