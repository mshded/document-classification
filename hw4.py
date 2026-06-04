import nltk
import string
from datasets import load_dataset

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')

dataset1 = load_dataset("emotion")

print("DATASET: emotion")

for i in range(5):
    text = dataset1["train"][i]["text"]

    print(f"\nТекст {i+1} (RAW):")
    print(text)

    tokens = nltk.word_tokenize(text.lower())
    tokens = [t for t in tokens if t not in string.punctuation]
    tokens = [t for t in tokens if t.isalpha()]
    tokens = [t for t in tokens if t not in ["i", "im", "ive", "id"]]
    tokens = [t for t in tokens if t not in ["didnt", "dont", "cant", "wont", "isnt", "arent", "wasnt", "werent", "havent", "hasnt", "hadnt"]]
    tagged_tokens = nltk.pos_tag(tokens)

    # 4.1 N + J
    tokens_NJ = [word for word, tag in tagged_tokens if tag.startswith('N') or tag.startswith('J')]

    # 4.2 N + J + V
    tokens_NJV = [word for word, tag in tagged_tokens if tag.startswith('N') or tag.startswith('J') or tag.startswith('V')]

    print("\nNouns+Adjectives:")
    print(tokens_NJ)

    print("\nNouns+Adjectives+Verbs:")
    print(tokens_NJV)

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

    print(f"\nТекст {i+1} (RAW:")
    print(text[:500] + ("..." if len(text) > 500 else ""))

    tokens = nltk.word_tokenize(text.lower())
    tokens = [t for t in tokens if t not in string.punctuation]
    tokens = [t for t in tokens if t.isalpha()]
    tokens = [t for t in tokens if t not in ["i", "im", "ive", "id"]]
    tokens = [t for t in tokens if t != "please"]
    tokens = [t for t in tokens if t not in ["didnt", "dont", "cant", "wont", "isnt", "arent", "wasnt", "werent", "havent", "hasnt", "hadnt"]]

    tagged_tokens = nltk.pos_tag(tokens)

    tokens_NJ = [word for word, tag in tagged_tokens if tag.startswith('N') or tag.startswith('J')]
    tokens_NJV = [word for word, tag in tagged_tokens if tag.startswith('N') or tag.startswith('J') or tag.startswith('V')]

    print("\nNouns+Adjectives:")
    print(tokens_NJ[:60])

    print("\nNouns+Adjectives+Verbs:")
    print(tokens_NJV[:60])