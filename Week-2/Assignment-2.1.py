import math
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import sys
sys.stdout=open('output.txt','w')
corpus = [
    'the sun is a star',
    'the moon is a satellite',
    'the sun and moon are celestial bodies'
]

docs = [doc.lower().split() for doc in corpus]

# Step 2: Vocabulary
vocab = sorted(set(word for doc in docs for word in doc))

# Step 3: Term Frequency (TF)
def compute_tf(doc):
    tf = {}
    total_terms = len(doc)
    for word in doc:
        tf[word] = tf.get(word, 0) + 1
    for word in tf:
        tf[word] /= total_terms
    return tf

tf_list = [compute_tf(doc) for doc in docs]

# Step 4: Inverse Document Frequency (IDF)
def compute_idf(docs):
    idf = {}
    total_docs = len(docs)
    for word in vocab:
        containing = sum(1 for doc in docs if word in doc)
        idf[word] = math.log(total_docs / (1 + containing)) + 1  # smooth
    return idf

idf = compute_idf(docs)

# Step 5: TF-IDF
def compute_tfidf(tf, idf):
    tfidf = {}
    for word in vocab:
        tfidf[word] = tf.get(word, 0) * idf[word]
    return tfidf

tfidf_vectors = [compute_tfidf(tf, idf) for tf in tf_list]

# Print TF-IDF vectors
for i, vec in enumerate(tfidf_vectors):
    print(f"\nDocument {i+1} TF-IDF:")
    for word in vocab:
        print(f"{word:10s}: {vec[word]:.4f}")

# -----------------------------
# 5. Count Vectorizer (sklearn)
# -----------------------------
print("\n==== Count Vectorizer (sklearn) ====")
count_vectorizer = CountVectorizer()
count_matrix = count_vectorizer.fit_transform(corpus)
print(count_vectorizer.get_feature_names_out())
print(count_matrix.toarray())


tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
print(tfidf_vectorizer.get_feature_names_out())
print(tfidf_matrix.toarray())