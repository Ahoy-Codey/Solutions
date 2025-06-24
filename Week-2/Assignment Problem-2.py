import pandas as pd
import numpy as np
import re
import nltk
import contractions
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import gensim.downloader as api

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt_tab')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = text.lower()
    text = contractions.fix(text)
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r"@\w+|#\w+", '', text)
    text = re.sub(r"[^\w\s]", '', text)
    text = re.sub(r"\d+", '', text)
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return tokens

df = pd.read_csv("Tweets.csv")[['text', 'airline_sentiment']]
df.dropna(inplace=True)
df['Tokens'] = df['text'].apply(preprocess)

w2v_model = api.load("word2vec-google-news-300")

def get_avg_vector(tokens, model, vector_size=100):
    vectors = [model[word] for word in tokens if word in model]
    return np.mean(vectors, axis=0) if vectors else np.zeros(vector_size)

df['Vector'] = df['Tokens'].apply(lambda x: get_avg_vector(x, w2v_model))
df = df[df['Vector'].apply(lambda x: np.any(x))]

X = np.stack(df['Vector'].values)
y = df['airline_sentiment']  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

model = LogisticRegression(max_iter=1000, multi_class='multinomial')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, y_pred))

def predict_tweet_sentiment(model, w2v_model, tweet):
    tokens = preprocess(tweet)
    vector = get_avg_vector(tokens, w2v_model).reshape(1, -1)
    return model.predict(vector)[0]

print(predict_tweet_sentiment(model, w2v_model, "American Airlines has the worst delays ever"))
print(predict_tweet_sentiment(model, w2v_model, "JetBlue service was fantastic!"))
print(predict_tweet_sentiment(model, w2v_model, "My flight was fine, nothing special."))
