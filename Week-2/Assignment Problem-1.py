import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import gensim.downloader as api

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()  
    text = re.sub(r'[^a-z\s]', '', text)  
    tokens = text.split()  
    return [word for word in tokens if word not in stop_words]  

df = pd.read_csv('spam.csv', encoding='latin-1')[['v1', 'v2']]
df.columns = ['Label', 'Message']
df['Tokens'] = df['Message'].apply(preprocess)

print("Loading Word2Vec model...")
w2v_model = api.load("word2vec-google-news-300")

def get_avg_vector(tokens, model, size=300):
    vectors = [model[word] for word in tokens if word in model]
    return np.mean(vectors, axis=0) if vectors else np.zeros(size)

df['Vector'] = df['Tokens'].apply(lambda x: get_avg_vector(x, w2v_model))
df = df[df['Vector'].apply(lambda x: np.any(x))]  # Drop rows with all-zero vectors
X = np.stack(df['Vector'].values)
y = df['Label'].map({'ham': 0, 'spam': 1}).values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, y_pred))

def predict_message_class(model, w2v_model, message):
    tokens = preprocess(message)
    vector = get_avg_vector(tokens, w2v_model).reshape(1, -1)
    return 'spam' if model.predict(vector)[0] == 1 else 'ham'

print(predict_message_class(model, w2v_model, "Win a free iPhone now!"))
print(predict_message_class(model, w2v_model, "Let's meet at 6pm today."))
