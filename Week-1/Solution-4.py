import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

np.random.seed(42)

positive_phrases = [
    "This movie was absolutely brilliant! A masterpiece from start to finish.",
    "Highly recommend watching this film. The acting was superb and the story captivating.",
    "A truly heartwarming and inspiring movie. I loved every minute of it.",
    "The special effects were breathtaking and the plot was incredibly engaging.",
    "An absolute must-see! It kept me on the edge of my seat.",
    "One of the best films I've seen all year. The director did an amazing job.",
    "Fantastic performances and a brilliant screenplay. Pure cinematic gold.",
    "I was thoroughly entertained. This movie exceeded all my expectations.",
    "Simply stunning! The visuals and soundtrack were perfectly harmonized.",
    "A powerful and emotional journey. I was moved to tears.",
    "Loved the unique storyline and the unexpected twists.",
    "This film is a triumph of storytelling. A true classic in the making.",
    "The chemistry between the lead actors was undeniable.",
    "Every scene was meticulously crafted. A joy to watch.",
    "Incredible direction and a truly memorable experience.",
    "So much fun! I laughed out loud multiple times.",
    "A cinematic gem that will stay with you long after it ends.",
    "The pacing was perfect, never a dull moment.",
    "Absolutely captivating. I couldn't take my eyes off the screen.",
    "A refreshing take on the genre, truly innovative.",
    "Brilliantly written and perfectly executed.",
    "I'm still thinking about it days later. So impactful.",
    "The production design was out of this world.",
    "A perfect blend of action, drama, and humor.",
    "This film deserves all the awards.",
    "An unforgettable performance by the entire cast.",
    "Such a compelling narrative, I was hooked.",
    "The climax was incredibly satisfying.",
    "A visual feast and an intellectual treat.",
    "I'd watch it again in a heartbeat.",
    "Masterful storytelling and incredible cinematography.",
    "This movie has it all: great plot, great acting, great everything.",
    "Pure genius! A groundbreaking film.",
    "Exceeded expectations in every way possible.",
    "A true cinematic experience.",
    "The character development was outstanding.",
    "I was completely immersed in this world.",
    "A triumph of creativity and vision.",
    "Highly engaging and thought-provoking.",
    "This film just made my day better.",
    "So glad I saw this one in theaters.",
    "A shining example of modern filmmaking.",
    "The ending left me speechless in a good way.",
    "Every detail was perfect.",
    "Couldn't ask for a better movie night.",
    "This is what cinema is all about.",
    "Phenomenal from beginning to end.",
    "A masterpiece of its kind.",
    "Left me feeling hopeful and inspired.",
    "Such a powerful message conveyed beautifully."
]

negative_phrases = [
    "This movie was an absolute disaster. A complete waste of time and money.",
    "I regret watching this film. The acting was terrible and the story was incoherent.",
    "A truly boring and uninspired movie. I wanted it to end quickly.",
    "The special effects were poor and the plot was incredibly dull.",
    "An absolute must-avoid! It put me to sleep.",
    "One of the worst films I've seen all year. The director clearly missed the mark.",
    "Terrible performances and a dreadful screenplay. Pure cinematic trash.",
    "I was thoroughly disappointed. This movie fell short of all expectations.",
    "Simply awful! The visuals and soundtrack were jarringly bad.",
    "A tiresome and emotionless experience. I felt nothing.",
    "Hated the predictable storyline and the obvious twists.",
    "This film is a failure of storytelling. Not a classic in the making.",
    "The chemistry between the lead actors was non-existent.",
    "Every scene felt forced and unnatural. A chore to watch.",
    "Incredible direction in the wrong direction. A truly forgettable experience.",
    "So boring! I checked my phone multiple times.",
    "A cinematic flop that should be avoided.",
    "The pacing was terrible, it dragged on endlessly.",
    "Absolutely uncaptivating. I couldn't wait for it to be over.",
    "A tired rehash of old tropes, nothing innovative.",
    "Poorly written and badly executed.",
    "I forgot about it as soon as it ended. So unimpactful.",
    "The production design was cheap and unconvincing.",
    "A terrible blend of disjointed scenes.",
    "This film deserves no awards.",
    "An unconvincing performance by most of the cast.",
    "Such a convoluted narrative, I was lost.",
    "The climax was incredibly unsatisfying.",
    "A visual mess and an intellectual vacuum.",
    "I wouldn't watch it again even if paid.",
    "Amateurish storytelling and bland cinematography.",
    "This movie has nothing: bad plot, bad acting, bad everything.",
    "Pure mediocrity! A forgettable film.",
    "Failed to meet expectations in every way possible.",
    "A truly painful cinematic experience.",
    "The character development was non-existent.",
    "I was completely detached from this world.",
    "A failure of creativity and vision.",
    "Highly unengaging and utterly pointless.",
    "This film just made my day worse.",
    "So glad I didn't see this one in theaters.",
    "A sad example of modern filmmaking.",
    "The ending left me utterly confused in a bad way.",
    "Every detail was off.",
    "Couldn't ask for a worse movie night.",
    "This is not what cinema is all about.",
    "Unbearable from beginning to end.",
    "A disaster of its kind.",
    "Left me feeling irritated and disappointed.",
    "Such a weak message conveyed poorly."
]

positive_reviews = np.random.choice(positive_phrases, 50, replace=True).tolist()
positive_sentiments = ['positive'] * 50

negative_reviews = np.random.choice(negative_phrases, 50, replace=True).tolist()
negative_sentiments = ['negative'] * 50

all_reviews = positive_reviews + negative_reviews
all_sentiments = positive_sentiments + negative_sentiments

df = pd.DataFrame({
    'Review': all_reviews,
    'Sentiment': all_sentiments
})

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

vectorizer = CountVectorizer(max_features=500, stop_words='english')
X = vectorizer.fit_transform(df['Review'])
y = df['Sentiment'] 


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)


model = MultinomialNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy on Test Set: {accuracy:.2f}")

def predict_review_sentiment(model, vectorizer, review):

    review_vectorized = vectorizer.transform([review])
    prediction = model.predict(review_vectorized)
    return prediction[0]


test_review_positive = "This film was incredibly captivating and visually stunning. A true masterpiece!"
predicted_sentiment_positive = predict_review_sentiment(model, vectorizer, test_review_positive)
print(f"Review: '{test_review_positive}' -> Predicted Sentiment: {predicted_sentiment_positive}")

test_review_negative = "The acting was dull and the plot was utterly boring. I hated it."
predicted_sentiment_negative = predict_review_sentiment(model, vectorizer, test_review_negative)
print(f"Review: '{test_review_negative}' -> Predicted Sentiment: {predicted_sentiment_negative}")

test_review_neutral = "It was okay, not great, not terrible."
predicted_sentiment_neutral = predict_review_sentiment(model, vectorizer, test_review_neutral)
print(f"Review: '{test_review_neutral}' -> Predicted Sentiment: {predicted_sentiment_neutral}")