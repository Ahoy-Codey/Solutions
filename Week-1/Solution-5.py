import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score

np.random.seed(42)

good_feedback_phrases = [
    "Excellent product, highly recommend!",
    "Works perfectly, very satisfied with this purchase.",
    "Great quality and easy to use. Five stars!",
    "This is exactly what I needed, superb performance.",
    "Fast delivery and the item arrived in perfect condition.",
    "Very happy with the design and functionality.",
    "Good value for money, exceeded my expectations.",
    "Love it! Will definitely buy from this brand again.",
    "User-friendly interface, very intuitive.",
    "Much better than anticipated, a pleasant surprise.",
    "Reliable and durable, built to last.",
    "Fantastic customer support, quick and helpful.",
    "Efficient and powerful, makes tasks much easier.",
    "Looks great and performs even better.",
    "Best product in its category, highly effective.",
    "Exactly as described, no complaints at all.",
    "Definitely worth the investment.",
    "A game-changer for my daily routine.",
    "Impressed with the features for the price.",
    "Simply the best, couldn't ask for more.",
    "Very responsive and smooth operation.",
    "Compact and powerful, perfect for travel.",
    "Improved my workflow significantly.",
    "Highly functional and aesthetically pleasing.",
    "An essential item for every home.",
    "Quick setup and immediate results.",
    "Fantastic for its intended purpose.",
    "Sturdy construction, feels premium.",
    "Efficient power consumption.",
    "Made my life so much easier.",
    "Exceptional battery life.",
    "Crystal clear display.",
    "Seamless connectivity.",
    "Intuitive controls.",
    "Highly adaptable.",
    "Perfect size.",
    "Very quiet.",
    "Solid build.",
    "Top notch.",
    "Works flawlessly.",
    "Amazing!",
    "Superb!",
    "Love it!",
    "Happy!",
    "Great!",
    "Good!",
    "Best!",
    "Nice!",
    "Perfect!",
    "Awesome!"
]

bad_feedback_phrases = [
    "Terrible product, completely useless.",
    "Doesn't work as advertised, very disappointed.",
    "Poor quality and difficult to use. Waste of money.",
    "This is not what I expected, horrible performance.",
    "Slow delivery and the item arrived damaged.",
    "Very unhappy with the design and functionality.",
    "Bad value for money, totally failed my expectations.",
    "Hate it! Will never buy from this brand again.",
    "Confusing interface, very unintuitive.",
    "Much worse than anticipated, a terrible surprise.",
    "Unreliable and flimsy, likely to break soon.",
    "Horrible customer support, slow and unhelpful.",
    "Inefficient and weak, makes tasks much harder.",
    "Looks cheap and performs even worse.",
    "Worst product in its category, highly ineffective.",
    "Nothing like described, full of complaints.",
    "Definitely not worth the investment.",
    "A headache for my daily routine.",
    "Disappointed with the features for the price.",
    "Simply the worst, couldn't be more wrong.",
    "Very unresponsive and choppy operation.",
    "Bulky and weak, useless for travel.",
    "Hindered my workflow significantly.",
    "Poorly functional and ugly.",
    "A useless item for any home.",
    "Complicated setup and no results.",
    "Terrible for its intended purpose.",
    "Flimsy construction, feels cheap.",
    "High power consumption.",
    "Made my life so much harder.",
    "Dreadful battery life.",
    "Blurry display.",
    "Erratic connectivity.",
    "Confusing controls.",
    "Not adaptable.",
    "Wrong size.",
    "Very noisy.",
    "Weak build.",
    "Bottom tier.",
    "Fails often.",
    "Awful!",
    "Horrible!",
    "Dislike!",
    "Sad!",
    "Bad!",
    "Poor!",
    "Worst!",
    "Useless!",
    "Fail!",
    "Frustrating!"
]

good_feedback = np.random.choice(good_feedback_phrases, 50, replace=True).tolist()
good_labels = ['good'] * 50

bad_feedback = np.random.choice(bad_feedback_phrases, 50, replace=True).tolist()
bad_labels = ['bad'] * 50

all_feedback = good_feedback + bad_feedback
all_labels = good_labels + bad_labels

df = pd.DataFrame({
    'Feedback': all_feedback,
    'Label': all_labels
})

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

vectorizer = TfidfVectorizer(max_features=300, lowercase=True, stop_words='english')
X = vectorizer.fit_transform(df['Feedback']) 
y = df['Label'] 


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)


model = LogisticRegression(random_state=42, solver='liblinear') 
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

precision = precision_score(y_test, y_pred, pos_label='good') 
recall = recall_score(y_test, y_pred, pos_label='good')
f1 = f1_score(y_test, y_pred, pos_label='good')

print(f"\nModel Performance on Test Set (Positive Label: 'good'):")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-score: {f1:.2f}")


def text_preprocess_vectorize(texts, vectorizer):
    vectorized_matrix = vectorizer.transform(texts)
    return vectorized_matrix

new_texts = [
    "This product is absolutely amazing, great value!",
    "Terrible customer service and the device broke quickly.",
    "It's okay, nothing special, nothing bad."
]

new_texts_vectorized = text_preprocess_vectorize(new_texts, vectorizer)
print("Shape of newly vectorized texts:", new_texts_vectorized.shape)

new_predictions = model.predict(new_texts_vectorized)
print(f"Predictions for new texts: {new_predictions}")
