import json
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# load dataset
data = json.load(open("outputs/labeled_data.json"))

texts = [x["text"] for x in data]
labels = [x["risk_category"] for x in data]

# vectorizer
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(texts)

# model
model = LogisticRegression(max_iter=1000)
model.fit(X, labels)

# save
joblib.dump(vectorizer, "outputs/vectorizer.pkl")
joblib.dump(model, "outputs/model.pkl")

print("Model trained!")