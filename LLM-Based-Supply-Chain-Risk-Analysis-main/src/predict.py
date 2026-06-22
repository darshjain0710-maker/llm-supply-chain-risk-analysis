import joblib

vectorizer = joblib.load("outputs/vectorizer.pkl")
model = joblib.load("outputs/model.pkl")


def predict(text):

    X = vectorizer.transform([text])

    return {
        "risk_category": model.predict(X)[0]
    }


if __name__ == "__main__":

    text = "Port strike causing shipment delays in China"

    print(predict(text))