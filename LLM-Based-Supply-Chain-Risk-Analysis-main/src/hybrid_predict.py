import joblib
import numpy as np
import google.generativeai as genai

# load ML model
vectorizer = joblib.load("outputs/vectorizer.pkl")
model = joblib.load("outputs/model.pkl")

# Gemini setup (optional fallback)
genai.configure(api_key="YOUR_API_KEY")
llm = genai.GenerativeModel("gemini-1.5-flash")

THRESHOLD = 0.65


def ml_predict(text):

    X = vectorizer.transform([text])

    probs = model.predict_proba(X)[0]
    label = model.classes_[np.argmax(probs)]
    confidence = np.max(probs)

    return label, confidence


def gemini_fallback(text):

    prompt = f"""
Return JSON:
- risk_category
- severity_score
- root_cause
- explanation

Text:
{text}
"""

    response = llm.generate_content(prompt)

    return response.text


def predict(text):

    label, conf = ml_predict(text)

    if conf >= THRESHOLD:

        return {
            "mode": "ML",
            "risk_category": label,
            "confidence": float(conf)
        }

    else:

        return {
            "mode": "LLM",
            "llm_output": gemini_fallback(text)
        }


if __name__ == "__main__":

    text = "Factory shutdown in Vietnam disrupting supply chain"

    print(predict(text))