from taxonomy import RISK_TAXONOMY


def assign_risk(text):

    text = text.lower()


    for category, keywords in RISK_TAXONOMY.items():

        for word in keywords:

            if word in text:
                return category


    return "No Risk"