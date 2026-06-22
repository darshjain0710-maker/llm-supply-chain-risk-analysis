def weak_label(text):

    if not text:
        return "Unknown"

    text = text.lower()

    if any(x in text for x in ["strike", "port", "delay", "shipping", "logistics"]):
        return "Transportation Risk"

    if any(x in text for x in ["tariff", "war", "sanctions", "china", "ukraine"]):
        return "Geopolitical Risk"

    if any(x in text for x in ["factory", "production", "manufacturing"]):
        return "Operational Risk"

    if any(x in text for x in ["demand", "sales drop", "consumption"]):
        return "Demand Risk"

    if any(x in text for x in ["supplier", "procurement", "vendor"]):
        return "Supplier Risk"

    return "Unknown"