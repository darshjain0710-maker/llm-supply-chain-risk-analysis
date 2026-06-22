import os
import json

from dotenv import load_dotenv
from google import genai


load_dotenv()


client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)



def extract_risk(text):

    prompt = f"""

You are a supply chain risk analyst.

Analyze this logistics text.

Extract:

risk_category
supply_chain_stage
severity_score
sentiment
root_cause
explanation

Return only JSON.

Text:
{text}

"""


    response = client.models.generate_content(
        model ="gemini-2.5-flash",
        contents=prompt
    )


    try:
        clean = response.text.replace("```json","").replace("```","").strip()

        result = json.loads(clean)


        if isinstance(result, list):
            result = result[0]


        return result

    except:

        return {
            "raw_output": response.text
        }