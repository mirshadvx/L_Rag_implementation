from dotenv import load_dotenv
import os
from google import genai

load_dotenv()

client = genai.Client(
    api_key=os.environ.get("GEMINI_API_KEY"),
)

response = client.models.generate_content(
    model='models/gemini-3.8-flash',
    contents="what is ai hallucinations"
)

print(response.text)