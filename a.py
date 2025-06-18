import os
import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPEN_API_KEY")

try:
    res = openai.Embedding.create(input="Hello", model="text-embedding-3-small")
    print("✅ Success:", res)
except Exception as e:
    print("❌ Error:", e)