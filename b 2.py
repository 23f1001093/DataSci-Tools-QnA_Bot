import openai
import os
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPEN_API_KEY")

response = openai.Embedding.create(
    input="Hello",
    model="text-embedding-3-small"
)
print(response)