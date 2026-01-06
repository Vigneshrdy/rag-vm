import os
from google import genai

client = genai.Client(
    api_key=os.environ["GEMINI_API_KEY"]
)

response = client.models.generate_content(
    model="gemini-1.5-flash",
    contents="What is criminal breach of trust under IPC?"
)

print(response.text)
