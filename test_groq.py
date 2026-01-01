from langchain_groq import ChatGroq
import os

llm = ChatGroq(
    groq_api_key=os.environ["GROQ_API_KEY"],
    model="llama-3.1-8b-instant",
)

print(llm.invoke("Say hello in one word").content)

