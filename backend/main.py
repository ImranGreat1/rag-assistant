from dotenv import load_dotenv
from fastapi import FastAPI

load_dotenv()

app = FastAPI()


@app.get("/")
def root():
    return { "Introduction": "Hello, I am helpful assistant. You can ask me anything" }


@app.get("/chat")
def chat():
    return "Ask me anything"
