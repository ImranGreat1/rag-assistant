from fastapi import FastAPI


app = FastAPI()


@app.get("/")
def root():
    return { "Introduction": "Hello, I am helpful assistant. You can ask me anything" }


@app.get("/chat")
def chat():
    return "Ask me anything"
