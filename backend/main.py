from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from utils.rag_pipeline import RagPipeline

load_dotenv()


class UserQuery(BaseModel):
    query: str

app = FastAPI()


@app.get("/")
def root():
    return { "Introduction": "Hello, I am helpful assistant. You can ask me anything" }


@app.post("/chat")
async def chat(user_query: UserQuery):
    pipeline = RagPipeline()
    response = await pipeline.run_pipeline(user_query.query)
    return { "query_response": response }
