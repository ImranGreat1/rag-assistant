from contextlib import asynccontextmanager

from fastapi import FastAPI
from infra.pinecone import PineconeDB


@asynccontextmanager
async def lifespan(app: FastAPI):
    pc = PineconeDB()
    app.state.vector_db = pc

    yield

    app.state.vector_db.close()