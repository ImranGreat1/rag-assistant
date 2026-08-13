import os
from typing import Any

from dotenv import load_dotenv
from pinecone import AsyncPinecone as Pinecone
from pinecone import Index
from pydantic import BaseModel


class FieldMap(BaseModel):
    text: str


class EmbeddingConfig(BaseModel):
    model: str
    field_map: FieldMap


class Inputs(BaseModel):
    text: str


class Query(BaseModel):
    top_k: int
    inputs: Inputs


class Rerank(BaseModel):
    model: str
    top_n: int
    rank_fields: list[str]


class PineconeDB:
    def __init__(self):
        load_dotenv()
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("PINECONE API KEY is not provided")

        self.pc = Pinecone(api_key=api_key)

    async def create_or_get_index(
        self,
        name: str,
        cloud: str = "aws",
        region: str = "us-east-1",
        embed: EmbeddingConfig = None,
    ) -> Index:
        if not await self.pc.has_index(name=name):
            embedding_config = {
                "model": "llama-text-embed-v2",
                "field_map": {"text": "content"},
            }
            if embed:
                embedding_config = embed

            await self.pc.create_index_for_model(
                name=name, cloud=cloud, region=region, embed=embedding_config
            )
        index = await self.pc.index(name)
        return index

    async def get_index(self, name: str) -> Index:
        if not await self.pc.has_index(name=name):
            raise ValueError(f"Index {name} does not exist")
        index = await self.pc.index(name)
        return index

    async def upsert_records(self, index: Index, namespace: str, docs: list[Any]):
        await index.upsert_records(namespace=namespace, docs=docs)

    async def search_index(self, index: Index, namespace: str, query: Query, rerank: Rerank):
        results = await index.search(namespace=namespace, query=query, rerank=rerank)
        return [(hit.fields["content"], round(hit.score, 2)) for hit in results["result"]["hits"]]

    async def close(self):
        await self.pc.close()
