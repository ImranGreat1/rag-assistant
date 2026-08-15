import asyncio
import os

from dotenv import load_dotenv
from google import genai


class GeminiEmbeddingModel:
    def __init__(self, model: str = "gemini-embedding-001", batch: int = 200):
        load_dotenv()
        if not os.getenv("GOOGLE_API_KEY"):
            raise ValueError("GOOGLE API KEY is not set")

        self.client = genai.Client()
        self.model = model
        self.batch = batch

    async def get_embedding(self, text: str):
        result = await self.client.aio.models.embed_content(
            model=self.model, contents=[text]
        )
        return result.embeddings[0].values

    async def get_embeddings(self, text_list: list[str]):
        batches = [
            text_list[i : i + self.batch] for i in range(0, len(text_list), self.batch)
        ]

        async def process_batch(batch: list[str]):
            result = await self.client.aio.models.embed_content(
                model=self.model, contents=batch
            )
            return [embedding.values for embedding in result.embeddings]

        results = await asyncio.gather(*[process_batch(batch) for batch in batches])
        return [embedding for batch_result in results for embedding in batch_result]


if __name__ == "__main__":
    embedding_model = GeminiEmbeddingModel(batch=1)
    embeddings = asyncio.run(embedding_model.get_embeddings(["hello world", "Hi dear"]))
    print(len(embeddings))
    print(embeddings[0][0:10])
    print(type(embeddings[0]))
