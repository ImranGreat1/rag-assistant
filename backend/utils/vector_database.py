import asyncio
from collections import defaultdict
from collections.abc import Callable

import numpy as np
from embedding import GeminiEmbeddingModel


def consine_similarity(vector_a: np.array, vector_b: np.array):
    dot_product = np.dot(vector_a, vector_b)
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)
    return dot_product / (norm_a * norm_b)


class VectorDatabase:
    def __init__(self, embedding_model: GeminiEmbeddingModel):
        self.vectors = defaultdict(np.array)
        self.embedding_model = embedding_model

    def insert(self, text: str, vector: np.array):
        self.vectors[text] = vector

    async def build_from_text(self, text_list: list[str]) -> "VectorDatabase":
        embeddings = await self.embedding_model.get_embeddings(text_list)
        for text, vector in zip(text_list, embeddings):
            self.insert(text, vector)

        return self

    async def search_by_text(
        self,
        query: str,
        distance_measure: Callable = consine_similarity,
        k: int = 3,
        return_as_text: bool = False,
    ):
        query_vector = await self.embedding_model.get_embedding(query)
        results = [
            (key, distance_measure(query_vector, vector))
            for key, vector in self.vectors.items()
        ]

        results = sorted(results, key=lambda x: x[1], reverse=True)[:k]
        return [result[0] for result in results] if return_as_text else results


async def main():
    text_list = [
        "I like to eat broccoli and bananas.",
        "I ate a banana and spinach smoothie for breakfast.",
        "Chinchillas and kittens are cute.",
        "My sister adopted a kitten yesterday.",
        "Look at this cute hamster munching on a piece of broccoli.",
    ]
    vector_db = VectorDatabase(embedding_model=GeminiEmbeddingModel())
    vector_db = await vector_db.build_from_text(text_list=text_list)

    relevant_text = await vector_db.search_by_text(
        query="Pet",
        k=2,
    )
    return relevant_text


if __name__ == "__main__":
    results = asyncio.run(main())
    print(results)
