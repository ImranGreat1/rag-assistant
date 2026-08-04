from infra.pinecone import PineconeDB

if __name__ == "__main__":
    pc = PineconeDB()
    index = pc.create_or_get_index(name="rag-assistance")
    pc.search_index(
        index=index,
        namespace="health-wellness-guide",
        query={
            "top_k": 5,
            "inputs": { "text": "What are some exercises or habits that can significantly improve my sleep" },
        },
        rerank={"model": "bge-reranker-v2-m3", "top_n": 3, "rank_fields": ["content"]},
    )
