import asyncio
import os

from dotenv import load_dotenv
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_qdrant import QdrantVectorStore
from utils.template import RAG_SYSTEM_TEMPLATE, RAG_USER_TEMPLATE


class LangChainRagPipeline:

    async def run_pipeline(self, query: str, stream: bool = False):
        if stream:
            return self.run_pipeline_stream(query) # You don't await async generator functions
        return await self.run_pipeline_sync(query)

    # Run the pipeline and stream the response
    async def run_pipeline_stream(self, query: str):
        prompt = ChatPromptTemplate.from_messages(
            [RAG_SYSTEM_TEMPLATE, RAG_USER_TEMPLATE]
        )
        model = ChatGoogleGenerativeAI(model="gemini-3.6-flash")
        output_parser = StrOutputParser()

        chain = RunnableLambda(self.find_relevant_documents) | prompt | model | output_parser
        result = chain.astream({"query": query})
        async for chunk in result:
            yield chunk

    # Run pipeline and await the complete response
    async def run_pipeline_sync(self, query: str):
        prompt = ChatPromptTemplate.from_messages(
            [RAG_SYSTEM_TEMPLATE, RAG_USER_TEMPLATE]
        )
        model = ChatGoogleGenerativeAI(model="gemini-3.6-flash")
        output_parser = StrOutputParser()

        chain = RunnableLambda(self.find_relevant_documents) | prompt | model | output_parser
        result = await chain.ainvoke({"query": query})
        return result

    # Find relevant documents from Qdrant vector store
    async def find_relevant_documents(self, query: str, k: int = 3):
        embedding = FastEmbedEmbeddings(model="BAAI/bge-small-en-v1.5")
        vector_store = QdrantVectorStore.from_existing_collection(
            collection_name="rag-assistant", 
            embedding=embedding, 
            path="./vector_store",
            # url=os.getenv("QDRANT_CLUSTER_ENDPOINT") # For cloud hosted Qdrant
        )
        result = await vector_store.asimilarity_search(query=query, k=k)
        context = "\n".join(doc.page_content for doc in result)
        # Prompts runnable expects these values to replace the placeholders
        return {
            "response_style": "precise",
            "response_length": "brief",
            "context": context,
            "context_count": len(context),
            "query": query,
            "similarity_scores": []
        }


async def main():
    pipeline = LangChainRagPipeline()
    result = await pipeline.run_pipeline(
        query="What are some exercises or habits that can significantly improve my sleep",
        stream=True
    )

    async for chunk in result:
        print(chunk)


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
