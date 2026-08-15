import asyncio

from infra.pinecone import PineconeDB
from langsmith import traceable
from utils.chat_models import ChatOpenAI
from utils.prompt import Prompt
from utils.template import RAG_SYSTEM_TEMPLATE, RAG_USER_TEMPLATE


class ManualRagPipeline:
    def __init__(self):
        pass

    async def run_pipeline(
        self, query: str, top_k: int = 5, top_n: int = 3, stream: bool = True
    ):
        if stream:
            return self._run_pipeline_stream(query, top_k, top_n)
        return await self._run_pipeline_sync(query, top_k, top_n)

    @traceable
    async def _run_pipeline_stream(self, query: str, top_k: int, top_n: int):
        """
        A simple RAG pipeline that perform densed vector retrieval, format prompts and context,
        pass formatted prompt to LLM and stream the response
        """
        # Retrieve documents that are relevant to the query
        relevant_contexts = await self.retrieve_context(
            query=query, top_k=top_k, top_n=top_n
        )

        # Create system and user prompts from predefined templates
        prompts = self.construct_prompts(query, relevant_contexts)

        # Call LLM with the formatted prompts and stream/return the response
        chat_model = self._create_client()
        result = await chat_model.run(messages=prompts, stream=True)
        async for chunk in result:
            yield chunk

    @traceable
    async def _run_pipeline_sync(self, query: str, top_k: int, top_n: int):
        relevant_contexts = await self.retrieve_context(query, top_k, top_n)
        prompts = self.construct_prompts(query, relevant_contexts)
        chat_model = self._create_client()
        return await chat_model.run(messages=prompts, stream=False)

    def _create_client(self) -> ChatOpenAI:
        chat_model = ChatOpenAI(
            model_name="gemini-3.6-flash",
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key_var="GOOGLE_API_KEY",
            tracing=True,
        )
        return chat_model

    @traceable
    async def retrieve_context(self, query: str, top_k: int, top_n: int):
        """Retrieve documents that are relevant to the query"""
        pc = PineconeDB()
        index = await pc.create_or_get_index(name="rag-assistance")
        relevant_contexts = await pc.search_index(
            index=index,
            namespace="health-wellness-guide",
            query={
                "top_k": top_k,
                "inputs": {"text": query},
            },
            rerank={
                "model": "bge-reranker-v2-m3",
                "top_n": top_n,
                "rank_fields": ["content"],
            },
        )
        return relevant_contexts

    @traceable
    def construct_prompts(self, query: str, relevant_contexts: list[tuple]):
        """
        Construct user and system prompt using predefined templates
        Use "developer" for openai models and system for gemini models.
        develope role is newly introduced by openai to replace system role
        """
        system_prompt = Prompt(role="system", prompt=RAG_SYSTEM_TEMPLATE)
        formatted_system_prompt = system_prompt.create_message(
            response_style="concise", response_length="brief"
        )

        user_prompt = Prompt(role="user", prompt=RAG_USER_TEMPLATE)
        formatted_user_prompt = user_prompt.create_message(
            context="\n".join(ctx[0] for ctx in relevant_contexts),
            context_count=len(relevant_contexts),
            similarity_scores=", ".join(str(ctx[1]) for ctx in relevant_contexts),
            query=query,
        )
        return [formatted_system_prompt, formatted_user_prompt]


async def main():
    rag_pipeline = ManualRagPipeline()
    response = await rag_pipeline.run_pipeline("Explain exercise basics", stream=False)
    print(response)


if __name__ == "__main__":
    asyncio.run(main())
