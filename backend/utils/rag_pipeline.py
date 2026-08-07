import asyncio

from infra.pinecone import PineconeDB

from utils.chat_models import ChatOpenAI
from utils.prompt import Prompt
from utils.template import RAG_SYSTEM_TEMPLATE, RAG_USER_TEMPLATE


class RagPipeline:
    def __init__(self):
        pass

    async def run_pipeline(
        self, user_query: str, top_k: int = 5, top_n: int = 3, stream: bool = True
    ):
        """
        A simple RAG pipeline that perform densed vector retrieval, format prompts and context,
        pass formatted prompt to LLM and stream the response
        """
        # Retrieve documents that are relevant to the query
        pc = PineconeDB()
        index = pc.create_or_get_index(name="rag-assistance")
        relevant_contexts = pc.search_index(
            index=index,
            namespace="health-wellness-guide",
            query={
                "top_k": top_k,
                "inputs": {
                    "text": "What are some exercises or habits that can significantly improve my sleep"
                },
            },
            rerank={
                "model": "bge-reranker-v2-m3",
                "top_n": top_n,
                "rank_fields": ["content"],
            },
        )

        # Create system prompt from a predefined template
        system_prompt = Prompt(role="developer", prompt=RAG_SYSTEM_TEMPLATE)
        formatted_system_prompt = system_prompt.create_message(
            response_style="concise", response_length="brief"
        )

        # Create user prompt using user query and relevant documents
        user_prompt = Prompt(role="user", prompt=RAG_USER_TEMPLATE)
        formatted_user_prompt = user_prompt.create_message(
            context="\n".join(ctx[0] for ctx in relevant_contexts),
            context_count=len(relevant_contexts),
            similarity_scores=", ".join(str(ctx[1]) for ctx in relevant_contexts),
            user_query=user_query,
        )

        # Call LLM with the formatted prompts and stream the response
        chat_model = ChatOpenAI(
            model_name="gemini-3.6-flash",
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key_var="GEMINI_API_KEY",
        )
        return await chat_model.run(
            messages=[formatted_system_prompt, formatted_user_prompt], stream=stream
        )


async def main():
    rag_pipeline = RagPipeline()
    response = await rag_pipeline.run_pipeline("Explain exercise basics", stream=False)
    print(response)


if __name__ == "__main__":
    asyncio.run(main())
