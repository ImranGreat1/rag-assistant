import asyncio

from utils.chat_models import ChatOpenAI
from utils.embedding import GeminiEmbeddingModel
from utils.prompt import Prompt
from utils.template import RAG_SYSTEM_TEMPLATE, RAG_USER_TEMPLATE
from utils.text import CharacterTextSplitter, TextFileLoader
from utils.vector_database import VectorDatabase


async def init_vector_store():
    embedding_model = GeminiEmbeddingModel()
    vector_db = VectorDatabase(embedding_model=embedding_model)
    document = TextFileLoader("data/HealthWellnessGuide.txt")
    text_splitter = CharacterTextSplitter()
    chunks = text_splitter.split_texts(document.load_document())
    await vector_db.build_from_text(chunks)
    return vector_db


class RagPipeline:
    def __init__(self):
        pass

    async def run_pipeline(self, user_query: str, k: int = 3):
        vector_db = await init_vector_store()
        relevant_contexts = await vector_db.search_by_text(
            query=user_query, k=k
        )
        system_prompt = Prompt(role="developer", prompt=RAG_SYSTEM_TEMPLATE)
        formatted_system_prompt = system_prompt.create_message(
            response_style="concise", response_length="brief"
        )

        user_prompt = Prompt(role="user", prompt=RAG_USER_TEMPLATE)
        formatted_user_prompt = user_prompt.create_message(
            context="\n".join(ctx[0] for ctx in relevant_contexts),
            context_count=len(relevant_contexts),
            similarity_scores=", ".join(str(ctx[1]) for ctx in relevant_contexts),
            user_query=user_query,
        )

        chat_model = ChatOpenAI(
            model_name="gemini-3.6-flash",
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key_var="GEMINI_API_KEY",
        )

        response = chat_model.run(
            messages=[formatted_system_prompt, formatted_user_prompt], 
            text_only=True
        )

        return response


async def main():
    rag_pipeline = RagPipeline()
    await rag_pipeline.run_pipeline("Explain exercise basics")


if __name__ == "__main__":
    asyncio.run(main())
