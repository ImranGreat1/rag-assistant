import asyncio
import os
from typing import Any

from dotenv import load_dotenv
from google import genai
from langsmith.wrappers import wrap_openai
from openai import AsyncOpenAI


class ChatOpenAI:
    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key_var: str | None = None,
        tracing: bool = False,
    ):
        api_key = os.getenv(api_key_var if api_key_var else "OPENAI_API_KEY")
        if not api_key:
            raise ValueError("API key must be provided (openai, gemini)")

        self.model_name = model_name
        self.base_url = base_url
        self.api_key = api_key
        self.client = self.init_client(api_key, base_url, tracing)

    # Initial OpenAI Client SDK
    def init_client(self, api_key: str, base_url: str, tracing: bool) -> AsyncOpenAI:
        kwargs = { "api_key": api_key }
        if base_url:
            kwargs["base_url"] = base_url

        client = (
            wrap_openai(AsyncOpenAI(**kwargs)) if tracing else AsyncOpenAI(**kwargs)
        )
        return client

    # Perform Inference - call the LLM
    async def run(self, messages: list[dict[str, Any]], stream: bool = False):
        if stream:
            return self._stream_response(messages)
        else:
            response = await self.client.chat.completions.create(
                model=self.model_name, messages=messages, stream=stream
            )
            return response.choices[0].message.content

    # Stream LLM response
    async def _stream_response(self, messages: list[dict[str, Any]]):
        response = await self.client.chat.completions.create(
            model=self.model_name, messages=messages, stream=True
        )
        async for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


class ChatGoogleGenAI:
    def __init__(self, model_name: str):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Gemini API key is not provided")

        self.model_name = model_name
        self.api_key = api_key

    def run(self, input: str, instruction: str):
        client = genai.Client()
        interaction = client.interactions.create(
            model=self.model_name, input=input, system_instruction=instruction
        )
        return interaction

    def conversation(self, input: str, previous_interaction_id: str):
        """
        For multi-turn conversation, the conversation method is used which
        accepts previous_interaction_id it back with the next interaction
        """
        client = genai.Client()
        interaction = client.interactions.create(
            model=self.model_name,
            input=input,
            previous_interaction_id=previous_interaction_id,
        )
        return interaction


async def main():
    load_dotenv()

    # Testing Google chat API
    # chat_google = ChatGoogleGenAI(model_name="gemini-3.6-flash")
    # response = chat_google.run(
    #     input="Hi gemini",
    #     instruction="You are helpful assistant that gives brief and precise response",
    # )
    # print(response.output_text)

    # Testing OpenAI chat API
    chat_openai = ChatOpenAI(
        model_name="gemini-3.6-flash",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key_var="GEMINI_API_KEY",
    )
    messages = [
        {
            "role": "developer",
            "content": "You are helpful assistant that gives brief and precise response",
        },
        {
            "role": "user",
            "content": "In not less than 100 words, give a brief explaination on AI and Agents",
        },
    ]
    stream_gen = await chat_openai.run(messages=messages, stream=True)
    async for chunk in stream_gen:
        print(chunk)


if __name__ == "__main__":
    asyncio.run(main())
