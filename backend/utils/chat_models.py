import os
from typing import Any

from dotenv import load_dotenv
from google import genai
from openai import OpenAI


class ChatOpenAI:
    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key_var: str | None = None,
    ):
        api_key = os.getenv(api_key_var if api_key_var else "OPENAI_API_KEY")
        if not api_key:
            raise ValueError("API key must be provided (openai, gemini)")

        self.model_name = model_name
        self.base_url = base_url
        self.api_key = api_key

    def run(self, messages: list[dict[str, Any]], text_only: bool = True):
        client = (
            OpenAI(api_key=self.api_key, base_url=self.base_url)
            if self.base_url
            else OpenAI()
        )
        response = client.chat.completions.create(
            model=self.model_name, messages=messages
        )

        if text_only:
            return response.choices[0].message.content
        return response


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


if __name__ == "__main__":
    load_dotenv()

    # Testing Google chat API
    chat_google = ChatGoogleGenAI(model_name="gemini-3.6-flash")
    response = chat_google.run(
        input="Hi gemini",
        instruction="You are helpful assistant that gives brief and precise response",
    )
    print(response.output_text)

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
        {"role": "user", "content": "Hi GPT"},
    ]
    response = chat_openai.run(messages=messages, text_only=True)
    print(response)
