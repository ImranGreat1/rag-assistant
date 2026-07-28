import re
from typing import Any

from template import RAG_SYSTEM_TEMPLATE


class Prompt:
    def __init__(
        self,
        role: str,
        prompt: str,
        strict: bool = False,
        default: dict[str, Any] | None = None,
    ):
        self.role = role
        self.prompt = prompt
        self.strict = strict
        self.defaults = default or {}
        self._pattern = re.compile(r"\{(\w+)\}")

    def create_message(self, **kwargs):
        """
        Replace placeholder variables with provided values or default values
        in the prompt
        """
        variables = self.get_all_variables()
        values = {**self.defaults, **kwargs}
        missing_vars = set(variables) - set(values.keys())

        if self.strict and missing_vars:
            raise ValueError(f"Missing required variables: {missing_vars}")
        formated_values = {
            key: values.get(key, self.defaults.get(key, "")) for key in variables
        }
        return self.prompt.format(**formated_values)

    def get_all_variables(self):
        """Find all variables {} in the prompt"""
        return self._pattern.findall(self.prompt)


if __name__ == "__main__":
    prompt = Prompt(
        role="system", 
        prompt=RAG_SYSTEM_TEMPLATE, 
        default={"response_style": "concise"}
    )
    result = prompt.create_message(response_style="concise", response_length="brief")
    print(result)
