import random

from langchain_core.tools import tool


@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression. Use this for any math calculations.

    Args:
        expression: A mathematical expression to evaluate (e.g., '2 + 2', '10 * 5')
    """
    try:
        # Using eval with restricted globals for safety
        result = eval(expression, {"__builtins__": {}}, {})
        return f"The result of {expression} is {result}"
    except Exception as e:
        return f"Error evaluating expression: {e}"


@tool
def get_current_time():
    """Get the current date and time. Use this when the user asks about the current time or date."""
    from datetime import datetime

    return (
        f"The current date and time is: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )


@tool
def fahrenheit_to_celcius(f: float) -> float:
    """Converts temperature from Fahrenheit to Celcius

    Args:
        f: Temperature in Fahrenheit (float)
    Returns:
        Temperature in Celcius (float)
    """
    return (f - 32.0) * 5.0 / 9.0


@tool
def rand_int(start: int, end: int) -> int:
    """Generate a random number within a given range.
    Args:
        start: The start position of the range
        end: The end position of the range
    Returns:
        A random number between
    """
    return random.randint(start, end)