# app/core/tools.py
# Tool definitions for the multimodal research agent.
# Same tools as LangChain Copilot — reusable across projects.

from langchain_core.tools import tool
from datetime import datetime
import math
import os

@tool
def calculator(expression: str) -> str:
    """
    Evaluates a mathematical expression and returns the result.

    Use this tool whenever the user asks for:
    - Percentage calculations
    - Growth rate calculations
    - Arithmetic on numbers extracted from documents or charts
    - Any computation where precision matters

    Args:
        expression: A valid Python math expression as a string.
                   Examples: "15228 * 1.0835", "(31940 - 15228) / 15228 * 100"

    Returns:
        The result as a string, or an error message if invalid.
    """
    try:
        allowed_names = {
            "abs": abs, "round": round, "min": min, "max": max,
            **{name: getattr(math, name) for name in dir(math)
               if not name.startswith("_")}
        }
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return f"{result:.4f}" if isinstance(result, float) else str(result)
    except ZeroDivisionError:
        return "Error: division by zero"
    except Exception as e:
        return f"Error evaluating expression: {str(e)}"

@tool
def get_current_date() -> str:
    """
    Returns today's date and current year.

    Use this tool when:
    - The user asks about the current date or year
    - You need to calculate how many years ago something happened
    - You need to determine if data is recent or outdated

    Returns:
        Current date as a formatted string.
    """
    now = datetime.now()
    return f"Today is {now.strftime('%A, %B %d, %Y')}. Current year: {now.year}."

@tool
def compute_growth_rate(
    start_value: float,
    end_value: float,
    periods: int = 1
) -> str:
    """
    Computes growth rate between two values over a number of periods.

    Use this tool when the user asks about:
    - GDP growth between two years
    - Percentage change between two economic figures
    - Compound annual growth rate (CAGR) over multiple years

    Args:
        start_value: The initial value
        end_value:   The final value
        periods:     Number of periods between start and end.
                    For CAGR over multiple years, pass the number of years.

    Returns:
        String showing simple percentage change and CAGR if periods > 1.
    """
    if start_value == 0:
        return "Error: start_value cannot be zero"

    simple_change = ((end_value - start_value) / start_value) * 100

    if periods == 1:
        return f"Growth: {simple_change:.2f}%"

    cagr = ((end_value / start_value) ** (1 / periods) - 1) * 100

    return (
        f"Total growth: {simple_change:.2f}%\n"
        f"CAGR over {periods} periods: {cagr:.2f}% per period"
    )

@tool
def web_search(query: str) -> str:
    """
    Searches the web for current information not available in documents.

    Use this tool when:
    - The user asks about information not in the uploaded documents
    - The user asks to compare document data with current real-world figures
    - The user asks about recent events after the document time period

    Do NOT use this for questions answerable from uploaded documents.

    Args:
        query: A clear, specific search query string

    Returns:
        Search results as a formatted string with sources
    """
    api_key = os.getenv("TAVILY_API_KEY", "")

    if not api_key:
        return (
            f"[Web search unavailable — TAVILY_API_KEY not set]\n"
            f"Query was: '{query}'"
        )

    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=api_key)

        response = client.search(
            query=query,
            search_depth="basic",
            max_results=3
        )

        results = []
        for i, result in enumerate(response["results"], 1):
            results.append(
                f"[Result {i}]\n"
                f"Title: {result['title']}\n"
                f"Source: {result['url']}\n"
                f"Content: {result['content']}"
            )

        return "\n\n".join(results) if results else "No results found."

    except Exception as e:
        return f"Web search error: {str(e)}"