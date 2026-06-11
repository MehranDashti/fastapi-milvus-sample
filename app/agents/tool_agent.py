import json

from openai import OpenAI

from app.agents.tools import TOOL_SCHEMAS, execute_tool
from app.config import settings
from app.logger import get_logger

logger = get_logger(__name__)

_client = OpenAI(api_key=settings.openai.API_KEY)

SYSTEM_PROMPT = """You are an intelligent assistant with access to several tools.

Available tools:
1. milvus_search    - Search internal documents (FastAPI, Milvus, RAG, Python, etc.)
2. web_search       - Search the web for current/real-time information
3. calculator       - Evaluate mathematical expressions
4. get_current_date - Get current date and time

Decision rules:
- For questions about known topics in documents → use milvus_search first
- For latest versions, current events, real-time data → use web_search
- For math calculations → use calculator
- For time/date questions → use get_current_date
- You can call multiple tools in sequence if needed
- When you have enough information → provide a final answer without calling tools
- Always cite your sources (document name or URL)
- Be concise and precise
"""


def run_tool_agent(
    question: str,
    history: list[dict] | None = None,
    max_iterations: int = 8,
) -> dict:
    if history is None:
        history = []

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history:
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": question})

    tools_used = []
    tool_results = []
    iterations = 0

    while iterations < max_iterations:
        iterations += 1
        logger.info(f"[ToolAgent] Iteration {iterations}/{max_iterations}")

        response = _client.chat.completions.create(
            model=settings.openai.LLM_MODEL,
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
            temperature=0.1,
        )

        message = response.choices[0].message

        if message.tool_calls:
            messages.append(
                {
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in message.tool_calls
                    ],
                }
            )

            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)

                logger.info(f"[ToolAgent] GPT requests tool: '{tool_name}' args={tool_args}")

                tool_result = execute_tool(tool_name, tool_args)

                tools_used.append(tool_name)
                tool_results.append(
                    {
                        "tool": tool_name,
                        "args": tool_args,
                        "result": tool_result[:300],
                    }
                )

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_result,
                    }
                )
        else:
            answer = message.content.strip()
            logger.info(f"[ToolAgent] Final answer after {iterations} iterations")
            return {
                "answer": answer,
                "tools_used": list(dict.fromkeys(tools_used)),
                "tool_results": tool_results,
                "iterations": iterations,
            }

    logger.warning(f"[ToolAgent] Hit max_iterations ({max_iterations})")
    return {
        "answer": "I was unable to complete the task within the allowed number of steps.",
        "tools_used": tools_used,
        "tool_results": tool_results,
        "iterations": iterations,
    }
