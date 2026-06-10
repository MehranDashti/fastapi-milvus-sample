import json
from openai import OpenAI
from tools import TOOL_SCHEMAS, execute_tool
from config import settings
from logger import get_logger

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
    history: list[dict] = None,
    max_iterations: int = 8,
) -> dict:
    """
    Run the tool-use agent loop.

    The loop:
    1. Send question + tools to GPT
    2. If GPT returns tool_call → execute tool → send result back
    3. If GPT returns text → done
    4. Repeat up to max_iterations times

    Args:
        question:       user's question
        history:        optional conversation history
        max_iterations: safety limit to prevent infinite loops

    Returns:
        dict with answer, tools_used, iterations, tool_results
    """
    if history is None:
        history = []

    # Build initial messages
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Add conversation history
    for turn in history:
        messages.append({"role": turn["role"], "content": turn["content"]})

    # Add current question
    messages.append({"role": "user", "content": question})

    tools_used = []         # track which tools were called
    tool_results = []       # track tool inputs and outputs
    iterations = 0

    while iterations < max_iterations:
        iterations += 1
        logger.info(f"[ToolAgent] Iteration {iterations}/{max_iterations}")

        # Call GPT with tools
        response = _client.chat.completions.create(
            model=settings.openai.LLM_MODEL,
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",     # GPT decides: call tool or answer directly
            temperature=0.1,
        )

        message = response.choices[0].message

        # ── Case A: GPT wants to call a tool ──────────────────────────────
        if message.tool_calls:
            # Add GPT's tool request to message history
            messages.append({
                "role": "assistant",
                "content": message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        }
                    }
                    for tc in message.tool_calls
                ]
            })

            # Execute each requested tool
            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)

                logger.info(f"[ToolAgent] GPT requests tool: '{tool_name}' args={tool_args}")

                # Execute the actual tool function
                tool_result = execute_tool(tool_name, tool_args)

                # Track what happened
                tools_used.append(tool_name)
                tool_results.append({
                    "tool": tool_name,
                    "args": tool_args,
                    "result": tool_result[:300],  # truncate for response
                })

                # Send tool result back to GPT
                # GPT will read this and decide next step
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result,
                })

            # Continue loop — GPT will process tool results and decide next step

        # ── Case B: GPT has enough info, returns final answer ─────────────
        else:
            answer = message.content.strip()
            logger.info(f"[ToolAgent] Final answer after {iterations} iterations")

            return {
                "answer": answer,
                "tools_used": list(dict.fromkeys(tools_used)),  # unique, ordered
                "tool_results": tool_results,
                "iterations": iterations,
            }

    # Safety fallback — hit max_iterations without a final answer
    logger.warning(f"[ToolAgent] Hit max_iterations ({max_iterations})")
    return {
        "answer": "I was unable to complete the task within the allowed number of steps.",
        "tools_used": tools_used,
        "tool_results": tool_results,
        "iterations": iterations,
    }