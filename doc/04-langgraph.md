# 04 — LangGraph: Stateful Agent Graphs

> **Goal of this document:** Understand how LangGraph models agent logic as a
> directed graph, how state flows through nodes, how conditional edges create
> branching and loops, and why this architecture is superior to manually
> coded agent loops for production systems.

---

## 1. Why Chains Are Not Enough for Agents

A LangChain chain is a fixed sequence: A → B → C → done. It cannot:

- Loop back to an earlier step based on the result of a later step
- Take different paths depending on runtime state
- Retry a step when the output is below a quality threshold
- Pause and wait for human approval before continuing

An **agent** needs all of these. Agents are not pipelines — they are
decision-making processes that observe state, decide on actions, execute those
actions, observe the new state, and repeat.

LangGraph represents this as a directed graph where:

- **Nodes** are Python functions (computation steps)
- **Edges** are connections between nodes (fixed or conditional)
- **State** is a typed dictionary that travels through every node
- **Conditional edges** implement branching and loops

---

## 2. The Graph Mental Model

Think of a LangGraph agent as a finite state machine where the "states" are
nodes (processing steps) and "transitions" are edges (which step runs next).

```
           ┌──────────┐
    ┌──────→  RETRIEVE │
    │      └────┬─────┘
    │           ↓
    │      ┌──────────┐
    │      │ EVALUATE │
    │      └────┬─────┘
    │           │
    │     ┌─────┴──────┐
    │   NO│             │YES (quality OK)
    │     ↓             ↓
    │  ┌──────────┐  ┌──────────┐
    └──┤ REPHRASE │  │ GENERATE │
       └──────────┘  └────┬─────┘
                          ↓
                        [END]
```

This graph implements the agentic RAG loop: retrieve → evaluate quality →
either generate an answer or rephrase the question and retry.

---

## 3. State — The Shared Blackboard

State is a `TypedDict` that all nodes read from and write to. It is the
"working memory" of the agent — everything a node needs to do its job is
in the state. Everything a node produces is added back to the state.

```python
# app/agents/rag_agent.py
from typing import TypedDict

class AgentState(TypedDict):
    question:            str           # original question — never changes
    rephrased_question:  str | None    # GPT-rephrased version for better retrieval
    chunks:              list          # retrieved chunks from Milvus
    best_score:          float         # highest similarity score among chunks
    answer:              str | None    # final generated answer
    attempts:            int           # how many retrieval attempts so far
    quality_ok:          bool          # did retrieval meet quality threshold?
    reasoning:           list[str]     # human-readable log of agent decisions
```

**Key design principles:**

1. **Immutable original input:** `question` is set once at the beginning and
   never modified. Even when rephrasing, the original question is preserved for
   the final answer generation.

2. **Accumulate reasoning:** `reasoning` is a list that grows with each node.
   This creates an audit trail of every decision the agent made.

3. **Explicit quality gate:** `quality_ok` is a boolean written by
   `evaluate_node` and read by the conditional edge router. It separates the
   evaluation logic from the routing logic.

---

## 4. Nodes — Pure Functions Over State

Every node is a Python function that:
- Receives the **full** current state
- Returns a **partial** update (a dict with only the fields it changes)
- LangGraph merges the returned dict into the state automatically

```python
# app/agents/rag_agent.py
def retrieve_node(state: AgentState) -> dict:
    """Node 1: Retrieve chunks from Milvus."""
    # Read from state
    query    = state.get("rephrased_question") or state["question"]
    attempts = state.get("attempts", 0) + 1

    # Do work
    chunks     = search(query, top_k=5)
    best_score = max((c["score"] for c in chunks), default=0.0)

    # Append to reasoning log (immutable pattern — copy then extend)
    reasoning = state.get("reasoning", [])
    reasoning.append(
        f"Attempt {attempts}: retrieved {len(chunks)} chunks, best score={best_score:.4f}"
    )

    # Return PARTIAL update — only the fields this node changes
    return {
        "chunks":     chunks,
        "best_score": best_score,
        "attempts":   attempts,
        "reasoning":  reasoning,
    }
```

Notice: the node does NOT return `question` or `answer` or `quality_ok`.
LangGraph merges `{"chunks": ..., "best_score": ..., ...}` into the existing
state, leaving all other fields unchanged.

```python
def evaluate_node(state: AgentState) -> dict:
    """Node 2: Decide if retrieval quality is acceptable."""
    best_score = state["best_score"]
    attempts   = state["attempts"]

    # Quality passes if score is good enough OR we've used all attempts
    quality_ok = best_score >= SCORE_THRESHOLD or attempts >= MAX_ATTEMPTS

    reasoning = state.get("reasoning", [])
    if quality_ok:
        reasoning.append(
            f"Quality PASSED (score={best_score:.4f} >= {SCORE_THRESHOLD} "
            f"OR attempts={attempts} >= {MAX_ATTEMPTS})"
        )
    else:
        reasoning.append(
            f"Quality FAILED (score={best_score:.4f} < {SCORE_THRESHOLD}) → rephrasing"
        )

    return {"quality_ok": quality_ok, "reasoning": reasoning}
```

```python
def rephrase_node(state: AgentState) -> dict:
    """Node 3: Ask GPT to rephrase the question for better retrieval."""
    original = state["question"]

    messages = [
        SystemMessage(content=(
            "You are a search query optimizer. "
            "Rephrase the given question to improve document retrieval. "
            "Make it more specific. Return ONLY the rephrased question."
        )),
        HumanMessage(content=f"Original: {original}\nRephrased:"),
    ]

    response  = _llm.invoke(messages)
    rephrased = response.content.strip()

    reasoning = state.get("reasoning", [])
    reasoning.append(f"Rephrased question: '{rephrased}'")

    return {"rephrased_question": rephrased, "reasoning": reasoning}
```

```python
def generate_node(state: AgentState) -> dict:
    """Node 4: Generate the final answer from retrieved chunks."""
    result = ask_with_score_filter(
        query=state["question"],
        chunks=state["chunks"],
        min_score=0.35,   # lower threshold here — agent already verified quality
    )

    reasoning = state.get("reasoning", [])
    reasoning.append(f"Generated answer from {len(state['chunks'])} chunks")

    return {"answer": result["answer"], "reasoning": reasoning}
```

---

## 5. Edges — Connecting Nodes

### Fixed Edges

A fixed edge always routes from node A to node B. No condition.

```python
graph.add_edge("retrieve", "evaluate")   # retrieve always goes to evaluate
graph.add_edge("rephrase", "retrieve")   # rephrase always goes back to retrieve
graph.add_edge("generate", END)          # generate always terminates the graph
```

`END` is a special sentinel from `langgraph.graph` that signals graph
termination.

### Conditional Edges

A conditional edge calls a routing function that returns the **name** of
the next node.

```python
def route_after_evaluate(state: AgentState) -> str:
    """Decide: go to generate (good quality) or rephrase (bad quality)?"""
    if state["quality_ok"]:
        return "generate"
    return "rephrase"

graph.add_conditional_edges(
    "evaluate",            # from this node
    route_after_evaluate,  # call this function with current state
    {
        "generate": "generate",   # if function returns "generate" → go to generate node
        "rephrase": "rephrase",   # if function returns "rephrase" → go to rephrase node
    }
)
```

The mapping dict `{"generate": "generate"}` maps *return values* to *node
names*. Return values and node names can differ (useful for readability).

---

## 6. Building and Compiling the Graph

```python
# app/agents/rag_agent.py
def build_agent():
    graph = StateGraph(AgentState)

    # Register all nodes
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("evaluate", evaluate_node)
    graph.add_node("rephrase", rephrase_node)
    graph.add_node("generate", generate_node)

    # Entry point
    graph.set_entry_point("retrieve")

    # Fixed edges
    graph.add_edge("retrieve", "evaluate")
    graph.add_edge("rephrase", "retrieve")    # ← creates the retry loop
    graph.add_edge("generate", END)

    # Conditional edge (branching decision)
    graph.add_conditional_edges(
        "evaluate",
        route_after_evaluate,
        {"generate": "generate", "rephrase": "rephrase"},
    )

    return graph.compile()   # validate graph, freeze it, return runnable
```

`graph.compile()` validates the graph — checks for unreachable nodes,
missing edges, type mismatches. A compiled graph is a `CompiledGraph` that
behaves like any LangChain `Runnable`: it has `.invoke()`, `.stream()`, and
`.batch()`.

**Compilation happens once at startup (singleton):**

```python
_agent = None

def get_agent():
    global _agent
    if _agent is None:
        _agent = build_agent()
        logger.info("[Agent] LangGraph agent compiled and ready.")
    return _agent
```

This avoids re-compiling the graph on every request — graph compilation is
expensive (graph validation, JIT optimization). The compiled graph is
thread-safe and can be reused across concurrent requests.

---

## 7. Invoking the Graph

```python
# app/agents/rag_agent.py
def agent_query(question: str) -> dict:
    agent = get_agent()

    # Initial state — required fields must be set, optional fields can be None/default
    initial_state: AgentState = {
        "question":           question,
        "rephrased_question": None,
        "chunks":             [],
        "best_score":         0.0,
        "answer":             None,
        "attempts":           0,
        "quality_ok":         False,
        "reasoning":          [],
    }

    final_state = agent.invoke(initial_state)

    sources = list({c["source"] for c in final_state["chunks"]})

    return {
        "answer":     final_state["answer"],
        "sources":    sources,
        "attempts":   final_state["attempts"],
        "best_score": round(final_state["best_score"], 4),
        "reasoning":  final_state["reasoning"],
        "chunks_used": len(final_state["chunks"]),
    }
```

`agent.invoke(initial_state)` runs the graph to completion and returns the
final state. Every field in the returned dict reflects the last value written
by any node.

---

## 8. Execution Trace

For the query "Who created FastAPI?", assuming first retrieval scores are poor
and second are good:

```
t=0   graph.invoke({"question": "Who created FastAPI?", "attempts": 0, ...})

t=1   retrieve_node executes
      → query = "Who created FastAPI?"  (no rephrased question yet)
      → chunks = [score=0.41, score=0.38, score=0.35, score=0.31, score=0.28]
      → best_score = 0.41
      → attempts = 1
      state: {chunks: [...], best_score: 0.41, attempts: 1, reasoning: ["Attempt 1: ..."]}

t=2   evaluate_node executes
      → 0.41 < 0.50 (SCORE_THRESHOLD) AND 1 < 2 (MAX_ATTEMPTS)
      → quality_ok = False
      state: {quality_ok: False, reasoning: [..., "Quality FAILED..."]}

t=3   route_after_evaluate(state) returns "rephrase"
      → routing to rephrase_node

t=4   rephrase_node executes
      → GPT call: "Who created FastAPI?" → "What is the creator and history of FastAPI framework?"
      state: {rephrased_question: "What is the creator and history of FastAPI framework?"}

t=5   retrieve_node executes again (loop!)
      → query = "What is the creator and history of FastAPI framework?"
      → chunks = [score=0.91, score=0.85, score=0.72, score=0.68, score=0.55]
      → best_score = 0.91
      → attempts = 2
      state: {chunks: [...], best_score: 0.91, attempts: 2}

t=6   evaluate_node executes
      → 0.91 >= 0.50 → quality_ok = True
      state: {quality_ok: True, reasoning: [..., "Quality PASSED..."]}

t=7   route_after_evaluate(state) returns "generate"
      → routing to generate_node

t=8   generate_node executes
      → GPT generates answer from the high-quality chunks
      state: {answer: "FastAPI was created by Sebastián Ramírez..."}

t=9   graph hits END
      → invoke() returns final_state
```

The reasoning list in the returned state documents every decision —
invaluable for debugging why the agent took a particular path.

---

## 9. Checkpointing — Pause and Resume

LangGraph supports checkpointing — persisting state at each node so execution
can be paused and resumed. This enables:

- Long-running agents that span multiple HTTP requests
- Human-in-the-loop approval gates (pause, await human, resume)
- Fault recovery (resume from last checkpoint after crash)

```python
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()   # in-memory (dev)
# For production: use SqliteSaver, PostgresSaver, or RedisSaver

agent = graph.compile(checkpointer=checkpointer)

# First invocation with thread_id
config = {"configurable": {"thread_id": "user-session-123"}}
state  = agent.invoke(initial_state, config=config)

# Resume later — state is automatically loaded from checkpoint
state  = agent.invoke({"question": "follow-up question"}, config=config)
```

With checkpointing, the agent maintains state across calls identified by
`thread_id`. This is how multi-turn agentic conversations work — not by
passing full history, but by persisting and resuming a graph.

---

## 10. Interrupt — Human-in-the-Loop

LangGraph can pause graph execution at specified nodes and wait for human
input before continuing. This is essential for approval-gated operations
(e.g., "the agent wants to delete data — approve?"):

```python
graph.compile(
    checkpointer=checkpointer,
    interrupt_before=["delete_node"],   # pause BEFORE this node runs
)

# First call — runs until it reaches delete_node, then pauses
state = agent.invoke(initial_state, config=config)
print("Agent wants to delete. Approve? (y/n)")

# Human approves — resume
agent.invoke(None, config=config)   # None = continue from checkpoint
```

The graph serializes its state to the checkpointer, returns control to
the caller, and resumes from the exact paused node when reinvoked. No
state is lost between pause and resume.

---

## 11. Streaming Node Output

```python
# Stream node-by-node updates as they happen
for chunk in agent.stream(initial_state):
    # chunk is {node_name: partial_state_update}
    node_name = list(chunk.keys())[0]
    update    = chunk[node_name]
    print(f"[{node_name}] {update}")

# Output:
# [retrieve] {'chunks': [...], 'best_score': 0.41, 'attempts': 1, ...}
# [evaluate] {'quality_ok': False, 'reasoning': [...]}
# [rephrase] {'rephrased_question': 'What is the creator...'}
# [retrieve] {'chunks': [...], 'best_score': 0.91, 'attempts': 2, ...}
# [evaluate] {'quality_ok': True, 'reasoning': [...]}
# [generate] {'answer': 'FastAPI was created by...', 'reasoning': [...]}
```

This is how you build real-time UIs that show the agent's thinking as it
works — each node output is streamed as a Server-Sent Event.

---

## 12. LangGraph vs Manual Agent Loop

The `tool_agent.py` in this project implements a manual agent loop:

```python
# app/agents/tool_agent.py — manual while loop
while iterations < max_iterations:
    response = _client.chat.completions.create(...)
    if response.choices[0].message.tool_calls:
        # execute tools, append results to messages, continue
    else:
        return {"answer": response.choices[0].message.content}
```

This works but has significant limitations at scale:

| Capability | Manual Loop | LangGraph |
|---|---|---|
| Branching logic | Embedded in code (fragile) | Explicit graph edges (auditable) |
| Checkpointing / pause-resume | Must implement manually | Built-in |
| Human-in-the-loop | Must implement manually | Built-in |
| State inspection | Debug the running process | Query the state object |
| Streaming | Must implement SSE manually | `.stream()` built-in |
| Parallel branches | Complex async code | `Send` API |
| Testing | Mock every LLM call | Inject state at any node |
| Observability | Add logging everywhere | LangSmith traces every node |

Use the manual loop for simple, linear tool-use. Use LangGraph for agents
with complex control flow, retries, approval gates, or multi-turn state.

---

## 13. Graph Patterns

### Linear Chain

```
A → B → C → END
```
Use when: sequence is fixed, no branching needed.

### Retry Loop (This Project)

```
START → Retrieve → Evaluate → rephrase → Retrieve → ...
                           ↘ generate → END
```
Use when: need quality gates with automatic retry.

### Parallel Branches

```python
# Using Send API for parallel execution
from langgraph.constants import Send

def fan_out(state):
    return [Send("process", {"item": item}) for item in state["items"]]

graph.add_conditional_edges("start", fan_out)
```

Use when: processing a list of items concurrently (e.g., summarize 10
documents in parallel).

### Subgraph

```python
subgraph = StateGraph(SubState).compile()
main_graph.add_node("sub", subgraph)
```

Use when: a complex step deserves its own graph (encapsulation, reuse).

---

## 14. Summary

LangGraph models agent logic as a directed graph where nodes are stateless
Python functions and state is an explicit typed dictionary. Fixed edges create
linear flow; conditional edges create branching and loops. The result is an
agent whose logic is visible (inspect the graph), auditable (reasoning log in
state), pausable (checkpointing), and streamable. Compared to a manual while
loop, LangGraph trades simplicity for correctness — the explicit state machine
prevents subtle bugs in complex control flow and makes the agent's decisions
transparent at every step.

**Next:** `05-putting-it-together.md` — how RAG, Milvus, LangChain, and
LangGraph combine into the full production service architecture, and the
design decisions behind each layer.
