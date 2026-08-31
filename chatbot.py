"""
chatbot.py
==========
A multi-turn customer-support chatbot with tool calling,
using Claude Sonnet through the Anthropic API.

Tools:
  - get_order_status(order_id)
  - get_refund_policy(category)

chat() returns:
  - reply
  - history
  - tools_called
"""

from dotenv import load_dotenv
load_dotenv()

import json
from anthropic import Anthropic


# ---------------------------------------------------------------------------
# Claude client
# ---------------------------------------------------------------------------

client = Anthropic()


# ---------------------------------------------------------------------------
# In-memory data
# ---------------------------------------------------------------------------

ORDERS = {
    "ORD-1042": {"status": "Shipped", "eta": "2026-05-13"},
    "ORD-2099": {"status": "Delivered", "eta": "2026-05-08"},
    "ORD-7777": {"status": "Processing", "eta": "2026-05-15"},
}


REFUND_POLICIES = {
    "electronics": "Electronics can be returned within 15 days, unopened.",
    "clothing": "Clothing can be returned within 30 days with tags attached.",
    "food": "Food items are non-returnable for safety reasons.",
    "furniture": "Furniture can be returned within 30 days if unassembled.",
}


# ---------------------------------------------------------------------------
# Tool definitions for Claude
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "name": "get_order_status",
        "description": (
            "Look up the shipping status of a customer order by its ID."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "The order ID, e.g. ORD-1042",
                }
            },
            "required": ["order_id"],
        },
    },
    {
        "name": "get_refund_policy",
        "description": (
            "Return the refund/return policy for a product category."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": (
                        "Product category, e.g. electronics, clothing, food"
                    ),
                }
            },
            "required": ["category"],
        },
    },
]


# ---------------------------------------------------------------------------
# Tool executor
# ---------------------------------------------------------------------------

def _execute_tool(name: str, args: dict) -> str:

    if name == "get_order_status":

        order_id = args.get("order_id", "").upper()

        order = ORDERS.get(order_id)

        if not order:
            return f"No order found with ID {order_id}."

        return (
            f"Order {order_id} is {order['status']}. "
            f"ETA: {order['eta']}."
        )

    if name == "get_refund_policy":

        category = args.get("category", "").lower()

        policy = REFUND_POLICIES.get(category)

        if not policy:
            return f"No refund policy on file for '{category}'."

        return policy

    return f"Unknown tool: {name}"


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """
You are a friendly and professional customer-support chatbot
for ShopEasy, an online retail store.

You have access to two tools:

- get_order_status:
  Use whenever a customer asks about a specific order.

- get_refund_policy:
  Use whenever a customer asks about returns or refunds.

Rules:

- Always use the tools when you have the information needed to call them.
- Be concise and polite.
- Never discuss topics outside of ShopEasy customer support.
- Remember everything the customer tells you in the current conversation.
- If the customer refers to something mentioned earlier, use the conversation
  history to answer correctly.
"""


# ---------------------------------------------------------------------------
# chat()
# ---------------------------------------------------------------------------

def chat(
    user_message: str,
    history: list[dict],
) -> tuple[str, list[dict], list[dict]]:

    """
    Send one user message to Claude.

    Returns:

        reply:
            Final assistant response.

        history:
            Updated conversation history.

        tools_called:
            List of tools used during this turn.
    """

    # Add current user message
    history = history + [
        {
            "role": "user",
            "content": user_message,
        }
    ]

    tools_called = []

    # -----------------------------------------------------------------------
    # Tool-calling loop
    # -----------------------------------------------------------------------

    while True:

        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=history,
            tools=TOOLS,
        )

        # -------------------------------------------------------------------
        # Check whether Claude wants to call a tool
        # -------------------------------------------------------------------

        if response.stop_reason != "tool_use":

            # Claude has produced the final answer
            reply_parts = []

            for block in response.content:

                if block.type == "text":
                    reply_parts.append(block.text)

            reply = "".join(reply_parts)

            # Add assistant response to history
            history = history + [
                {
                    "role": "assistant",
                    "content": response.content,
                }
            ]

            return reply, history, tools_called

        # -------------------------------------------------------------------
        # Claude requested one or more tools
        # -------------------------------------------------------------------

        # Add Claude's response containing the tool_use block
        history = history + [
            {
                "role": "assistant",
                "content": response.content,
            }
        ]

        tool_results = []

        for block in response.content:

            if block.type != "tool_use":
                continue

            tool_name = block.name
            tool_input = block.input

            # Execute our local tool
            result = _execute_tool(
                tool_name,
                tool_input
            )

            # Store tool information for DeepEval/debugging
            tools_called.append(
                {
                    "name": tool_name,
                    "args": tool_input,
                    "result": result,
                }
            )

            # Claude requires tool results in this format
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                }
            )

        # Add tool results to conversation history
        history = history + [
            {
                "role": "user",
                "content": tool_results,
            }
        ]


# ---------------------------------------------------------------------------
# Standalone interactive demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    print(
        "ShopEasy Support Chatbot "
        "(Claude Sonnet + Tools)"
    )

    print("Type 'quit' to exit.\n")

    history = []

    while True:

        user_input = input("You: ").strip()

        if user_input.lower() in ("quit", "exit"):
            break

        reply, history, tools = chat(
            user_input,
            history
        )

        # Print tools used
        if tools:

            for tool in tools:

                print(
                    f"  [tool] "
                    f"{tool['name']}({tool['args']}) "
                    f"→ {tool['result']}"
                )

        print(f"Bot: {reply}\n")