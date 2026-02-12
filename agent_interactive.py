"""
Interactive agentic financial assistant.

Claude autonomously decides which tools to call based on your questions.
Supports market data, portfolio, options, news, and order placement.

Works with BOTH:
  - Anthropic API key (native tool-use, most reliable)
  - Claude Max subscription via `claude` CLI (prompt-based tool calling)

Usage:
    python agent_interactive.py              # dry-run mode (orders simulated)
    python agent_interactive.py --live       # live mode (real orders!)
"""

import argparse
import json
import re
import sys

import config
from connect import get_session
from tools import TOOL_SCHEMAS, dispatch_tool

SYSTEM_PROMPT = """You are a sharp financial assistant for Indian equity markets (NSE/BSE).

You have access to tools that connect to a real brokerage (AngelOne SmartAPI). Use them
to answer the user's questions with live data. You can call multiple tools in sequence
to build a complete picture before responding.

Guidelines:
- When the user mentions a stock by name, use search_stock first to find the token,
  then use other tools with that token.
- Be concise and data-driven. Show numbers, not vague statements.
- For prices, use the Indian rupee symbol (₹) and format with commas.
- If a tool returns an error, tell the user what happened and suggest alternatives.
- For order placement: always confirm what you're about to do before calling place_order.
  In dry-run mode, orders are simulated — mention this clearly.
- Never fabricate data. If you don't have data, say so and offer to fetch it.
- When analyzing trends, reference specific numbers from the data you fetched."""

DEFAULT_MAX_TOOL_ROUNDS = 15


# ---------------------------------------------------------------------------
# Build tool descriptions for prompt-based mode
# ---------------------------------------------------------------------------

def _build_tool_descriptions() -> str:
    """Convert TOOL_SCHEMAS into a human-readable text block for the system prompt."""
    lines = []
    for tool in TOOL_SCHEMAS:
        lines.append(f"### {tool['name']}")
        lines.append(tool["description"])
        props = tool["input_schema"].get("properties", {})
        required = tool["input_schema"].get("required", [])
        if props:
            lines.append("Parameters:")
            for pname, pinfo in props.items():
                req = " (required)" if pname in required else " (optional)"
                desc = pinfo.get("description", "")
                ptype = pinfo.get("type", "string")
                enum_vals = pinfo.get("enum")
                enum_str = f" One of: {enum_vals}" if enum_vals else ""
                lines.append(f"  - {pname} ({ptype}{req}): {desc}{enum_str}")
        lines.append("")
    return "\n".join(lines)


CLI_SYSTEM_PROMPT = SYSTEM_PROMPT + """

## Available Tools

""" + _build_tool_descriptions() + """
## How to call tools

When you need data, respond with ONLY a JSON array wrapped in <tool_calls> tags:

<tool_calls>
[{"name": "tool_name", "input": {"param": "value"}}]
</tool_calls>

You can call multiple tools at once by putting multiple objects in the array.
After you receive the results, you can call more tools or give your final answer.

When you have your final answer, respond with plain text (no tags, no JSON wrapping).
IMPORTANT: Either output ONLY <tool_calls>...</tool_calls> OR ONLY your final text answer. Never both in the same response."""

TOOL_CALL_PATTERN = re.compile(r"<tool_calls>\s*(.*?)\s*</tool_calls>", re.DOTALL)


# ---------------------------------------------------------------------------
# SDK mode (native tool-use) — used when ANTHROPIC_API_KEY is available
# ---------------------------------------------------------------------------

def run_agent_loop_sdk(smart_api, client, dry_run: bool, max_rounds: int = DEFAULT_MAX_TOOL_ROUNDS):
    """REPL loop using the Anthropic SDK's native tool-use."""
    messages = []
    mode_label = "DRY-RUN" if dry_run else "LIVE"

    _print_banner(mode_label, "API Key")

    while True:
        user_input = _get_input()
        if user_input is None:
            break
        cmd = _handle_command(user_input, messages, mode_label, dry_run)
        if cmd == "quit":
            break
        if cmd == "handled":
            continue

        messages.append({"role": "user", "content": user_input})

        for _round in range(max_rounds):
            try:
                response = client.messages.create(
                    model=config.CLAUDE_MODEL,
                    max_tokens=4096,
                    system=SYSTEM_PROMPT,
                    tools=TOOL_SCHEMAS,
                    messages=messages,
                )
            except Exception as e:
                print(f"  [ERROR] API call failed: {e}")
                messages.pop()
                break

            assistant_content = response.content
            messages.append({"role": "assistant", "content": assistant_content})

            if response.stop_reason == "end_turn":
                for block in assistant_content:
                    if block.type == "text":
                        print(f"\nAgent: {block.text}\n")
                break

            if response.stop_reason == "tool_use":
                tool_results = []
                for block in assistant_content:
                    if block.type == "tool_use":
                        print(f"  [calling {block.name}...]")
                        result_json = dispatch_tool(
                            block.name, block.input, smart_api, dry_run=dry_run
                        )
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result_json,
                        })
                messages.append({"role": "user", "content": tool_results})
            else:
                for block in assistant_content:
                    if block.type == "text":
                        print(f"\nAgent: {block.text}\n")
                break
        else:
            print("  [Safety cap reached — too many tool rounds.]\n")


# ---------------------------------------------------------------------------
# CLI mode (prompt-based tool calling) — used with Claude Max subscription
# ---------------------------------------------------------------------------

def run_agent_loop_cli(smart_api, client, dry_run: bool, max_rounds: int = DEFAULT_MAX_TOOL_ROUNDS):
    """REPL loop using prompt-based tool calling via the claude CLI."""
    transcript = []  # list of text segments representing the conversation
    mode_label = "DRY-RUN" if dry_run else "LIVE"

    _print_banner(mode_label, "Claude CLI (Max)")

    while True:
        user_input = _get_input()
        if user_input is None:
            break
        cmd = _handle_command(user_input, transcript, mode_label, dry_run)
        if cmd == "quit":
            break
        if cmd == "handled":
            continue

        transcript.append(f"User: {user_input}")

        for _round in range(max_rounds):
            # Build full conversation as a single prompt
            prompt_text = "\n\n".join(transcript) + "\n\nAssistant:"

            try:
                response = client.messages.create(
                    model=config.CLAUDE_MODEL,
                    max_tokens=4096,
                    system=CLI_SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": prompt_text}],
                )
            except Exception as e:
                print(f"  [ERROR] Claude call failed: {e}")
                transcript.pop()
                break

            text = response.content[0].text.strip()

            # Check for tool calls
            match = TOOL_CALL_PATTERN.search(text)
            if match:
                try:
                    tool_calls = json.loads(match.group(1))
                except json.JSONDecodeError:
                    # Claude returned malformed JSON — treat as final answer
                    transcript.append(f"Assistant: {text}")
                    print(f"\nAgent: {text}\n")
                    break

                transcript.append(f"Assistant: {text}")

                # Execute each tool
                results = []
                for tc in tool_calls:
                    name = tc.get("name", "unknown")
                    inp = tc.get("input", {})
                    print(f"  [calling {name}...]")
                    result_str = dispatch_tool(name, inp, smart_api, dry_run=dry_run)
                    results.append({"tool": name, "result": json.loads(result_str)})

                transcript.append(f"Tool Results:\n{json.dumps(results, indent=2)}")
            else:
                # Final answer — no tool calls
                transcript.append(f"Assistant: {text}")
                print(f"\nAgent: {text}\n")
                break
        else:
            print("  [Safety cap reached — too many tool rounds.]\n")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _print_banner(mode_label: str, auth_method: str):
    print(f"\n{'=' * 55}")
    print(f"  Financial Agent — Interactive Mode ({mode_label})")
    print(f"  Auth: {auth_method}")
    print(f"{'=' * 55}")
    print(f"  Ask me anything about stocks, your portfolio, or markets.")
    print(f"  Commands: /clear (reset), /mode (show mode), /quit (exit)")
    if mode_label == "DRY-RUN":
        print(f"  Orders are SIMULATED. Use --live for real orders.")
    else:
        print(f"  WARNING: Orders are LIVE. Real money will be used!")
    print(f"{'=' * 55}\n")


def _get_input() -> str | None:
    try:
        return input("You: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nGoodbye!")
        return None


def _handle_command(user_input: str, state: list, mode_label: str, dry_run: bool) -> str:
    """Handle slash commands. Returns 'quit', 'handled', or 'not_command'."""
    lower = user_input.lower()
    if lower == "/quit":
        print("Goodbye!")
        return "quit"
    if lower == "/clear":
        state.clear()
        print("  [Conversation cleared]\n")
        return "handled"
    if lower == "/mode":
        print(f"  Mode: {mode_label}")
        print(f"  Orders: {'SIMULATED (safe)' if dry_run else 'LIVE (real money!)'}\n")
        return "handled"
    if not user_input:
        return "handled"
    return "not_command"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Interactive financial agent with tool use")
    parser.add_argument("--live", action="store_true",
                        help="Enable LIVE order placement (default is dry-run)")
    parser.add_argument("--max-rounds", type=int, default=DEFAULT_MAX_TOOL_ROUNDS,
                        help=f"Max tool-call rounds per query (default: {DEFAULT_MAX_TOOL_ROUNDS})")
    args = parser.parse_args()

    dry_run = not args.live

    # Get the right client
    client = config.get_anthropic_client()
    use_sdk = config.ANTHROPIC_API_KEY is not None

    if use_sdk:
        print("Using Anthropic API key (native tool-use)")
    else:
        print("Using Claude CLI (Max subscription, prompt-based tools)")

    # Connect to SmartAPI
    print("Connecting to AngelOne SmartAPI...")
    smart_api = get_session()

    # Start the right REPL
    if use_sdk:
        run_agent_loop_sdk(smart_api, client, dry_run=dry_run, max_rounds=args.max_rounds)
    else:
        run_agent_loop_cli(smart_api, client, dry_run=dry_run, max_rounds=args.max_rounds)


if __name__ == "__main__":
    main()
