import argparse
from datetime import datetime
from pathlib import Path

from agent.agent import Agent
from agent.config import Settings
from agent.llm import LLMError, make_ollama_llm
from agent.tools import build_tools


def ask_human(name: str, args: dict) -> bool:
    print(f"\n  ** APPROVAL NEEDED: {name}({args}) **")
    return input("  Approve? [y/N] ").strip().lower() == "y"


def main():
    ap = argparse.ArgumentParser(prog="support-agent")
    ap.add_argument("message", help="the customer's message, in quotes")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--auto-approve", action="store_true", help="approve risky actions (testing only)")
    mode.add_argument("--auto-deny", action="store_true", help="deny risky actions")
    args = ap.parse_args()

    s = Settings.from_env()
    approver = ask_human
    if args.auto_approve:
        approver = lambda n, a: True
    elif args.auto_deny:
        approver = lambda n, a: False

    Path(s.traces_dir).mkdir(exist_ok=True)
    trace = str(Path(s.traces_dir) / f"{datetime.now():%Y%m%d-%H%M%S}.jsonl")
    agent = Agent(make_ollama_llm(s), build_tools(s), s, approver, trace)

    try:
        result = agent.run(args.message)
    except LLMError as e:
        raise SystemExit(f"LLM error: {e}")

    print()
    for e in result.events:
        if e["type"] == "tool":
            print(f"step {e['step']}: {e['tool']}({e['args']}) -> {str(e['result'])[:200]}")
        elif e["type"] == "error":
            print(f"step {e['step']}: model error -> {e['error']}")
        elif e["type"] == "fallback_ticket":
            print(f"step {e['step']}: fallback ticket -> {e['result']}")
    print(f"\n[{result.status} in {result.steps} steps]")
    print(f"Reply to customer: {result.final}")
    print(f"Trace saved: {trace}")


if __name__ == "__main__":
    main()
