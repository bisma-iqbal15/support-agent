"""The agent loop: model picks a tool -> our code validates and runs it -> result goes back to the model."""
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from agent.config import Settings
from agent.prompts import build_system_prompt
from agent.tools import Tool

OBS_LIMIT = 1500  # keep observations short so a small model's context isn't flooded


@dataclass
class AgentResult:
    final: str
    status: str  # "completed" or "max_steps"
    steps: int
    tool_calls: list[str] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)


def parse_action(text: str):
    """Return ('final', message, None) or ('tool', name, args). Raises ValueError if malformed."""
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            raise ValueError("Reply was not valid JSON")
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            raise ValueError("Reply was not valid JSON")
    if not isinstance(obj, dict):
        raise ValueError("Reply must be a JSON object")
    if "final" in obj:
        return "final", str(obj["final"]), None
    if "tool" in obj:
        args = obj.get("args", {})
        if not isinstance(args, dict):
            raise ValueError('"args" must be an object')
        return "tool", str(obj["tool"]), args
    raise ValueError('Reply must contain either "tool" or "final"')


class Agent:
    def __init__(self, llm: Callable, tools: dict[str, Tool], settings: Settings,
                 approver: Callable[[str, dict], bool] | None = None, trace_file: str | None = None):
        self.llm, self.tools, self.s = llm, tools, settings
        self.approver = approver or (lambda name, args: False)  # safe default: deny risky actions
        self.trace_file = trace_file

    def _log(self, events: list, **event):
        events.append(event)
        if self.trace_file:
            Path(self.trace_file).parent.mkdir(parents=True, exist_ok=True)
            with open(self.trace_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(event) + "\n")

    def _observe(self, result: dict) -> str:
        return "Observation: " + json.dumps(result)[:OBS_LIMIT]

    def _run_tool(self, name: str, args: dict) -> dict:
        tool = self.tools.get(name)
        if tool is None:
            return {"error": f"Unknown tool '{name}'. Available tools: {', '.join(self.tools)}"}
        missing = [p for p in tool.params if p not in args and p not in tool.optional]
        if missing:
            return {"error": f"Missing arguments for {name}: {', '.join(missing)}"}
        clean = {k: str(v) for k, v in args.items() if k in tool.params}
        if tool.risky and not self.approver(name, clean):
            return {"error": "DENIED: a human did not approve this action. Do not retry it. "
                             "Create a ticket so a human can follow up."}
        try:
            return tool.fn(**clean)
        except Exception as e:  # a tool crash must never crash the agent
            return {"error": f"Tool {name} failed: {type(e).__name__}: {e}"}

    def run(self, user_message: str) -> AgentResult:
        messages = [{"role": "system", "content": build_system_prompt(self.tools)},
                    {"role": "user", "content": f"Customer: {user_message}"}]
        events: list[dict] = []
        tool_calls: list[str] = []
        seen: set[str] = set()

        for step in range(1, self.s.max_steps + 1):
            t0 = time.perf_counter()
            raw = self.llm(messages, True)
            ms = int((time.perf_counter() - t0) * 1000)
            self._log(events, step=step, type="llm", ms=ms, output=raw[:500])
            messages.append({"role": "assistant", "content": raw})

            try:
                kind, a, b = parse_action(raw)
            except ValueError as e:
                self._log(events, step=step, type="error", error=str(e))
                messages.append({"role": "user", "content": f"ERROR: {e}. Reply with ONE valid JSON object only."})
                continue

            if kind == "final":
                self._log(events, step=step, type="final", message=a)
                return AgentResult(a, "completed", step, tool_calls, events)

            key = a + json.dumps(b, sort_keys=True)
            if key in seen:
                result = {"error": "You already made this exact call. Use the earlier result or take a different step."}
            else:
                seen.add(key)
                tool_calls.append(a)
                result = self._run_tool(a, b)
            self._log(events, step=step, type="tool", tool=a, args=b, result=result)
            messages.append({"role": "user", "content": self._observe(result)})

        # Out of steps: never leave the customer hanging; hand off to a human.
        ticket_note = ""
        if "create_ticket" in self.tools:
            t = self.tools["create_ticket"].fn(summary=f"Agent could not finish: {user_message}", priority="medium")
            tool_calls.append("create_ticket")
            self._log(events, step=self.s.max_steps, type="fallback_ticket", result=t)
            ticket_note = f" (reference {t['ticket_id']})"
        final = f"I couldn't finish this myself, so I've passed it to a human agent{ticket_note}."
        return AgentResult(final, "max_steps", self.s.max_steps, tool_calls, events)
