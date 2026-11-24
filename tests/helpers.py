import json
import shutil
import tempfile
from dataclasses import replace
from pathlib import Path

from agent.config import ROOT, Settings


def temp_settings(**overrides) -> Settings:
    """Settings that read the real sample orders but write tickets/refunds to a temp folder."""
    tmp = Path(tempfile.mkdtemp())
    shutil.copy(ROOT / "data" / "orders.json", tmp / "orders.json")
    s = Settings(orders_path=str(tmp / "orders.json"), tickets_path=str(tmp / "tickets.json"),
                 refunds_path=str(tmp / "refunds.json"), traces_dir=str(tmp / "traces"))
    return replace(s, **overrides) if overrides else s


class ScriptedLLM:
    """A fake model that returns pre-written replies, so the loop can be tested without AI."""

    def __init__(self, replies):
        self.replies = [r if isinstance(r, str) else json.dumps(r) for r in replies]
        self.calls = 0

    def __call__(self, messages, json_mode=True):
        reply = self.replies[min(self.calls, len(self.replies) - 1)]
        self.calls += 1
        return reply


def tool(name, **args):
    return {"tool": name, "args": args}
