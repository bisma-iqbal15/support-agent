import json
import unittest
from pathlib import Path

from agent.agent import Agent, parse_action
from agent.tools import build_tools
from tests.helpers import ScriptedLLM, temp_settings, tool

FINAL = {"final": "Done."}


def make_agent(replies, approver=None, **overrides):
    s = temp_settings(**overrides)
    return Agent(ScriptedLLM(replies), build_tools(s), s, approver), s


class ParseTests(unittest.TestCase):
    def test_valid_tool_and_final(self):
        self.assertEqual(parse_action('{"tool": "x", "args": {"a": "1"}}'), ("tool", "x", {"a": "1"}))
        self.assertEqual(parse_action('{"final": "hi"}'), ("final", "hi", None))

    def test_json_wrapped_in_text_is_recovered(self):
        self.assertEqual(parse_action('Sure! {"final": "hi"} thanks')[0], "final")

    def test_garbage_raises(self):
        for bad in ["hello", "[1, 2]", '{"nothing": 1}', '{"tool": "x", "args": "no"}']:
            with self.assertRaises(ValueError):
                parse_action(bad)


class AgentLoopTests(unittest.TestCase):
    def test_happy_path_late_refund_escalates(self):
        agent, s = make_agent([
            tool("lookup_order", order_id="1042"),
            tool("check_refund_eligibility", order_id="1042"),
            tool("create_ticket", summary="Late refund request for 1042", priority="medium"),
            {"final": "I've passed this to our team."},
        ])
        r = agent.run("Refund order 1042 please")
        self.assertEqual(r.status, "completed")
        self.assertEqual(r.tool_calls, ["lookup_order", "check_refund_eligibility", "create_ticket"])
        self.assertTrue(Path(s.tickets_path).exists())
        self.assertFalse(Path(s.refunds_path).exists())

    def test_invalid_json_is_recovered(self):
        agent, _ = make_agent(["this is not json", FINAL])
        r = agent.run("hi")
        self.assertEqual(r.status, "completed")
        self.assertEqual([e["type"] for e in r.events], ["llm", "error", "llm", "final"])

    def test_unknown_tool_gets_error_observation(self):
        agent, _ = make_agent([tool("delete_database"), FINAL])
        r = agent.run("hi")
        self.assertIn("Unknown tool", r.events[1]["result"]["error"])

    def test_missing_args_are_reported(self):
        agent, _ = make_agent([tool("lookup_order"), FINAL])
        self.assertIn("Missing arguments", agent.run("hi").events[1]["result"]["error"])

    def test_risky_tool_denied_by_default(self):
        agent, s = make_agent([tool("issue_refund", order_id="1043"), FINAL])
        r = agent.run("refund 1043")
        self.assertIn("DENIED", r.events[1]["result"]["error"])
        self.assertFalse(Path(s.refunds_path).exists())

    def test_risky_tool_runs_when_approved_and_eligible(self):
        agent, s = make_agent([tool("issue_refund", order_id="1043"), FINAL], approver=lambda n, a: True)
        agent.run("refund 1043")
        self.assertIn("1043", json.loads(Path(s.refunds_path).read_text()))

    def test_approved_but_ineligible_is_still_refused_by_policy(self):
        agent, s = make_agent([tool("issue_refund", order_id="1042"), FINAL], approver=lambda n, a: True)
        r = agent.run("refund 1042")
        self.assertIn("Refused", r.events[1]["result"]["error"])
        self.assertFalse(Path(s.refunds_path).exists())

    def test_duplicate_call_is_blocked(self):
        agent, _ = make_agent([tool("lookup_order", order_id="1042")] * 2 + [FINAL])
        r = agent.run("hi")
        self.assertIn("already made this exact call", r.events[3]["result"]["error"])
        self.assertEqual(r.tool_calls, ["lookup_order"])

    def test_step_limit_creates_fallback_ticket(self):
        agent, s = make_agent(["not json"], max_steps=3)
        r = agent.run("hi")
        self.assertEqual(r.status, "max_steps")
        self.assertIn("T-0001", r.final)
        self.assertEqual(len(json.loads(Path(s.tickets_path).read_text())), 1)

    def test_tool_crash_does_not_crash_agent(self):
        agent, _ = make_agent([tool("lookup_order", order_id="1042"), FINAL])
        agent.tools["lookup_order"].fn = lambda order_id: 1 / 0
        r = agent.run("hi")
        self.assertIn("failed", r.events[1]["result"]["error"])
        self.assertEqual(r.status, "completed")


if __name__ == "__main__":
    unittest.main()
