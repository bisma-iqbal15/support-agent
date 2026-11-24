import json
import unittest
from pathlib import Path

from agent.tools import build_tools
from tests.helpers import temp_settings


class ToolTests(unittest.TestCase):
    def setUp(self):
        self.s = temp_settings()
        self.t = build_tools(self.s)

    def test_lookup_order_handles_hash_and_unknown(self):
        self.assertEqual(self.t["lookup_order"].fn("#1042")["order_id"], "1042")
        self.assertIn("error", self.t["lookup_order"].fn("9999"))

    def test_eligibility_boundaries(self):
        elig = self.t["check_refund_eligibility"].fn
        self.assertTrue(elig("1043")["eligible"])       # 5 days
        self.assertTrue(elig("1044")["eligible"])       # exactly 14 days is still inside the window
        self.assertFalse(elig("1042")["eligible"])      # 21 days
        self.assertFalse(elig("1045")["eligible"])      # 40 days
        self.assertIn("error", elig("0000"))

    def test_issue_refund_rechecks_policy_in_code(self):
        self.assertIn("Refused", self.t["issue_refund"].fn("1042")["error"])
        self.assertTrue(self.t["issue_refund"].fn("1043")["refunded"])

    def test_cannot_refund_twice(self):
        self.t["issue_refund"].fn("1043")
        self.assertFalse(self.t["check_refund_eligibility"].fn("1043")["eligible"])

    def test_create_ticket_numbers_and_priority(self):
        a = self.t["create_ticket"].fn(summary="first", priority="HIGH")
        b = self.t["create_ticket"].fn(summary="second", priority="nonsense")
        self.assertEqual((a["ticket_id"], a["priority"]), ("T-0001", "high"))
        self.assertEqual((b["ticket_id"], b["priority"]), ("T-0002", "medium"))
        self.assertEqual(len(json.loads(Path(self.s.tickets_path).read_text())), 2)

    def test_only_issue_refund_is_risky(self):
        self.assertEqual([n for n, t in self.t.items() if t.risky], ["issue_refund"])


if __name__ == "__main__":
    unittest.main()
