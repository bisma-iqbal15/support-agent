"""The tools the agent can call. Policy decisions live HERE, in plain code, not in the model."""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from agent.config import Settings


@dataclass
class Tool:
    name: str
    description: str
    params: dict[str, str]  # parameter name -> short description (all values are strings)
    fn: Callable[..., dict]
    risky: bool = False  # risky tools need human approval
    optional: tuple = ()  # parameters the model may leave out


def _read(path: str, default):
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default


def _write(path: str, data) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")


def _clean_id(order_id: str) -> str:
    return str(order_id).strip().lstrip("#").strip()


def build_tools(s: Settings) -> dict[str, Tool]:
    def lookup_order(order_id: str) -> dict:
        oid = _clean_id(order_id)
        order = _read(s.orders_path, {}).get(oid)
        if not order:
            return {"error": f"Order {oid} not found"}
        refunded = oid in _read(s.refunds_path, {})
        return {"order_id": oid, **order, "refund_already_issued": refunded}

    def check_refund_eligibility(order_id: str) -> dict:
        info = lookup_order(order_id)
        if "error" in info:
            return info
        days = info["days_since_purchase"]
        if info["refund_already_issued"]:
            return {"eligible": False, "days_since_purchase": days, "reason": "A refund was already issued"}
        if days <= s.refund_window_days:
            return {"eligible": True, "days_since_purchase": days,
                    "reason": f"Within the {s.refund_window_days}-day refund window"}
        return {"eligible": False, "days_since_purchase": days,
                "reason": f"Purchased {days} days ago; the refund window is {s.refund_window_days} days. "
                          "Late refunds are only possible after a long outage, so a human must review it."}

    def issue_refund(order_id: str) -> dict:
        # Defense in depth: re-check the policy in code even though a human approved this.
        check = check_refund_eligibility(order_id)
        if not check.get("eligible"):
            return {"error": f"Refused by policy check: {check.get('reason', check.get('error'))}"}
        oid = _clean_id(order_id)
        refunds = _read(s.refunds_path, {})
        order = _read(s.orders_path, {})[oid]
        refunds[oid] = {"amount_usd": order["amount_usd"]}
        _write(s.refunds_path, refunds)
        return {"refunded": True, "order_id": oid, "amount_usd": order["amount_usd"]}

    def create_ticket(summary: str, priority: str = "medium") -> dict:
        tickets = _read(s.tickets_path, [])
        priority = priority.lower() if priority.lower() in {"low", "medium", "high"} else "medium"
        ticket = {"ticket_id": f"T-{len(tickets) + 1:04d}", "summary": summary[:300],
                  "priority": priority, "status": "open"}
        tickets.append(ticket)
        _write(s.tickets_path, tickets)
        return ticket

    def search_docs(question: str) -> dict:
        import httpx

        try:
            r = httpx.post(f"{s.rag_url}/query", json={"question": question}, timeout=300)
            r.raise_for_status()
        except httpx.ConnectError:
            return {"error": f"Docs service not reachable at {s.rag_url}. Start the RAG project's server."}
        except httpx.HTTPError as e:
            return {"error": f"Docs service error: {e}"}
        data = r.json()
        return {"answer": data["answer"], "sources": sorted({x["source"] for x in data.get("sources", [])})}

    tools = [
        Tool("search_docs", "Look up company policies and how-to answers in the documentation.",
             {"question": "the customer's question"}, search_docs),
        Tool("lookup_order", "Get the details of an order.",
             {"order_id": "the order number, digits only"}, lookup_order),
        Tool("check_refund_eligibility", "Check whether an order can be refunded under the refund policy.",
             {"order_id": "the order number, digits only"}, check_refund_eligibility),
        Tool("issue_refund", "Refund an order. Needs human approval. Only for eligible orders.",
             {"order_id": "the order number, digits only"}, issue_refund, risky=True),
        Tool("create_ticket", "Create a ticket so a human agent follows up.",
             {"summary": "what the customer needs and what you found", "priority": "low, medium or high"},
             create_ticket, optional=("priority",)),
    ]
    return {t.name: t for t in tools}
