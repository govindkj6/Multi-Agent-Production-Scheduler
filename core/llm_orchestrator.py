"""
LangGraph orchestrator: the natural-language front door.

Flow: parse the user's free-text request into a structured order (LLM,
forced tool-call so the output is always valid JSON) -> add it to the
order book and re-run the deterministic core (optimizer + digital twin +
coordinator, all non-LLM) -> narrate the real, computed numbers into a
readable report (LLM, but it is only allowed to restate numbers it was
handed, never invent new ones).
"""

import os
import json
from typing import TypedDict, Optional

import pandas as pd
from anthropic import Anthropic
from langgraph.graph import StateGraph, END

from core.optimizer import build_optimized_schedule
from core.digital_twin import run_digital_twin
from core.scheduler import build_baseline_schedule, summarize
from core.coordinator import evaluate

MODEL = os.environ.get("ORCHESTRATOR_MODEL", "claude-haiku-5-5")
BASELINE_ORDER_COUNT = 25  # number of orders in the original synthetic data/orders.csv

PRODUCT_NAME_TO_ID = {
    "Gear Housing": "P1",
    "Bracket": "P2",
    "Panel": "P3",
    "Valve": "P4",
    "Frame": "P5",
}

EXTRACT_ORDER_TOOL = {
    "name": "extract_order",
    "description": "Extract a structured production order from the user's natural-language request.",
    "input_schema": {
        "type": "object",
        "properties": {
            "product_name": {"type": "string", "enum": list(PRODUCT_NAME_TO_ID.keys())},
            "quantity": {"type": "integer", "minimum": 1},
            "due_in_hours": {"type": "number", "description": "Hours from the reference 'now' until the due date/time."},
        },
        "required": ["product_name", "quantity", "due_in_hours"],
    },
}


WRITE_REPORTS_TOOL = {
    "name": "write_reports",
    "description": "Submit both the concise on-screen summary and the full detailed report.",
    "input_schema": {
        "type": "object",
        "properties": {
            "concise_summary": {"type": "string", "description": "Crisp, medium-length Markdown summary (5-7 sentences / a short paragraph plus a few bullets)."},
            "detailed_report": {"type": "string", "description": "Full multi-section Markdown report."},
        },
        "required": ["concise_summary", "detailed_report"],
    },
}


class OrchestratorState(TypedDict):
    user_prompt: str
    user_priority: str
    reference_now: pd.Timestamp
    orders: pd.DataFrame
    routing: pd.DataFrame
    machines: pd.DataFrame
    energy: pd.DataFrame
    parsed_order: Optional[dict]
    error: Optional[str]
    updated_orders: Optional[pd.DataFrame]
    schedule: Optional[pd.DataFrame]
    metrics: Optional[dict]
    verdict: Optional[dict]
    report: Optional[str]
    summary: Optional[str]
    new_order_id: Optional[str]
    new_order_on_time: Optional[bool]
    new_order_finish: Optional[pd.Timestamp]
    late_orders: Optional[list]
    busiest_machine: Optional[str]
    least_busy_machine: Optional[str]
    avg_price_paid: Optional[float]
    cheapest_price: Optional[float]
    priciest_price: Optional[float]
    comparison_note: Optional[str]


def _client() -> Anthropic:
    return Anthropic()


def _baseline_comparison_note(optimized_metrics: dict, baseline_metrics: dict) -> str:
    on_time_diff = optimized_metrics["on_time_pct"] - baseline_metrics["on_time_pct"]
    cost_diff_pct = (baseline_metrics["total_cost"] - optimized_metrics["total_cost"]) / baseline_metrics["total_cost"] * 100

    if on_time_diff > 0:
        on_time_phrase = f"**{on_time_diff:.1f} percentage points higher** on-time delivery"
    elif on_time_diff < 0:
        on_time_phrase = f"**{abs(on_time_diff):.1f} percentage points lower** on-time delivery"
    else:
        on_time_phrase = "the same on-time delivery"

    if cost_diff_pct > 0:
        cost_phrase = f"**{cost_diff_pct:.1f}% lower** cost"
    elif cost_diff_pct < 0:
        cost_phrase = f"**{abs(cost_diff_pct):.1f}% higher** cost"
    else:
        cost_phrase = "the same cost"

    return (
        f"**NB:** Compared to a simple rule-based baseline schedule on this same order book, this "
        f"AI-optimized plan achieves {on_time_phrase} ({optimized_metrics['on_time_pct']}% vs "
        f"{baseline_metrics['on_time_pct']}%) at {cost_phrase} (₹{optimized_metrics['total_cost']:,} "
        f"vs ₹{baseline_metrics['total_cost']:,})."
    )


def parse_order_node(state: OrchestratorState) -> dict:
    now = state["reference_now"]
    system = (
        f"You convert factory order requests into structured data. "
        f"The reference 'now' is {now.strftime('%A, %Y-%m-%d %H:%M')}. "
        f"Compute due_in_hours relative to this 'now'. "
        f"When the user gives a specific calendar date (e.g. '9th Nov 2026', 'November 9'), always "
        f"interpret it as the END of that day (23:59), never the start -- this applies even when that "
        f"date is the SAME calendar day as 'now'. For example, if 'now' is "
        f"{now.strftime('%Y-%m-%d')} 00:00 and the user says 'by {now.day} {now.strftime('%b %Y')}', that date "
        f"has not finished yet, so due_in_hours should be approximately 24 (end of that day), never 0 or "
        f"negative. due_in_hours must always be a positive number greater than 0. "
        f"Only these products exist: {', '.join(PRODUCT_NAME_TO_ID.keys())}. "
        f"Priority is provided separately by the user through a required selector -- ignore any priority "
        f"wording in the request itself."
    )
    client = _client()
    response = client.messages.create(
        model=MODEL,
        max_tokens=300,
        system=system,
        messages=[{"role": "user", "content": state["user_prompt"]}],
        tools=[EXTRACT_ORDER_TOOL],
        tool_choice={"type": "tool", "name": "extract_order"},
    )
    tool_use = next(b for b in response.content if b.type == "tool_use")
    parsed = tool_use.input

    if parsed["due_in_hours"] <= 0:
        return {"error": "The due date/time must be in the future relative to the current order book."}
    return {"parsed_order": parsed}


def run_pipeline_node(state: OrchestratorState) -> dict:
    parsed = state["parsed_order"]
    now = state["reference_now"]
    due_date = now + pd.Timedelta(hours=parsed["due_in_hours"])

    existing_ids = state["orders"]["order_id"].tolist()
    new_id = f"O{BASELINE_ORDER_COUNT + len(existing_ids) + 1:03d}"
    new_row = dict(
        order_id=new_id,
        product_id=PRODUCT_NAME_TO_ID[parsed["product_name"]],
        quantity=int(parsed["quantity"]),
        created_at=now,
        due_date=due_date,
        priority=state["user_priority"],
    )
    updated_orders = pd.concat([state["orders"], pd.DataFrame([new_row])], ignore_index=True)

    schedule, _status = build_optimized_schedule(updated_orders, state["routing"], state["machines"], state["energy"])
    # digital twin validates feasibility of the same priority logic (sanity check, not shown by default)
    run_digital_twin(updated_orders, state["routing"], state["machines"], state["energy"])

    metrics = summarize(schedule)
    verdict = evaluate(schedule, state["machines"], state["energy"])

    baseline_schedule = build_baseline_schedule(updated_orders, state["routing"], state["machines"], state["energy"])
    baseline_metrics = summarize(baseline_schedule)
    comparison_note = _baseline_comparison_note(metrics, baseline_metrics)

    new_order_rows = schedule[schedule["order_id"] == new_id]
    new_order_on_time = bool((new_order_rows["end"].max() <= due_date))

    per_order_end = schedule.groupby("order_id")["end"].max()
    per_order_due = schedule.groupby("order_id")["due_date"].first()
    per_order_priority = schedule.groupby("order_id")["priority"].first()
    late_mask = per_order_end > per_order_due
    late_orders = [
        dict(
            order_id=oid,
            priority=per_order_priority[oid],
            hours_late=round((per_order_end[oid] - per_order_due[oid]).total_seconds() / 3600, 1),
        )
        for oid in per_order_end[late_mask].index
    ]

    busy_hours = schedule.groupby("machine_id")["duration_hours"].sum().round(1)
    busiest_machine = busy_hours.idxmax()
    least_busy_machine = busy_hours.idxmin()

    avg_price_paid = round(schedule["cost"].sum() / schedule["energy_kwh"].sum(), 4)
    cheapest_price = round(state["energy"]["price_per_kwh"].min(), 4)
    priciest_price = round(state["energy"]["price_per_kwh"].max(), 4)

    return dict(
        updated_orders=updated_orders,
        schedule=schedule,
        metrics=metrics,
        verdict=verdict,
        new_order_id=new_id,
        new_order_on_time=new_order_on_time,
        new_order_finish=new_order_rows["end"].max(),
        late_orders=late_orders,
        busiest_machine=f"{busiest_machine} ({busy_hours[busiest_machine]}h busy)",
        least_busy_machine=f"{least_busy_machine} ({busy_hours[least_busy_machine]}h busy)",
        avg_price_paid=avg_price_paid,
        cheapest_price=cheapest_price,
        priciest_price=priciest_price,
        comparison_note=comparison_note,
    )


def generate_report_node(state: OrchestratorState) -> dict:
    parsed = state["parsed_order"]
    metrics = state["metrics"]
    verdict = state["verdict"]

    late_orders_text = (
        "\n".join(
            f"  - {o['order_id']} (priority {o['priority']}): {o['hours_late']}h late"
            for o in state["late_orders"]
        )
        if state["late_orders"] else "  - None, every order in the book is on time."
    )

    facts = (
        f"NEW ORDER\n"
        f"  Product: {parsed['product_name']} (id {PRODUCT_NAME_TO_ID[parsed['product_name']]})\n"
        f"  Quantity: {parsed['quantity']} units\n"
        f"  Priority (user-selected): {state['user_priority']}\n"
        f"  Requested due date: {parsed['due_in_hours']:.1f} hours from now\n"
        f"  Predicted finish time: {state['new_order_finish']}\n"
        f"  Will this specific order be on time: {state['new_order_on_time']}\n\n"
        f"WHOLE ORDER BOOK AFTER RE-PLANNING (includes this new order plus all prior orders)\n"
        f"  Total cost: ₹{metrics['total_cost']}\n"
        f"  Total energy: {metrics['total_energy_kwh']} kWh\n"
        f"  Total CO2: {metrics['total_co2_kg']} kg\n"
        f"  On-time orders: {metrics['orders_on_time']}/{metrics['orders_total']} ({metrics['on_time_pct']}%)\n"
        f"  Makespan: {metrics['makespan_hours']} hours\n\n"
        f"ORDERS CURRENTLY PROJECTED LATE\n{late_orders_text}\n\n"
        f"MACHINE UTILIZATION\n"
        f"  Busiest machine: {state['busiest_machine']}\n"
        f"  Least busy machine: {state['least_busy_machine']}\n\n"
        f"ENERGY PRICING CONTEXT (all amounts in Indian Rupees, INR)\n"
        f"  Average price actually paid per kWh: ₹{state['avg_price_paid']}\n"
        f"  Cheapest available price in the horizon: ₹{state['cheapest_price']}/kWh\n"
        f"  Most expensive available price in the horizon: ₹{state['priciest_price']}/kWh\n\n"
        f"COORDINATOR AGENT SCORES (0-100 each, weighted total out of 100)\n"
        f"  On-Time agent: {verdict['scores']['order']} (weight {verdict['weights']['order']*100:.0f}%)\n"
        f"  Energy/Cost agent: {verdict['scores']['energy']} (weight {verdict['weights']['energy']*100:.0f}%)\n"
        f"  Sustainability agent: {verdict['scores']['sustainability']} (weight {verdict['weights']['sustainability']*100:.0f}%)\n"
        f"  Machine Balance agent: {verdict['scores']['machine']} (weight {verdict['weights']['machine']*100:.0f}%)\n"
        f"  Weighted total: {verdict['weighted_total']}/100"
    )

    system = (
        "You write production-planning reports for a factory manager, using the AI decision-support "
        "system's own output. Use ONLY the facts given below -- never invent, estimate, or round a number "
        "that isn't directly stated; if something isn't given, say it isn't available. All currency amounts "
        "are in Indian Rupees (INR); use the Rupee symbol, not $.\n\n"
        "Produce TWO versions of the report, via the write_reports tool:\n\n"
        "1. concise_summary -- a crisp, medium-length screen summary. Not a one-liner and not the full "
        "detailed report: about 5-7 sentences or a short paragraph plus a few bullet points. Hit the most "
        "important points only -- will the new order be on time, the whole-book on-time rate, total cost, "
        "and the single biggest trade-off -- with the key numbers in bold.\n\n"
        "2. detailed_report -- the full report in Markdown, with these sections in this order:\n"
        "## New Order Summary -- restate what was requested and whether it will be met, with key numbers in bold.\n"
        "## Schedule & Delivery Impact -- effect on the whole order book's on-time rate and makespan; "
        "list any at-risk/late orders by ID with how late they are.\n"
        "## Cost & Energy Breakdown -- total cost, energy, average price paid vs. the cheapest/priciest "
        "available price, and what that implies.\n"
        "## Sustainability & Machine Balance -- CO2 total, and the busiest vs. least-busy machine.\n"
        "## Coordinator Verdict -- walk through each agent's score and its weight, explain in plain English "
        "why the weighted total ended up where it did, and state the central trade-off being made.\n"
        "## Recommendation -- one short, concrete suggestion for the factory manager, grounded only in the "
        "facts above.\n"
        "Bold every key number the first time it appears in each section. Use bullet points for lists."
    )
    client = _client()
    response = client.messages.create(
        model=MODEL,
        max_tokens=2600,
        thinking={"type": "disabled"},
        system=system,
        messages=[{"role": "user", "content": f"Facts:\n{facts}\n\nWrite both reports."}],
        tools=[WRITE_REPORTS_TOOL],
        tool_choice={"type": "tool", "name": "write_reports"},
    )
    tool_use = next(b for b in response.content if b.type == "tool_use")
    return {"summary": tool_use.input["concise_summary"], "report": tool_use.input["detailed_report"]}


def error_report_node(state: OrchestratorState) -> dict:
    return {"report": f"Could not process this request: {state['error']}"}


def build_graph():
    graph = StateGraph(OrchestratorState)
    graph.add_node("parse_order", parse_order_node)
    graph.add_node("run_pipeline", run_pipeline_node)
    graph.add_node("generate_report", generate_report_node)
    graph.add_node("error_report", error_report_node)

    graph.set_entry_point("parse_order")
    graph.add_conditional_edges(
        "parse_order",
        lambda s: "error" if s.get("error") else "ok",
        {"error": "error_report", "ok": "run_pipeline"},
    )
    graph.add_edge("run_pipeline", "generate_report")
    graph.add_edge("generate_report", END)
    graph.add_edge("error_report", END)
    return graph.compile()


def handle_user_request(user_prompt: str, priority: str, orders: pd.DataFrame, routing: pd.DataFrame,
                         machines: pd.DataFrame, energy: pd.DataFrame) -> dict:
    if priority not in ("High", "Medium", "Low"):
        raise ValueError("priority must be explicitly one of 'High', 'Medium', 'Low'")
    reference_now = orders["created_at"].max() if len(orders) > 0 else energy["timestamp"].min()
    app = build_graph()
    result = app.invoke({
        "user_prompt": user_prompt,
        "user_priority": priority,
        "reference_now": reference_now,
        "orders": orders,
        "routing": routing,
        "machines": machines,
        "energy": energy,
        "error": None,
    })
    return result
