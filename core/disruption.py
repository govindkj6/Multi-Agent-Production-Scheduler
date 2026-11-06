"""
Turns a row from disruption_events.csv into something the scheduler,
digital twin, and optimizer can actually react to:
  - machine_breakdown -> a blocked time window on one machine
  - rush_order        -> a new row appended to the order book
  - late_material     -> an existing order's earliest start pushed back
"""

import re
import pandas as pd


def parse_rush_order_details(details: str) -> dict:
    fields = dict(part.split("=") for part in details.split(", "))
    return dict(
        product_id=fields["product"],
        quantity=int(fields["qty"]),
        due_offset_hours=int(re.sub(r"[^\d]", "", fields["due"])),
        priority=fields["priority"],
    )


def apply_disruption(orders: pd.DataFrame, event: pd.Series) -> tuple[pd.DataFrame, dict]:
    """Returns (possibly-updated orders_df, blocked_windows dict)."""
    blocked_windows = {}
    orders = orders.copy()

    if event["type"] == "machine_breakdown":
        start = pd.Timestamp(event["start_time"])
        end = start + pd.Timedelta(hours=float(event["duration_hours"]))
        blocked_windows[event["target"]] = [(start, end)]

    elif event["type"] == "rush_order":
        details = parse_rush_order_details(event["details"])
        created_at = pd.Timestamp(event["start_time"])
        due_date = created_at + pd.Timedelta(hours=details["due_offset_hours"])
        new_id = f"RUSH_{event['event_id']}"
        new_row = dict(
            order_id=new_id,
            product_id=details["product_id"],
            quantity=details["quantity"],
            created_at=created_at,
            due_date=due_date,
            priority=details["priority"],
        )
        orders = pd.concat([orders, pd.DataFrame([new_row])], ignore_index=True)

    elif event["type"] == "late_material":
        delay = pd.Timedelta(hours=float(event["duration_hours"]))
        mask = orders["order_id"] == event["target"]
        orders.loc[mask, "created_at"] = orders.loc[mask, "created_at"].apply(lambda t: max(t, event["start_time"]) + delay)

    return orders, blocked_windows
