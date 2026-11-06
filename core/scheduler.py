"""
Baseline scheduler: the simplest possible rule -- "most urgent order first"
(Earliest Due Date, EDD). For each order, it walks through every production
step in order and places it on the first machine with the right capability
that becomes free soonest.

This exists as a reference point: later, a smarter optimizer has to beat
this to prove it's actually worth using.
"""

import pandas as pd


def _price_and_carbon_at(energy: pd.DataFrame, timestamp: pd.Timestamp) -> tuple[float, float]:
    hour = timestamp.floor("h")
    row = energy[energy["timestamp"] == hour]
    if row.empty:
        row = energy.iloc[(energy["timestamp"] - hour).abs().argsort()[:1]]
    return float(row["price_per_kwh"].iloc[0]), float(row["grid_carbon_intensity_g_per_kwh"].iloc[0])


def _push_past_blocked_windows(start: pd.Timestamp, end: pd.Timestamp, windows: list) -> tuple[pd.Timestamp, pd.Timestamp]:
    duration = end - start
    moved = True
    while moved:
        moved = False
        for w_start, w_end in windows:
            if start < w_end and end > w_start:
                start = w_end
                end = start + duration
                moved = True
    return start, end


def build_baseline_schedule(orders: pd.DataFrame, routing: pd.DataFrame, machines: pd.DataFrame,
                             energy: pd.DataFrame, blocked_windows: dict = None) -> pd.DataFrame:
    blocked_windows = blocked_windows or {}
    orders_sorted = orders.sort_values("due_date").reset_index(drop=True)

    machine_free_at = {m: pd.Timestamp.min for m in machines["machine_id"]}
    order_last_end = {}
    rows = []

    for _, order in orders_sorted.iterrows():
        steps = routing[routing["product_id"] == order["product_id"]].sort_values("step_order")
        prev_end = order["created_at"]

        for _, step in steps.iterrows():
            candidates = machines[machines["capability"] == step["capability"]]
            best_machine = min(candidates["machine_id"], key=lambda m: machine_free_at[m])

            m_row = machines[machines["machine_id"] == best_machine].iloc[0]
            duration_hours = (order["quantity"] * step["std_time_min_per_unit"]) / 60.0

            earliest_start = max(machine_free_at[best_machine], prev_end)
            start = earliest_start
            end = start + pd.Timedelta(hours=duration_hours)
            start, end = _push_past_blocked_windows(start, end, blocked_windows.get(best_machine, []))

            price, carbon = _price_and_carbon_at(energy, start)
            energy_kwh = duration_hours * m_row["energy_kw_active"]

            rows.append(dict(
                order_id=order["order_id"],
                product_id=order["product_id"],
                step_order=step["step_order"],
                capability=step["capability"],
                machine_id=best_machine,
                start=start,
                end=end,
                duration_hours=round(duration_hours, 2),
                energy_kwh=round(energy_kwh, 3),
                cost=round(energy_kwh * price, 3),
                co2_kg=round(energy_kwh * carbon / 1000.0, 3),
                due_date=order["due_date"],
                priority=order["priority"],
            ))

            machine_free_at[best_machine] = end
            prev_end = end

        order_last_end[order["order_id"]] = prev_end

    return pd.DataFrame(rows)


def summarize(schedule: pd.DataFrame) -> dict:
    per_order_end = schedule.groupby("order_id")["end"].max()
    per_order_due = schedule.groupby("order_id")["due_date"].first()
    on_time = (per_order_end <= per_order_due).sum()
    total_orders = len(per_order_end)

    return dict(
        total_cost=round(schedule["cost"].sum(), 2),
        total_energy_kwh=round(schedule["energy_kwh"].sum(), 2),
        total_co2_kg=round(schedule["co2_kg"].sum(), 2),
        makespan_hours=round((schedule["end"].max() - schedule["start"].min()).total_seconds() / 3600, 1),
        orders_on_time=int(on_time),
        orders_total=int(total_orders),
        on_time_pct=round(100 * on_time / total_orders, 1),
    )
