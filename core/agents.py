"""
Specialist agents: each one looks at a finished schedule from a single
narrow concern and returns a 0-100 score. None of them build the schedule
-- that's the scheduler/optimizer's job. They only judge it.
"""

import pandas as pd

PRIORITY_WEIGHT = {"High": 3, "Medium": 2, "Low": 1}


def order_agent_score(schedule: pd.DataFrame) -> float:
    per_order_end = schedule.groupby("order_id")["end"].max()
    per_order_due = schedule.groupby("order_id")["due_date"].first()
    per_order_priority = schedule.groupby("order_id")["priority"].first()

    on_time = (per_order_end <= per_order_due).astype(int)
    weights = per_order_priority.map(PRIORITY_WEIGHT)
    return round(100 * (on_time * weights).sum() / weights.sum(), 1)


def energy_agent_score(schedule: pd.DataFrame, energy: pd.DataFrame) -> float:
    avg_price_paid = schedule["cost"].sum() / schedule["energy_kwh"].sum()
    lo, hi = energy["price_per_kwh"].min(), energy["price_per_kwh"].max()
    if hi == lo:
        return 100.0
    score = 100 * (1 - (avg_price_paid - lo) / (hi - lo))
    return round(min(100.0, max(0.0, score)), 1)


def sustainability_agent_score(schedule: pd.DataFrame, energy: pd.DataFrame) -> float:
    avg_intensity = schedule["co2_kg"].sum() * 1000 / schedule["energy_kwh"].sum()
    lo, hi = energy["grid_carbon_intensity_g_per_kwh"].min(), energy["grid_carbon_intensity_g_per_kwh"].max()
    if hi == lo:
        return 100.0
    score = 100 * (1 - (avg_intensity - lo) / (hi - lo))
    return round(min(100.0, max(0.0, score)), 1)


def machine_agent_score(schedule: pd.DataFrame, machines: pd.DataFrame) -> float:
    busy_hours = schedule.groupby("machine_id")["duration_hours"].sum()
    busy_hours = busy_hours.reindex(machines["machine_id"], fill_value=0.0)
    if busy_hours.mean() == 0:
        return 0.0
    cv = busy_hours.std() / busy_hours.mean()
    score = 100 * max(0.0, 1 - cv)
    return round(min(100.0, score), 1)
