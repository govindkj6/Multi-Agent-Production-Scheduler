"""
Smarter scheduler, built with OR-Tools (CP-SAT).

Unlike the baseline (one fixed "most urgent first" rule), this looks at
every order and every machine choice at once, and searches for an
assignment that:
  - respects each order's routing (steps must run in sequence),
  - never double-books a machine,
  - minimizes priority-weighted lateness,
  - and, as a secondary nudge, prefers the lower-energy machine when a
    step has more than one eligible machine.

The solver only decides WHICH machine and WHEN -- it doesn't know about
time-of-day energy prices. Real cost/CO2 are computed afterwards with the
same hourly price lookup used by the baseline and the digital twin, so all
three methods are compared on equal footing.
"""

import pandas as pd
from ortools.sat.python import cp_model

from core.scheduler import _price_and_carbon_at

PRIORITY_WEIGHT = {"High": 50, "Medium": 10, "Low": 2}
ENERGY_WEIGHT = 1
TIME_LIMIT_SECONDS = 20.0


def build_optimized_schedule(orders: pd.DataFrame, routing: pd.DataFrame, machines: pd.DataFrame,
                              energy: pd.DataFrame, blocked_windows: dict = None) -> pd.DataFrame:
    blocked_windows = blocked_windows or {}
    horizon_start = orders["created_at"].min()
    horizon_minutes = int((orders["due_date"].max() - horizon_start).total_seconds() / 60) + 2 * 24 * 60
    for windows in blocked_windows.values():
        for _, w_end in windows:
            horizon_minutes = max(horizon_minutes, int((w_end - horizon_start).total_seconds() / 60) + 24 * 60)

    model = cp_model.CpModel()
    machine_info = machines.set_index("machine_id")

    op_vars = {}                # (order_id, step_order) -> (start_var, end_var, duration_min, capability)
    presence_vars = {}          # (order_id, step_order, machine_id) -> BoolVar
    presence_by_machine = {m: [] for m in machines["machine_id"]}
    energy_terms = []
    tardiness_terms = []

    for machine_id, windows in blocked_windows.items():
        for w_start, w_end in windows:
            start_min = int((w_start - horizon_start).total_seconds() / 60)
            dur_min = int((w_end - w_start).total_seconds() / 60)
            fixed_interval = model.NewFixedSizeIntervalVar(start_min, dur_min, f"breakdown_{machine_id}_{start_min}")
            presence_by_machine[machine_id].append(fixed_interval)

    for _, order in orders.iterrows():
        steps = routing[routing["product_id"] == order["product_id"]].sort_values("step_order")
        release_min = int((order["created_at"] - horizon_start).total_seconds() / 60)
        due_min = int((order["due_date"] - horizon_start).total_seconds() / 60)

        prev_end_var = None
        last_end_var = None

        for _, step in steps.iterrows():
            duration_min = int(round(order["quantity"] * step["std_time_min_per_unit"]))
            key = (order["order_id"], step["step_order"])

            start_var = model.NewIntVar(release_min, horizon_minutes, f"start_{key}")
            end_var = model.NewIntVar(release_min, horizon_minutes, f"end_{key}")
            model.Add(end_var == start_var + duration_min)
            if prev_end_var is not None:
                model.Add(start_var >= prev_end_var)
            op_vars[key] = (start_var, end_var, duration_min, step["capability"])

            candidates = machines[machines["capability"] == step["capability"]]["machine_id"].tolist()
            presence_lits = []
            for m in candidates:
                presence = model.NewBoolVar(f"presence_{key}_{m}")
                interval = model.NewOptionalIntervalVar(start_var, duration_min, end_var, presence, f"iv_{key}_{m}")
                presence_by_machine[m].append(interval)
                presence_lits.append(presence)
                presence_vars[(order["order_id"], step["step_order"], m)] = presence

                energy_value = int(round((duration_min / 60.0) * machine_info.loc[m, "energy_kw_active"]))
                energy_terms.append(energy_value * presence)

            model.AddExactlyOne(presence_lits)

            prev_end_var = end_var
            last_end_var = end_var

        diff_var = model.NewIntVar(-horizon_minutes, horizon_minutes, f"diff_{order['order_id']}")
        model.Add(diff_var == last_end_var - due_min)
        tardiness_var = model.NewIntVar(0, horizon_minutes, f"tardy_{order['order_id']}")
        model.AddMaxEquality(tardiness_var, [0, diff_var])
        tardiness_terms.append(PRIORITY_WEIGHT[order["priority"]] * tardiness_var)

    for m, intervals in presence_by_machine.items():
        if intervals:
            model.AddNoOverlap(intervals)

    model.Minimize(sum(tardiness_terms) + ENERGY_WEIGHT * sum(energy_terms))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = TIME_LIMIT_SECONDS
    solver.parameters.num_search_workers = 8
    solver.parameters.random_seed = 42
    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError("OR-Tools could not find a feasible schedule within the time limit.")

    rows = []
    for _, order in orders.iterrows():
        steps = routing[routing["product_id"] == order["product_id"]].sort_values("step_order")
        for _, step in steps.iterrows():
            key = (order["order_id"], step["step_order"])
            start_var, end_var, duration_min, capability = op_vars[key]
            start_h = solver.Value(start_var) / 60.0
            duration_h = duration_min / 60.0

            candidates = machines[machines["capability"] == capability]["machine_id"].tolist()
            chosen = next(m for m in candidates if solver.Value(presence_vars[(order["order_id"], step["step_order"], m)]) == 1)
            m_row = machine_info.loc[chosen]

            start_ts = horizon_start + pd.Timedelta(hours=start_h)
            end_ts = start_ts + pd.Timedelta(hours=duration_h)
            price, carbon = _price_and_carbon_at(energy, start_ts)
            energy_kwh = duration_h * m_row["energy_kw_active"]

            rows.append(dict(
                order_id=order["order_id"],
                product_id=order["product_id"],
                step_order=step["step_order"],
                capability=capability,
                machine_id=chosen,
                start=start_ts,
                end=end_ts,
                duration_hours=round(duration_h, 2),
                energy_kwh=round(energy_kwh, 3),
                cost=round(energy_kwh * price, 3),
                co2_kg=round(energy_kwh * carbon / 1000.0, 3),
                due_date=order["due_date"],
                priority=order["priority"],
            ))

    schedule = pd.DataFrame(rows).sort_values(["order_id", "step_order"]).reset_index(drop=True)
    return schedule, status
