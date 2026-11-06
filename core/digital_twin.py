"""
Digital twin: re-plays the same earliest-due-date dispatch logic as the
baseline scheduler, but through a real discrete-event simulation (SimPy)
instead of hand-rolled bookkeeping. Machines are modeled as single-capacity
resources with real queueing, so two jobs can never land on the same
machine at the same time by construction.

If this produces (close to) the same numbers as the baseline, that's our
proof the baseline plan is physically feasible, not just correct on paper.
This also lays the groundwork for injecting disruptions later: a machine
breakdown just becomes another event competing for the same resource.
"""

import simpy
import pandas as pd

from core.scheduler import _price_and_carbon_at


def run_digital_twin(orders: pd.DataFrame, routing: pd.DataFrame, machines: pd.DataFrame,
                      energy: pd.DataFrame, blocked_windows: dict = None) -> pd.DataFrame:
    blocked_windows = blocked_windows or {}
    horizon_start = orders["created_at"].min()
    env = simpy.Environment()

    machine_res = {m: simpy.PriorityResource(env, capacity=1) for m in machines["machine_id"]}
    machine_free_at = {m: 0.0 for m in machines["machine_id"]}
    machine_info = machines.set_index("machine_id")

    orders_sorted = orders.sort_values("due_date").reset_index(drop=True)
    priority_rank = {row["order_id"]: rank for rank, row in orders_sorted.iterrows()}

    rows = []

    def order_process(order):
        yield env.timeout((order["created_at"] - horizon_start).total_seconds() / 3600.0)
        steps = routing[routing["product_id"] == order["product_id"]].sort_values("step_order")

        for _, step in steps.iterrows():
            candidates = machines[machines["capability"] == step["capability"]]["machine_id"].tolist()
            chosen = min(candidates, key=lambda m: machine_free_at[m])
            m_row = machine_info.loc[chosen]
            duration_h = (order["quantity"] * step["std_time_min_per_unit"]) / 60.0

            with machine_res[chosen].request(priority=priority_rank[order["order_id"]]) as req:
                yield req
                start_h = env.now
                yield env.timeout(duration_h)
                end_h = env.now

            machine_free_at[chosen] = end_h

            start_ts = horizon_start + pd.Timedelta(hours=start_h)
            end_ts = horizon_start + pd.Timedelta(hours=end_h)
            price, carbon = _price_and_carbon_at(energy, start_ts)
            energy_kwh = duration_h * m_row["energy_kw_active"]

            rows.append(dict(
                order_id=order["order_id"],
                product_id=order["product_id"],
                step_order=step["step_order"],
                capability=step["capability"],
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

    def breakdown_process(machine_id, start_h, duration_h):
        yield env.timeout(start_h)
        with machine_res[machine_id].request(priority=-1) as req:
            yield req
            yield env.timeout(duration_h)
        machine_free_at[machine_id] = env.now

    for _, order in orders.iterrows():
        env.process(order_process(order))

    for machine_id, windows in blocked_windows.items():
        for w_start, w_end in windows:
            start_h = (w_start - horizon_start).total_seconds() / 3600.0
            duration_h = (w_end - w_start).total_seconds() / 3600.0
            env.process(breakdown_process(machine_id, start_h, duration_h))

    env.run()

    return pd.DataFrame(rows).sort_values(["order_id", "step_order"]).reset_index(drop=True)
