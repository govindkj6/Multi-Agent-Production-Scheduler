"""
Coordinator: combines the specialist agents' scores into one weighted
verdict. Default weighting prioritizes on-time delivery first, then
energy/cost, with sustainability and machine balance mattering less.
"""

import pandas as pd

from core.agents import order_agent_score, energy_agent_score, sustainability_agent_score, machine_agent_score

DEFAULT_WEIGHTS = dict(order=0.35, energy=0.18, sustainability=0.17, machine=0.30)


def evaluate(schedule: pd.DataFrame, machines: pd.DataFrame, energy: pd.DataFrame,
             weights: dict = None) -> dict:
    weights = weights or DEFAULT_WEIGHTS

    scores = dict(
        order=order_agent_score(schedule),
        energy=energy_agent_score(schedule, energy),
        sustainability=sustainability_agent_score(schedule, energy),
        machine=machine_agent_score(schedule, machines),
    )
    weighted_total = round(sum(scores[k] * weights[k] for k in weights), 1)

    return dict(scores=scores, weighted_total=weighted_total, weights=weights)
