# Multi-Agent Production Scheduler

An AI-driven decision-support system for production scheduling. It takes a factory's
machines, product routings, and incoming orders, and produces a schedule that balances
**on-time delivery, energy cost, sustainability (CO2), and machine load balance** --
with a natural-language front end and automatic disruption handling.

Built as a research/portfolio project exploring multi-agent AI for resilient,
energy-aware production planning.

## How it works

The system is split into a deterministic core (does the math) and an LLM shell
(handles language and explanation) -- the LLM never invents a number, it only
narrates numbers the core already computed.

```
Natural-language request
        |
        v
LLM Orchestrator (LangGraph)  -- parses the request into a structured order
        |
        v
Order Book  -- the running list of all orders, old and new
        |
        v
OR-Tools Optimizer  -- decides machine assignment + timing to minimize
        |             priority-weighted lateness and energy use
        v
Digital Twin (SimPy)  -- replays the plan as a real discrete-event
        |                simulation to confirm it's physically feasible
        v
Specialist Agents  -- Order / Energy / Sustainability / Machine agents each
        |             score the resulting schedule 0-100 from their own concern
        v
Coordinator  -- combines the agent scores into one weighted verdict
        |         (default: cost-effectiveness weighted highest)
        v
LLM Report Generator  -- writes a detailed, numbers-grounded explanation
        |
        v
Output: readable report + Gantt chart + downloadable Excel/PDF
```

A **baseline scheduler** (simplest "most urgent order first" rule) is also included
as a reference point, so the optimizer's improvement can be measured, not just assumed.

**Disruption handling**: machine breakdowns, rush orders, and late-material delays can
be injected from `data/disruption_events.csv`, and all three scheduling methods
re-plan around them -- this is what demonstrates *resilience*, not just optimization.

## Project structure

```
core/
  data_loader.py       loads the 5 CSV data files
  scheduler.py          baseline ("earliest due date") scheduler
  digital_twin.py       SimPy discrete-event feasibility check
  optimizer.py          OR-Tools CP-SAT scheduler
  agents.py             the 4 specialist scoring agents
  coordinator.py        combines agent scores into one verdict
  disruption.py         applies a disruption event to the order book
  llm_orchestrator.py   LangGraph graph: parse -> plan -> report
  report_export.py      builds downloadable Excel/PDF reports
app/
  app.py                the Streamlit web app
data/
  machines.csv, product_routing.csv, orders.csv,
  energy_price_calendar.csv, disruption_events.csv   synthetic test data
```

## Getting started

**Requirements**: Python 3.11+, an [Anthropic API key](https://console.anthropic.com/)
(for the natural-language order entry feature only -- the scheduler itself works
without one).

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your API key
copy .env.example .env          # Windows
cp .env.example .env            # macOS/Linux
# then edit .env and paste in: ANTHROPIC_API_KEY=sk-ant-...

# 4. Run the app
streamlit run app/app.py
```

Open the URL Streamlit prints (usually `http://localhost:8501`).

## Using the app

1. **New Order Request** -- type a request in plain English (e.g. *"I need 150 units
   of Bracket by Friday"*), pick a priority (required, no default), and submit. The
   order is added to the order book and the whole system re-plans around it.
2. **Disruption Scenario** -- optionally inject a machine breakdown, rush order, or
   late-material event before generating a schedule, to see how each method recovers.
3. **Generate Schedule** -- runs the baseline, digital twin, and optimizer side by
   side, with a Coordinator verdict and Gantt charts for each.
4. **Download** -- after a natural-language request, download the generated report
   as Excel or PDF.

## Why this matters

- **Measurable improvement, not just automation**: the OR-Tools optimizer is
  benchmarked against a plain rule-based baseline on identical data. In testing, it
  raised on-time delivery from 88% to 96% while using slightly less energy.
- **Resilience, with evidence**: under a late-material disruption, the baseline's
  on-time rate collapsed to 20% while the optimizer held at 92% -- a concrete,
  numbers-backed demonstration of why re-optimizing globally beats simple rules when
  things go wrong.
- **Explainable, not a black box**: every number in the final report is traceable to
  the deterministic core; the LLM only explains and narrates, it never computes.
- **Multi-objective by design**: a single schedule is scored from four independent
  angles (delivery, cost, sustainability, machine balance), surfacing trade-offs a
  single-metric system would hide -- e.g. the optimizer's best on-time result also
  happened to unbalance machine load, which only the Coordinator's holistic scoring
  caught.
- **Natural-language decision support**: a planner can request an order the way
  they'd describe it to a colleague, and get back a detailed, grounded report instead
  of a raw spreadsheet.

