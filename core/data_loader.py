"""
Data loader: reads the 5 CSV files from data/ and hands them back as
pandas DataFrames, with date columns properly parsed as real timestamps
(not just text) so the scheduler can do time math on them.
"""

from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_machines() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "machines.csv")


def load_routing() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "product_routing.csv")


def load_orders() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "orders.csv", parse_dates=["created_at", "due_date"])
    return df


def load_energy_calendar() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "energy_price_calendar.csv", parse_dates=["timestamp"])


def load_disruptions() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "disruption_events.csv", parse_dates=["start_time"])


ORDER_BOOK_PATH = DATA_DIR / "order_book.csv"


ORDER_BOOK_COLUMNS = ["order_id", "product_id", "quantity", "created_at", "due_date", "priority"]


def load_order_book() -> pd.DataFrame:
    """The live, growing order book -- separate from orders.csv (the original
    synthetic baseline, kept untouched for reproducible benchmark comparisons).
    Starts empty (no synthetic test orders) and persists across app restarts
    once real orders are submitted."""
    if ORDER_BOOK_PATH.exists():
        df = pd.read_csv(ORDER_BOOK_PATH, parse_dates=["created_at", "due_date"])
        if len(df) == 0:
            df["created_at"] = pd.to_datetime(df["created_at"])
            df["due_date"] = pd.to_datetime(df["due_date"])
        return df
    df = pd.DataFrame(columns=ORDER_BOOK_COLUMNS)
    df["created_at"] = pd.to_datetime(df["created_at"])
    df["due_date"] = pd.to_datetime(df["due_date"])
    return df


def save_order_book(orders: pd.DataFrame) -> None:
    orders.to_csv(ORDER_BOOK_PATH, index=False)


def load_all() -> dict:
    return dict(
        machines=load_machines(),
        routing=load_routing(),
        orders=load_orders(),
        energy=load_energy_calendar(),
        disruptions=load_disruptions(),
    )
