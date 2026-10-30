"""Stockout projection, risk classification and replenishment policy (transparent, rule-based)."""
import math
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from src.forecasting.models import forecast_wabl

N_PLAN = 120  # planning array length (days)


@dataclass
class Params:
    horizon: int = 28          # days scanned for a stockout
    review_days: int = 7       # R: days between order reviews
    service_z: float = 1.65    # ~95% cycle service level (normal approximation)
    demand_mult: float = 1.0   # what-if demand multiplier
    extra_delay: int = 0       # what-if extra supplier delay (days)
    extra_stock: dict = field(default_factory=dict)  # what-if sku -> extra units on hand


def simulate_stock(on_hand: float, f: np.ndarray, arrivals: np.ndarray):
    """Day-by-day: arrivals land in the morning, then demand. Returns (first stockout day (1-indexed) or None, unmet units)."""
    s, unmet, first = float(on_hand), 0.0, None
    for d in range(len(f)):
        s += arrivals[d]
        sold = min(s, f[d]); short = f[d] - sold
        if short > 1e-9 and first is None: first = d + 1
        unmet += short; s -= sold
    return first, unmet


def supplier_delays(pos: pd.DataFrame, products: pd.DataFrame, asof) -> dict:
    """Mean observed lateness (days, >=0) per supplier using only POs received on/before asof (needs >=3 observations)."""
    if pos is None or pos.empty: return {}
    r = pos[pos["received_date"].notna() & (pos["received_date"] <= asof)].merge(products[["sku", "supplier_id"]], on="sku")
    if r.empty: return {}
    r = r.assign(late=(r["received_date"] - r["expected_date"]).dt.days.clip(lower=0))
    g = r.groupby("supplier_id")["late"].agg(["mean", "count"])
    return {k: float(v["mean"]) for k, v in g.iterrows() if v["count"] >= 3}


def round_order(qty: float, moq: float, pack: float) -> int:
    if qty <= 0: return 0
    q = max(qty, moq); pack = max(1, int(pack))
    return int(math.ceil(q / pack) * pack)


def to_wide(sales: pd.DataFrame) -> pd.DataFrame:
    w = sales.pivot_table(index="sale_date", columns="sku", values="units_sold", aggfunc="sum")
    full = pd.date_range(w.index.min(), w.index.max())
    return w.reindex(full).fillna(0.0)  # missing days = zero sales (documented assumption)


def analyze(sales_wide, products, suppliers, pos, on_hand, asof, p: Params = None, with_paths=False):
    """products: inventory table (sku, product_name, category, unit_price, supplier_id, moq, pack_size).
    on_hand: dict sku->units at end of `asof`. Uses only information available at `asof`."""
    p = p or Params(); asof = pd.Timestamp(asof)
    hist = sales_wide.loc[:asof]
    delays = supplier_delays(pos, products, asof)
    lead_map = suppliers.set_index("supplier_id")["lead_time_days"].to_dict()
    openpos = pd.DataFrame()
    if pos is not None and not pos.empty:
        m = pos["order_date"] <= asof
        openpos = pos[m & (pos["received_date"].isna() | (pos["received_date"] > asof))]
    rows, paths = [], {}
    for r in products.itertuples():
        s = hist[r.sku] if r.sku in hist.columns else pd.Series(dtype=float, index=hist.index[:0])
        base_f, sigma, method = forecast_wabl(s, N_PLAN)
        f = base_f * p.demand_mult; sigma = sigma * p.demand_mult
        n = len(s)
        avg7 = float(s.iloc[-7:].mean()) if n else 0.0; avg28 = float(s.iloc[-28:].mean()) if n else 0.0
        prev28 = float(s.iloc[-56:-28].mean()) if n >= 56 else float("nan")
        sd14 = float(s.iloc[-14:].std(ddof=0)) if n >= 14 else float("nan")
        sdprev = float(s.iloc[-42:-14].std(ddof=0)) if n >= 42 else float("nan")
        delay = delays.get(r.supplier_id, 0.0)
        L = int(lead_map.get(r.supplier_id, 7) + round(delay) + p.extra_delay)
        oh = float(on_hand.get(r.sku, 0)) + p.extra_stock.get(r.sku, 0)
        arr = np.zeros(N_PLAN); overdue = False; po_qty = 0.0; first_arrival = None
        for q in openpos[openpos["sku"] == r.sku].itertuples() if len(openpos) else []:
            day = (q.expected_date - asof).days + int(round(delay)) + p.extra_delay
            if day <= 0: overdue, day = True, 3   # overdue PO: assumed to arrive in 3 days (documented)
            day = min(day, N_PLAN)
            arr[day - 1] += q.quantity; po_qty += q.quantity
            first_arrival = day if first_arrival is None else min(first_arrival, day)
        H = p.horizon
        first, unmet = simulate_stock(oh, f[:H], arr[:H])
        cumF, cumA = np.cumsum(f), np.cumsum(arr)
        proj = lambda d: oh + (cumA[d - 1] - cumF[d - 1] if d > 0 else 0.0)
        safety = p.service_z * sigma * math.sqrt(L + p.review_days)
        tstar = next((t for t in range(1, N_PLAN + 1) if proj(t) < safety), None)
        order_qty, order_day, after = 0, None, None
        if tstar is not None:
            order_day = max(0, tstar - L)
            od = min(order_day, N_PLAN - 1)
            window = cumF[min(od + L + p.review_days, N_PLAN) - 1] - (cumF[od - 1] if od > 0 else 0.0)
            position = proj(od) + (cumA[-1] - (cumA[od - 1] if od > 0 else 0.0))
            order_qty = round_order(window + safety - position, r.moq, r.pack_size)
            if order_qty > 0: after = proj(min(od + L, N_PLAN)) + order_qty
            else: order_day = None
        risk = "Low"
        if first is not None:
            risk = "Critical" if first <= L else "High" if first <= L + p.review_days else "Watch"
        elif order_day == 0: risk = "Watch"
        ev = []
        if first is not None:
            ev.append(f"Projected stockout in {first} day(s) with on-hand {oh:.0f} and forecast ~{f[:min(first,14)].mean():.1f}/day.")
            if first <= L: ev.append(f"Stockout ({first}d) comes before a new order placed today could arrive ({L}d lead time).")
            if first_arrival is not None and first_arrival > first: ev.append(f"Next incoming PO arrives day {first_arrival}, after the projected stockout.")
        if avg28 > 0 and avg7 > 1.15 * avg28: ev.append(f"Last-7-day demand is {100*(avg7/avg28-1):.0f}% above the 28-day average.")
        if prev28 == prev28 and prev28 > 0 and avg28 > 1.15 * prev28: ev.append(f"28-day demand is {100*(avg28/prev28-1):.0f}% above the prior 28 days.")
        if avg28 > 0 and oh / avg28 < L: ev.append(f"Coverage {oh/avg28:.1f} days is shorter than effective lead time {L} days.")
        if overdue: ev.append("A purchase order is past its expected date and not received.")
        if delay >= 1: ev.append(f"Supplier has been late by {delay:.1f} day(s) on average; lead time adjusted.")
        if sd14 == sd14 and sdprev == sdprev and sdprev > 0 and sd14 > 1.3 * sdprev: ev.append("Demand variability has risen versus the prior period.")
        if p.demand_mult != 1 or p.extra_delay or p.extra_stock.get(r.sku): ev.append("What-if assumptions are applied.")
        if not ev: ev.append("No risk signals: coverage and incoming stock sustain forecast demand.")
        pass_ = r.unit_price
        rows.append(dict(sku=r.sku, product_name=r.product_name, category=r.category, supplier_id=r.supplier_id, on_hand=oh,
            avg_7=avg7, avg_28=avg28, forecast_14=float(f[:14].sum()), coverage_days=(oh / avg28 if avg28 > 0 else np.inf),
            lead_time_days=L, supplier_delay_est=delay, open_po_qty=po_qty, po_overdue=overdue,
            stockout_day=first, stockout_date=(asof + pd.Timedelta(days=first)) if first else pd.NaT,
            unmet_units=unmet, sales_at_risk=unmet * pass_, risk=risk, safety_stock=safety,
            reorder_qty=order_qty, order_date=(asof + pd.Timedelta(days=order_day)) if order_day is not None else pd.NaT,
            delivery_date=(asof + pd.Timedelta(days=order_day + L)) if order_day is not None else pd.NaT,
            projected_after_delivery=after, evidence=" | ".join(ev), method=method,
            fixed_rule_alert=bool(avg28 > 0 and oh / avg28 < 7), unit_price=r.unit_price))
        if with_paths: paths[r.sku] = dict(f=f[:H + 14], arr=arr[:H + 14], on_hand=oh, evidence=ev)
    df = pd.DataFrame(rows)
    return (df, paths) if with_paths else df
