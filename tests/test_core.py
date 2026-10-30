import numpy as np, pandas as pd
from src.inventory.engine import simulate_stock, round_order, analyze, Params, supplier_delays, to_wide
from src.forecasting.models import forecast_wabl, forecast_ma28, forecast_seasonal_naive, mae, wape
from src.data.validation import validate, cross_check
from src.data.generate import generate
from src.data.loader import load_bundled
from src.evaluation.benchmark import run_benchmark


def _mini(on_hand=100, demand=10, lead=5, pos=None, days=70, moq=1, pack=1):
    idx = pd.date_range("2026-01-01", periods=days)
    sales = pd.DataFrame({"sale_date": idx, "sku": "A", "units_sold": demand})
    inv = pd.DataFrame([dict(sku="A", product_name="A", category="c", unit_price=2.0, supplier_id="S", moq=moq, pack_size=pack, on_hand=on_hand)])
    sup = pd.DataFrame([dict(supplier_id="S", supplier_name="s", lead_time_days=lead)])
    pos = pos if pos is not None else pd.DataFrame(columns=["po_id", "sku", "order_date", "expected_date", "received_date", "quantity"])
    return to_wide(sales), inv, sup, pos, idx[-1]


def test_stockout_day_exact():
    first, unmet = simulate_stock(25, np.full(10, 10.0), np.zeros(10))
    assert first == 3 and abs(unmet - 75) < 1e-9

def test_incoming_stock_delays_stockout():
    arr = np.zeros(10); arr[1] = 100
    first, _ = simulate_stock(25, np.full(10, 10.0), arr)
    assert first is None

def test_round_order_moq_pack_and_no_negative():
    assert round_order(-5, 10, 6) == 0 and round_order(3, 24, 12) == 24 and round_order(25, 1, 12) == 36

def test_zero_demand_no_alert_no_divide_by_zero():
    w, inv, sup, pos, asof = _mini(on_hand=50, demand=0)
    r = analyze(w, inv, sup, pos, {"A": 50}, asof).iloc[0]
    assert r.risk == "Low" and r.reorder_qty == 0 and np.isinf(r.coverage_days)

def test_short_coverage_is_critical_and_orders():
    w, inv, sup, pos, asof = _mini(on_hand=30, demand=10, lead=5)
    r = analyze(w, inv, sup, pos, {"A": 30}, asof).iloc[0]
    assert r.risk == "Critical" and r.stockout_day == 4 and r.reorder_qty > 0 and r.order_date == asof

def test_incoming_po_removes_risk():
    po = pd.DataFrame([dict(po_id="P1", sku="A", order_date=pd.Timestamp("2026-03-09"), expected_date=pd.Timestamp("2026-03-12"), received_date=pd.NaT, quantity=500)])
    w, inv, sup, _, asof = _mini(on_hand=30, demand=10, lead=5, pos=po)
    r = analyze(w, inv, sup, po, {"A": 30}, asof).iloc[0]
    assert r.stockout_day is None or pd.isna(r.stockout_day)

def test_supplier_delay_uses_only_past_receipts():
    pos = pd.DataFrame([dict(po_id=f"P{i}", sku="A", order_date=pd.Timestamp("2026-01-01"), expected_date=pd.Timestamp("2026-01-05"),
                             received_date=pd.Timestamp("2026-01-08"), quantity=1) for i in range(3)])
    inv = pd.DataFrame([dict(sku="A", supplier_id="S")])
    assert supplier_delays(pos, inv, pd.Timestamp("2026-02-01"))["S"] == 3.0
    assert supplier_delays(pos, inv, pd.Timestamp("2026-01-06")) == {}

def test_whatif_delay_increases_lead_and_order():
    w, inv, sup, pos, asof = _mini(on_hand=200, demand=10, lead=5)
    a = analyze(w, inv, sup, pos, {"A": 200}, asof).iloc[0]
    b = analyze(w, inv, sup, pos, {"A": 200}, asof, Params(extra_delay=10)).iloc[0]
    assert b.lead_time_days == a.lead_time_days + 10 and b.reorder_qty >= a.reorder_qty

def test_forecast_short_history_fallback_and_empty():
    f, sg, m = forecast_wabl(pd.Series([5.0] * 10, index=pd.date_range("2026-01-01", periods=10)), 5)
    assert "fallback" in m and np.allclose(f, 5)
    assert forecast_wabl(pd.Series(dtype=float), 3)[0].sum() == 0

def test_no_future_leakage_in_forecast():
    idx = pd.date_range("2026-01-01", periods=80); s = pd.Series(np.arange(80.0), index=idx)
    a = forecast_wabl(s.iloc[:60], 7)[0]; s2 = s.copy(); s2.iloc[60:] = 9999
    assert np.allclose(a, forecast_wabl(s2.iloc[:60], 7)[0])

def test_metrics_zero_demand_safe():
    assert np.isnan(wape([0, 0], [1, 1])) and mae([1, 3], [2, 2]) == 1.0 and abs(wape([2, 2], [1, 3]) - 0.5) < 1e-9

def test_seasonal_naive_repeats_week():
    idx = pd.date_range("2026-01-01", periods=14); s = pd.Series(list(range(14)), index=idx, dtype=float)
    assert list(forecast_seasonal_naive(s, 7)) == [7, 8, 9, 10, 11, 12, 13]

def test_validation_catches_bad_input():
    _, e, _ = validate("sales", pd.DataFrame({"sale_date": ["2026-01-01", "oops"], "sku": ["A", "A"], "units_sold": [1, -3]}))
    assert any("negative" in x for x in e) and any("date" in x for x in e)
    _, e, _ = validate("inventory", pd.DataFrame({"sku": ["A"]}))
    assert "missing required" in e[0]

def test_validation_accepts_bundled_samples():
    d = load_bundled()
    for k in ["sales", "inventory", "suppliers", "purchase_orders"]:
        _, e, _ = validate(k, d[k].astype({c: str for c in d[k].columns if "date" in c}) if False else d[k])
        assert not e, (k, e)

def test_generator_reproducible():
    a, b = generate(7), generate(7)
    assert a["sales"].equals(b["sales"]) and a["purchase_orders"].equals(b["purchase_orders"])

def test_demo_mode_without_credentials(monkeypatch=None):
    import os
    from src.database.exasol_client import is_configured, ExasolClient
    for k in ("EXASOL_DSN", "EXASOL_USER", "EXASOL_PASSWORD"): os.environ.pop(k, None)
    assert not is_configured() and ExasolClient().health_check()["ok"] is False

def test_scenarios_alert_while_fixed_rule_silent():
    d = load_bundled(); w = to_wide(d["sales"]); oh = d["inventory"].set_index("sku")["on_hand"].to_dict()
    r = analyze(w, d["inventory"], d["suppliers"], d["purchase_orders"], oh, d["sales"]["sale_date"].max())
    hit = r[(~r.fixed_rule_alert) & r.risk.isin(["Critical", "High"])]
    assert len(hit) >= 2

def test_benchmark_beats_baseline_precision():
    r = run_benchmark(load_bundled())["classification"]
    assert r["StockSage (forecast + incoming POs)"]["precision"] > r["Baseline: coverage < 14 days"]["precision"]
