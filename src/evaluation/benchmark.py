"""Rolling-origin backtest: forecasts and stockout alerts are built only from data available at each origin."""
import numpy as np
import pandas as pd
from src.forecasting.models import forecast_wabl, forecast_ma28, forecast_seasonal_naive, mae, wape
from src.inventory.engine import Params, analyze, simulate_stock, to_wide

H_EVAL = 14


def run_benchmark(d: dict, test_days: int = 90, step: int = 7, horizon: int = H_EVAL, params: Params = None):
    sales_w = to_wide(d["sales"])
    sh = d["stock_history"]
    demand = sh.merge(d["sales"], left_on=["date", "sku"], right_on=["sale_date", "sku"])
    demand["demand"] = demand["units_sold"] + demand["unmet_units"]
    dem_w = demand.pivot(index="date", columns="sku", values="demand")
    oh_w = sh.pivot(index="date", columns="sku", values="on_hand")
    pos = d["purchase_orders"]; prods = d["inventory"]; sup = d["suppliers"]
    end = sales_w.index.max()
    origins = pd.date_range(end - pd.Timedelta(days=test_days), end - pd.Timedelta(days=horizon), freq=f"{step}D")
    p = params or Params(horizon=horizon)
    fc_rows, al_rows = [], []
    for t in origins:
        on_hand = oh_w.loc[t].to_dict()
        res = analyze(sales_w, prods, sup, pos, on_hand, t, p).set_index("sku")
        for sku in prods["sku"]:
            hist = sales_w.loc[:t, sku]
            actual = dem_w.loc[t + pd.Timedelta(days=1): t + pd.Timedelta(days=horizon), sku].to_numpy()
            f_w = forecast_wabl(hist, horizon)[0]
            fc_rows.append(dict(origin=t, sku=sku, a=actual, wabl=f_w, ma=forecast_ma28(hist, horizon), snaive=forecast_seasonal_naive(hist, horizon)))
            arrivals = np.zeros(horizon)
            q = pos[(pos["sku"] == sku) & (pos["order_date"] <= t) & pos["received_date"].notna()
                    & (pos["received_date"] > t) & (pos["received_date"] <= t + pd.Timedelta(days=horizon))]
            for x in q.itertuples(): arrivals[(x.received_date - t).days - 1] += x.quantity
            truth, _ = simulate_stock(on_hand[sku], actual.astype(float), arrivals)
            r = res.loc[sku]
            ss = r["stockout_day"] if (r["stockout_day"] == r["stockout_day"] and r["stockout_day"] is not None and r["stockout_day"] <= horizon) else None
            al_rows.append(dict(origin=t, sku=sku, truth_day=truth, stocksage=ss is not None, ss_day=ss,
                                cov7=bool(r["avg_28"] > 0 and on_hand[sku] / r["avg_28"] < 7),
                                cov14=bool(r["avg_28"] > 0 and on_hand[sku] / r["avg_28"] < 14)))
    fc, al = pd.DataFrame(fc_rows), pd.DataFrame(al_rows)
    fm = {}
    for m in ["wabl", "ma", "snaive"]:
        A = np.concatenate(fc["a"].to_numpy()); P = np.concatenate(fc[m].to_numpy())
        fm[m] = dict(MAE=mae(A, P), WAPE=wape(A, P))
    al["actual"] = al["truth_day"].notna()
    cls = {}
    for name, col in [("StockSage (forecast + incoming POs)", "stocksage"), ("Baseline: coverage < 14 days", "cov14"), ("Baseline: coverage < 7 days", "cov7")]:
        tp = int((al[col] & al["actual"]).sum()); fp = int((al[col] & ~al["actual"]).sum())
        fn = int((~al[col] & al["actual"]).sum()); tn = int((~al[col] & ~al["actual"]).sum())
        hit = al[al[col] & al["actual"]]
        cls[name] = dict(TP=tp, FP=fp, FN=fn, TN=tn,
                         precision=tp / (tp + fp) if tp + fp else float("nan"), recall=tp / (tp + fn) if tp + fn else float("nan"),
                         false_positive_rate=fp / (fp + tn) if fp + tn else float("nan"),
                         mean_days_warning=float(hit["truth_day"].mean()) if len(hit) else float("nan"))
    meta = dict(origins=len(origins), horizon_days=horizon, test_days=test_days, step_days=step,
                alert_cases=int(len(al)), actual_stockout_cases=int(al["actual"].sum()),
                first_origin=str(origins[0].date()), last_origin=str(origins[-1].date()))
    return dict(forecast=fm, classification=cls, meta=meta, alerts=al)
