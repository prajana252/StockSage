"""StockSage - See the stockout before it happens.  Run: streamlit run app.py"""
import datetime as dt
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

from src.data.loader import load_bundled, ROOT
from src.data.validation import validate, cross_check
from src.data.generate import FILES
from src.inventory.engine import Params, analyze, to_wide
from src.evaluation.benchmark import run_benchmark
from src.database import exasol_client as xc

load_dotenv()
st.set_page_config(page_title="StockSage", page_icon="📦", layout="wide")
st.markdown("""<style>
.block-container{padding-top:1.4rem;max-width:1400px}
[data-testid=stMetric]{background:#131c2e;border:1px solid #22304a;border-radius:12px;padding:14px 16px}
[data-testid=stMetricLabel]{color:#94a3b8} h1,h2,h3{letter-spacing:-.01em}
.pill{display:inline-block;padding:2px 10px;border-radius:99px;font-size:.78rem;font-weight:600}
.demo{background:#78350f;color:#fde68a}.live{background:#064e3b;color:#6ee7b7}.upl{background:#1e3a8a;color:#bfdbfe}
.note{color:#94a3b8;font-size:.85rem}
@media(max-width:700px){.block-container{padding:1rem .6rem}}
</style>""", unsafe_allow_html=True)
RISK_COLOR = {"Critical": "#ef4444", "High": "#f97316", "Watch": "#f59e0b", "Low": "#10b981"}
RISK_ORDER = ["Critical", "High", "Watch", "Low"]
LAYOUT = dict(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=10, r=10, t=40, b=10))

# ---------------- state ----------------
ss = st.session_state
if "data" not in ss:
    ss.data, ss.source, ss.log = load_bundled(), "demo", []

def restore_demo():
    ss.data, ss.source = load_bundled(), "demo"

def current_asof():
    return ss.data["sales"]["sale_date"].max()

# ---------------- sidebar ----------------
with st.sidebar:
    st.markdown("## 📦 StockSage")
    st.caption("See the stockout before it happens.")
    label = {"demo": ("DEMO DATA (synthetic, bundled)", "demo"), "upload": ("YOUR UPLOADED CSVs", "upl"), "exasol": ("LIVE EXASOL", "live")}[ss.source]
    st.markdown(f"<span class='pill {label[1]}'>{label[0]}</span>", unsafe_allow_html=True)
    st.caption(f"As-of date: **{current_asof().date()}**")
    st.markdown("#### Policy assumptions")
    z = st.select_slider("Service level", [0.84, 1.28, 1.65, 2.05, 2.33], 1.65, format_func=lambda v: {0.84: "80%", 1.28: "90%", 1.65: "95%", 2.05: "98%", 2.33: "99%"}[v])
    review = st.slider("Review period (days)", 1, 14, 7)
    horizon = st.slider("Stockout scan horizon (days)", 14, 56, 28)
    st.markdown("#### Exasol")
    st.caption("Configured via environment variables." if xc.is_configured() else "Not configured → running in demo mode.")
    if st.button("Test Exasol connection"):
        h = xc.ExasolClient().health_check(); (st.success if h["ok"] else st.warning)(h["detail"])
    if ss.source != "demo": st.button("↩ Restore demo dataset", on_click=restore_demo)

base = Params(horizon=horizon, review_days=review, service_z=z)
d = ss.data; asof = current_asof()
wide = to_wide(d["sales"]); oh = d["inventory"].set_index("sku")["on_hand"].to_dict()
res, paths = analyze(wide, d["inventory"], d["suppliers"], d["purchase_orders"], oh, asof, base, with_paths=True)

tabs = st.tabs(["Command centre", "Investigate", "What-if", "Benchmark", "Data import", "Action log", "Method & Exasol"])

# ---------------- command centre ----------------
with tabs[0]:
    st.title("Command centre")
    st.caption(f"Predictions as of {asof.date()}. Money figures are **model estimates**, not realised savings.")
    urgent = res[res.risk.isin(["Critical", "High"])]
    prior = res.avg_28.sum(); recent = res.avg_7.sum()
    c = st.columns(5)
    c[0].metric("Active products", len(res))
    c[1].metric("High/critical risk", len(urgent))
    c[2].metric("Est. sales at risk (28d)", f"{res.sales_at_risk.sum():,.0f}", help="Unmet forecast units × unit price if no further orders arrive. A modelled estimate.")
    c[3].metric("Need replenishment", int((res.reorder_qty > 0).sum()))
    finite = res.coverage_days.replace(np.inf, np.nan)
    c[4].metric("Median coverage (days)", f"{finite.median():.1f}")
    st.caption(f"Portfolio demand: last 7 days {recent:,.0f}/day vs 28-day average {prior:,.0f}/day ({(recent/prior-1)*100:+.1f}%).")
    a, b = st.columns([1, 2])
    cnt = res.risk.value_counts().reindex(RISK_ORDER).fillna(0)
    fig = go.Figure(go.Bar(x=cnt.index, y=cnt.values, marker_color=[RISK_COLOR[k] for k in cnt.index]))
    fig.update_layout(title="Products by risk", **LAYOUT, height=340); a.plotly_chart(fig, use_container_width=True)
    top = urgent.sort_values("stockout_day").head(12)
    if top.empty: b.info("No critical or high-risk products under the current assumptions.")
    else:
        fig = go.Figure()
        fig.add_bar(y=top.product_name, x=top.stockout_day, orientation="h", name="Days to stockout", marker_color="#ef4444")
        fig.add_scatter(y=top.product_name, x=top.lead_time_days, mode="markers", name="Effective lead time", marker=dict(color="#e6edf7", size=11, symbol="diamond"))
        fig.update_layout(title="Will it run out before a new order can arrive?", **LAYOUT, height=340, yaxis=dict(autorange="reversed"))
        b.plotly_chart(fig, use_container_width=True)
    st.subheader("Silent under a simple rule, risky under StockSage")
    sil = res[(~res.fixed_rule_alert) & res.risk.isin(["Critical", "High"])]
    if sil.empty: st.caption("No such products in the current data.")
    else: st.dataframe(sil[["sku", "product_name", "on_hand", "coverage_days", "lead_time_days", "stockout_day", "reorder_qty", "evidence"]].round(1), use_container_width=True, hide_index=True)
    st.caption("A '7-day coverage' rule stays silent on these because stock ≥ 7 days of recent demand, yet lead time, rising demand and/or late suppliers imply a shortage.")

# ---------------- investigate ----------------
with tabs[1]:
    st.title("Product investigation")
    f1, f2, f3, f4 = st.columns([2, 1, 1, 1])
    q = f1.text_input("Search product or SKU")
    cats = f2.multiselect("Category", sorted(res.category.unique())); rk = f3.multiselect("Risk", RISK_ORDER)
    sort = f4.selectbox("Sort by", ["Stockout risk", "Coverage (days)", "Reorder urgency"])
    v = res.copy()
    if q: v = v[v.sku.str.contains(q, case=False) | v.product_name.str.contains(q, case=False)]
    if cats: v = v[v.category.isin(cats)]
    if rk: v = v[v.risk.isin(rk)]
    v = v.assign(_r=v.risk.map({k: i for i, k in enumerate(RISK_ORDER)}), _s=v.stockout_day.fillna(999), _o=v.order_date.fillna(pd.Timestamp("2100-01-01")))
    v = v.sort_values({"Stockout risk": ["_r", "_s"], "Coverage (days)": ["coverage_days"], "Reorder urgency": ["_o"]}[sort])
    if v.empty: st.info("No products match these filters.")
    else:
        st.dataframe(v[["sku", "product_name", "category", "risk", "on_hand", "coverage_days", "lead_time_days", "stockout_date", "reorder_qty", "order_date"]].round(1), use_container_width=True, hide_index=True)
        sel = st.selectbox("Open product", v.sku, format_func=lambda s: f"{s} · {res.set_index('sku').loc[s, 'product_name']}")
        r = res.set_index("sku").loc[sel]; pth = paths[sel]
        st.subheader(f"{r.product_name}  ·  {r.risk}")
        m = st.columns(5)
        m[0].metric("On hand", f"{r.on_hand:.0f}"); m[1].metric("Avg demand (28d)", f"{r.avg_28:.1f}/day"); m[2].metric("Effective lead time", f"{r.lead_time_days} d")
        m[3].metric("Stockout", str(r.stockout_date.date()) if pd.notna(r.stockout_date) else f"none ≤ {horizon}d"); m[4].metric("Open PO units", f"{r.open_po_qty:.0f}")
        st.markdown("**Evidence (all derived from this product's data)**")
        for e in pth["evidence"]: st.markdown(f"- {e}")
        hist = wide[sel].iloc[-90:]; fdates = pd.date_range(asof + pd.Timedelta(days=1), periods=len(pth["f"]))
        g1, g2 = st.columns(2)
        fig = go.Figure(); fig.add_scatter(x=hist.index, y=hist.values, name="Actual sales", line=dict(color="#38bdf8"))
        fig.add_scatter(x=fdates, y=pth["f"], name="Forecast", line=dict(color="#10b981", dash="dash"))
        fig.update_layout(title=f"Demand: history & forecast ({r.method})", **LAYOUT, height=340); g1.plotly_chart(fig, use_container_width=True)
        stock = pth["on_hand"] + np.cumsum(pth["arr"] - pth["f"]); stock_floor = np.maximum(stock, 0)
        fig = go.Figure(); fig.add_scatter(x=fdates, y=stock, name="Projected stock (no new orders)", line=dict(color="#f59e0b"))
        fig.add_hline(y=0, line_color="#ef4444"); fig.add_vline(x=asof + pd.Timedelta(days=int(r.lead_time_days)), line_dash="dot", line_color="#e6edf7", annotation_text="New order could arrive")
        fig.update_layout(title="Projected inventory (negative = unmet demand)", **LAYOUT, height=340); g2.plotly_chart(fig, use_container_width=True)
        st.markdown("**Recommendation**")
        if r.reorder_qty > 0:
            st.success(f"Order **{int(r.reorder_qty)} units** on **{r.order_date.date()}** → expected **{r.delivery_date.date()}**; projected stock after delivery ≈ {r.projected_after_delivery:.0f}. "
                       f"Safety stock ≈ {r.safety_stock:.0f} units (z={z}, σ from last 28d, lead time + review = {r.lead_time_days + review} d).")
            if pd.notna(r.stockout_day) and r.stockout_day <= r.lead_time_days: st.warning("This order cannot land before the projected stockout: consider expediting or a partial local purchase.")
            if st.button("Log this recommendation", key="log_" + sel):
                ss.log.append(dict(time=dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), sku=sel, product=r.product_name, risk=r.risk, evidence=r.evidence,
                                   action=f"Order {int(r.reorder_qty)} on {r.order_date.date()}", status="Open")); st.toast("Logged (this session only)")
        else: st.info("No order needed under current assumptions.")
        po = d["purchase_orders"]; po = po[(po.sku == sel) & (po.order_date <= asof)].sort_values("order_date").tail(6)
        st.markdown("**Recent purchase orders & supplier**"); st.dataframe(po, use_container_width=True, hide_index=True)
        st.caption(f"Supplier {r.supplier_id}: estimated average lateness {r.supplier_delay_est:.1f} day(s) from received POs up to the as-of date.")

# ---------------- what-if ----------------
with tabs[2]:
    st.title("What-if simulator")
    st.caption("Changes are applied to a copy of the calculation. Source data is never modified.")
    w1, w2, w3, w4 = st.columns(4)
    dm = w1.slider("Demand change", -50, 100, 0, format="%d%%"); dd = w2.slider("Extra supplier delay (days)", 0, 21, 0)
    sk = w3.selectbox("Product for extra stock", res.sku, format_func=lambda s: f"{s} · {res.set_index('sku').loc[s, 'product_name']}", key="wi_sku")
    ex = w4.number_input("Extra units received", 0, 100000, 0)
    zz = st.select_slider("Safety-stock service level (scenario)", [0.84, 1.28, 1.65, 2.05, 2.33], z, format_func=lambda v: {0.84: "80%", 1.28: "90%", 1.65: "95%", 2.05: "98%", 2.33: "99%"}[v])
    sc = analyze(wide, d["inventory"], d["suppliers"], d["purchase_orders"], oh, asof,
                 Params(horizon=horizon, review_days=review, service_z=zz, demand_mult=1 + dm / 100, extra_delay=dd, extra_stock={sk: ex}))
    k = st.columns(3)
    for col, name in zip(k, ["Critical", "High", "Watch"]):
        col.metric(f"{name} products", int((sc.risk == name).sum()), int((sc.risk == name).sum() - (res.risk == name).sum()), delta_color="inverse")
    cmp_ = res.set_index("sku")[["risk", "stockout_day", "reorder_qty"]].join(sc.set_index("sku")[["risk", "stockout_day", "reorder_qty"]], rsuffix="_scenario")
    chg = cmp_[(cmp_.risk != cmp_.risk_scenario) | (cmp_.reorder_qty != cmp_.reorder_qty_scenario)]
    st.markdown(f"**{len(chg)} product(s) change** under this scenario"); st.dataframe(chg.join(res.set_index("sku").product_name), use_container_width=True)

# ---------------- benchmark ----------------
with tabs[3]:
    st.title("Benchmark: does it actually work?")
    st.markdown("""**Method.** Rolling-origin backtest on the bundled *synthetic* dataset: origins every 7 days across the final 90 days. At each origin only data up to that date is used
(sales, stock, purchase orders received so far). **Actual stockout** = with the stock and open POs known at the origin, *true* demand over the next 14 days exceeds supply (no further orders assumed).
**Alert** = predicted stockout within 14 days. Baselines ignore lead time, incoming orders, weekday pattern and trend.""")
    if ss.source != "demo": st.info("The benchmark needs true stock history and runs on the bundled demo dataset only.")
    @st.cache_data(show_spinner="Running backtest…")
    def bench(): 
        out = run_benchmark(load_bundled()); return out
    if st.button("Run / refresh benchmark") or "bench" in ss:
        ss.bench = bench(); b = ss.bench
        mt = b["meta"]; st.caption(f"{mt['origins']} origins ({mt['first_origin']} → {mt['last_origin']}), {mt['alert_cases']} product-origin cases, {mt['actual_stockout_cases']} actual stockouts.")
        cl = pd.DataFrame(b["classification"]).T; st.dataframe(cl.round(3), use_container_width=True)
        fm = pd.DataFrame(b["forecast"]).T.rename(index={"wabl": "WABL (StockSage)", "ma": "28-day moving average", "snaive": "Seasonal naive"})
        st.markdown("**Forecast error, 14-day horizon (daily units)**"); st.dataframe(fm.round(3), use_container_width=True)
        fig = go.Figure(); names = list(b["classification"])
        for metric in ["precision", "recall", "false_positive_rate"]: fig.add_bar(name=metric, x=names, y=[b["classification"][n][metric] for n in names])
        fig.update_layout(barmode="group", **LAYOUT, height=340); st.plotly_chart(fig, use_container_width=True)
    st.warning("**Limitations:** synthetic data generated by the same author; a rule-based risk engine, not a calibrated probability model; "
               "truth ignores orders placed after the origin; sales are censored when stock hits zero. Results show the method works *on this data*, not that it will in your shop.")

# ---------------- data import ----------------
with tabs[4]:
    st.title("Data import")
    st.caption("Upload your own CSVs. Every file is validated; nothing is used until all pass.")
    req = {"sales": "sale_date, sku, units_sold", "inventory": "sku, product_name, category, on_hand, unit_price, supplier_id (optional: unit_cost, moq, pack_size)",
           "suppliers": "supplier_id, supplier_name, lead_time_days", "purchase_orders": "po_id, sku, order_date, expected_date, quantity (optional: received_date, blank = still open)"}
    ups = {}
    for name, cols in req.items():
        c1, c2 = st.columns([3, 1]); c1.markdown(f"**{name}** — `{cols}`")
        with open(os.path.join(ROOT, "data", FILES[name]), "rb") as fh: c2.download_button("Sample CSV", fh.read(), FILES[name], key="dl_" + name)
        ups[name] = st.file_uploader(f"{name}.csv", type="csv", key="up_" + name, label_visibility="collapsed")
    if st.button("Validate & use uploaded files"):
        if not all(ups.values()): st.error("Upload all four files first (sales, inventory, suppliers, purchase_orders).")
        else:
            clean, errs, warns = {}, [], []
            for n, f in ups.items():
                try: df, e, w = validate(n, pd.read_csv(f))
                except Exception as ex_: df, e, w = None, [f"{n}: could not read CSV ({ex_})"], []
                clean[n] = df; errs += e; warns += w
            if not errs: e2, w2 = cross_check(clean); errs += e2; warns += w2
            for w_ in warns: st.warning(w_)
            if errs:
                for e in errs: st.error(e)
            else:
                clean["inventory"] = clean["inventory"][clean["inventory"].sku.notna()]
                clean["sales"] = clean["sales"][clean["sales"].sku.isin(clean["inventory"].sku)]
                clean["stock_history"] = pd.DataFrame(columns=["date", "sku", "on_hand", "unmet_units"]); clean["product_meta"] = pd.DataFrame(columns=["sku", "demand_profile"])
                ss.data, ss.source = clean, "upload"; st.success("Validated. Uploaded data is now active (session only)."); st.rerun()
    st.button("↩ Restore bundled demo dataset", on_click=restore_demo, key="restore2")

# ---------------- action log ----------------
with tabs[5]:
    st.title("Evidence & action log")
    st.warning("**Session-only log.** Entries live in this browser session and are lost on refresh. Download the CSV to keep them.")
    if not ss.log: st.info("Nothing logged yet. Open a product in *Investigate* and click *Log this recommendation*.")
    else:
        for i, e in enumerate(ss.log):
            c1, c2 = st.columns([5, 1]); c1.markdown(f"**{e['sku']} · {e['product']}** — {e['risk']} — {e['action']}  \n<span class='note'>{e['time']} · {e['evidence']}</span>", unsafe_allow_html=True)
            ss.log[i]["status"] = c2.selectbox("Status", ["Open", "Acknowledged", "Resolved"], ["Open", "Acknowledged", "Resolved"].index(e["status"]), key=f"st{i}")
        st.download_button("Download log (CSV)", pd.DataFrame(ss.log).to_csv(index=False), "stocksage_action_log.csv")

# ---------------- method & exasol ----------------
with tabs[6]:
    st.title("Method & Exasol")
    cfg = xc.config_from_env()
    st.markdown(f"**Mode:** {'Exasol credentials present' if xc.is_configured() else 'Demo mode (no Exasol credentials) — all calculations below run in pandas/NumPy, not Exasol'}.")
    st.markdown("""**Pipeline:** CSVs → validation → *(live mode)* Exasol tables → SQL features (rolling demand, supplier lateness, open POs) → Python forecasting (WABL) →
stockout projection → replenishment policy → dashboard.  
**Forecast:** weekday-adjusted blended level (60% last-7-day mean + 40% last-28-day mean) × shrunken weekday index. A transparent statistical model, **not machine learning**.  
**Stockout:** simulate day by day: on-hand + expected PO arrivals (shifted by the supplier's observed average lateness) − forecast demand.  
**Risk:** Critical = stockout before a new order could arrive; High = within lead time + review period; Watch = later within horizon, or order due now; Low = none.  
**Reorder:** order-up-to policy: forecast demand over (lead time + review) + z·σ·√(lead+review) − inventory position, rounded up to MOQ and pack size. No probability of stockout is shown because none has been calibrated.""")
    for f_ in ["features.sql", "rolling_demand.sql", "schema.sql"]:
        with st.expander(f"sql/{f_}"): st.code(open(os.path.join(ROOT, "sql", f_)).read(), language="sql")
    if xc.is_configured() and st.button("Run SQL feature query in Exasol and compare with pandas"):
        try:
            cl = xc.ExasolClient().connect(); fx = cl.sql_features(str(asof.date())); fx.columns = [c.lower() for c in fx.columns]
            cmpdf = fx.merge(res[["sku", "avg_28"]], on="sku", suffixes=("_exasol", "_pandas")); cmpdf["abs_diff"] = (cmpdf.avg_28_exasol.astype(float) - cmpdf.avg_28_pandas).abs()
            st.success(f"Exasol executed features.sql. Max |avg_28 difference| vs pandas: {cmpdf.abs_diff.max():.6f}"); st.dataframe(fx, use_container_width=True)
        except Exception as e: st.error(f"Exasol query failed: {e}")
