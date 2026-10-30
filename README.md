# 📦 StockSage — *See the stockout before it happens.*
ExaThon 2026 · Track: Customers, Channels & Demand

StockSage combines sales, inventory, supplier lead times and incoming purchase orders to answer one question per product:
**will it run out before the next replenishment can arrive?** It explains the evidence and recommends *when* and *how much* to reorder.

## Problem
Small retailers learn about shortages when a customer asks for the item. Sales history, stock, supplier delays and open orders live in different places,
so reorder timing is guesswork → missed sales and emergency purchases.

## Features
Command centre KPIs · stockout projection per product · explainable evidence · reorder quantity/date (MOQ + pack size aware) · searchable investigation table ·
what-if simulator (demand, supplier delay, extra stock, service level) · CSV import with validation · session action log · rolling-origin benchmark vs. baselines.

## Honest status
| Item | Status |
|---|---|
| Analytics core, data generator, validation, benchmark | ✅ built, **18 automated tests pass** |
| Streamlit UI (`app.py`) | ⚠️ written, syntax-checked; **not launched by the author** (no Streamlit in the build sandbox) — run it and report any error |
| Exasol integration (`src/database`, `sql/`) | ⚠️ written against the documented `pyexasol` API; **never run against a live Exasol** — verify with `scripts/check_exasol.py` |
| GitHub push / public deployment | ❌ not done; steps below |

## Run it (demo mode, no Exasol needed)
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/run_tests.py        # or: pytest
streamlit run app.py
```
The sidebar badge shows **DEMO DATA**; in this mode every number is computed in pandas/NumPy, *not* in Exasol.

## Exasol Personal (live mode)
1. Install/provision Exasol Personal following Exasol's **current official documentation** (I could not verify its install steps offline — please follow their guide and note the host, port (default 8563), user and password).
2. `cp .env.example .env` and fill `EXASOL_DSN` (`host:8563`), `EXASOL_USER`, `EXASOL_PASSWORD`. Never commit `.env`.
3. `python scripts/check_exasol.py` → expect `{'ok': True, ...}`.
4. `python scripts/load_exasol.py` → creates schema/tables (re-runnable), bulk-loads the dataset, runs `sql/features.sql` inside Exasol.
5. In the app: *Method & Exasol* → **Run SQL feature query in Exasol and compare with pandas** (checks Exasol's rolling demand equals the Python value).

**What actually runs in Exasol:** storage of sales/products/suppliers/POs; `features.sql` (7/28-day demand, std-dev, supplier lateness, open-PO position, coverage);
`rolling_demand.sql` (window functions). **What runs in Python:** forecasting, stockout simulation, risk, replenishment, what-if, benchmark.
If a driver call differs from your Exasol version, the fix is isolated in `src/database/exasol_client.py`.

## Method (all transparent, none is "AI magic")
* **Forecast – WABL:** 60% last-7-day mean + 40% last-28-day mean, times a shrunken weekday index (needs ≥56 days; documented fallbacks otherwise). Statistical, not machine learning.
* **Stockout:** day-by-day simulation of on-hand + expected PO arrivals − forecast. Open-PO dates are shifted by the supplier's *observed* average lateness (past receipts only).
* **Risk:** Critical (stockout before a new order could arrive) / High (within lead + review) / Watch / Low. No stockout *probability* is shown because none is calibrated.
* **Reorder:** order-up-to: demand over (lead+review) + z·σ·√(lead+review) − inventory position → MOQ → pack size. Never negative.
* Money figures are **modelled estimates** (unmet units × price), never "savings".

## Benchmark (synthetic data, 11 weekly origins, 14-day horizon, 396 cases, 40 true stockouts)
| Method | Precision | Recall | False-positive rate |
|---|---|---|---|
| **StockSage** | **0.865** | 0.800 | **0.014** |
| Coverage < 14 days | 0.254 | **0.850** | 0.281 |
| Coverage < 7 days | 0.037 | 0.025 | 0.073 |

Forecast WAPE: WABL 0.163 · 28-day MA 0.189 · seasonal-naive 0.196.
Reading: StockSage raises far fewer false alarms at similar recall; the 14-day coverage rule catches slightly *more* stockouts but ~3.4× more false alerts per true one.
**Limitations:** synthetic data created by the same author; single seed; truth assumes no orders after the origin; censored sales. This is evidence the method behaves sensibly, not proof it works in a real shop.
Reproduce: `python scripts/run_benchmark.py` → `docs/benchmark_results.json`.

## Repository map
`app.py` UI · `src/data` generator/validation/loader · `src/forecasting` models/metrics · `src/inventory` engine · `src/evaluation` benchmark · `src/database` Exasol client · `sql/` · `data/` CSVs · `tests/` · `scripts/` · `docs/`.
See `docs/ARCHITECTURE.md`, `docs/DATA_DICTIONARY.md`, `docs/DEMO_SCRIPT.md`.

## Push to GitHub
```bash
cd stocksage && git init && git add . && git commit -m "StockSage: initial commit"
git branch -M main
# create an EMPTY repo named stocksage on github.com first, then:
git remote add origin https://github.com/<your-username>/stocksage.git
git push -u origin main
```
Check `git status` shows no `.env` before committing.

## Deploy publicly (Streamlit Community Cloud)
Push to GitHub → share.streamlit.io → *New app* → pick repo, branch `main`, file `app.py`. Without secrets it runs in demo mode.
For live Exasol add the `EXASOL_*` values under *Settings → Secrets* (only if your Exasol host is reachable from the internet; a laptop-local instance is not).

## Limitations & future work
Demo data is synthetic · single-location, single-supplier-per-SKU · promotions/prices/substitutes not modelled · overdue POs assumed to arrive in 3 days · benchmark requires stock history (demo only) ·
ideas: calibrated probabilistic forecasts, multi-store, promo features, persistent action log in Exasol, uploaded stock history for backtesting.
